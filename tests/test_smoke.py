"""Offline smoke tests. Run from the project folder:  python -m unittest -v tests.test_smoke

Network calls (HumAID and every live source, Groq, Gemini) are mocked, so these pass with no internet.
They check parsing and graceful-failure logic, not the live services themselves.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import requests

from src.ingestion import curated_loader, live_sources
from src.ingestion.curated_loader import dataframe_to_reports, load_sample_data
from src.nlp.classifier import HumanitarianClassifier
from src.nlp.ner import extract_entities, extract_locations
from src.nlp.processor import CrisisProcessor
from src.rag import generator
from src.rag.indexer import CrisisIndex
from src.rag.retriever import CommanderRetriever


def _fake_response(status=200, json_data=None, content=b""):
    resp = mock.Mock()
    resp.status_code = status
    resp.json.return_value = json_data
    resp.content = content
    resp.text = str(json_data)[:200]
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


class SharedSystem:
    _cache = None

    @classmethod
    def get(cls):
        if cls._cache is None:
            df = load_sample_data()
            clf = HumanitarianClassifier()
            clf.train(df["text"].tolist(), df["label"].tolist())
            proc = CrisisProcessor(clf)
            reports = proc.process(dataframe_to_reports(df, "sample"))
            cls._cache = (proc, CrisisIndex.build(reports))
        return cls._cache


class TestNLP(unittest.TestCase):
    def test_spec_example(self):
        proc, _ = SharedSystem.get()
        r = proc.process_text("4 people trapped on the second floor near XYZ bridge, need rescue")
        self.assertEqual(r.primary_location, "XYZ bridge")
        self.assertEqual(r.primary_need, "Rescue")
        self.assertEqual(r.people_count, 4)
        self.assertEqual(r.priority, "Critical")

    def test_entities(self):
        e = extract_entities("Two people dead and 11 injured after a wall collapsed in Saidapet")
        self.assertEqual(e.people_count, 13)
        self.assertIn("Saidapet", e.locations)
        self.assertEqual(extract_entities("family of 6 stuck on the roof").people_count, 6)
        self.assertEqual(extract_locations("what is happening near xyz bridge right now", lenient=True), ["xyz bridge"])

    def test_tamil_message_is_translated_then_scored(self):
        from src.nlp import processor as processor_module
        proc, _ = SharedSystem.get()
        tamil = "வேளச்சேரியில் வெள்ளம் காரணமாக 5 பேர் கொண்ட குடும்பம் வீட்டின் மேற்கூரையில் சிக்கியுள்ளது, படகு தேவை."
        english = "A family of 5 is trapped on the roof of their house due to flooding in Velachery; a boat is needed."
        with mock.patch.object(processor_module, "translate_to_english", return_value=({tamil: (english, "Tamil")}, "")):
            r = proc.process_text(tamil)
        self.assertEqual(r.text, english + " [translated from Tamil]")
        self.assertEqual((r.original_text, r.language), (tamil, "Tamil"))
        self.assertEqual((r.primary_location, r.people_count, r.primary_need, r.priority), ("Velachery", 5, "Rescue", "Critical"))
        self.assertEqual(proc.process_text("Power cut in Tambaram").original_text, "")  # English is left alone

    def test_low_priority_chatter(self):
        proc, _ = SharedSystem.get()
        self.assertEqual(proc.process_text("Perfect rainy day for hot bajji and filter coffee").priority, "Low")

    def test_resolved_lowers_urgency(self):
        proc, _ = SharedSystem.get()
        a = proc.process_text("Missing boy from Saidapet, please help find him")
        b = proc.process_text("Update: the missing boy from Saidapet has been found safe")
        self.assertGreater(a.urgency_score, b.urgency_score)


class TestRAG(unittest.TestCase):
    def test_location_query(self):
        _, index = SharedSystem.get()
        results, plan = CommanderRetriever(index).retrieve("most critical situation near XYZ bridge right now", k=3)
        self.assertTrue(plan.location_filter_applied)
        self.assertTrue(all("xyz bridge" in r.report.text.lower() for r in results))
        self.assertEqual(results[0].report.priority, "Critical")

    def test_index_roundtrip(self):
        _, index = SharedSystem.get()
        with tempfile.TemporaryDirectory() as d:
            index.save(Path(d))
            loaded = CrisisIndex.load(Path(d))
            self.assertEqual(len(loaded), len(index))
            self.assertEqual(loaded.search("trapped rescue", 1)[0][0], index.search("trapped rescue", 1)[0][0])

    def test_extractive_without_keys(self):
        _, index = SharedSystem.get()
        results, plan = CommanderRetriever(index).retrieve("who needs rescue?", k=3)
        with mock.patch.object(generator, "GROQ_API_KEY", ""), mock.patch.object(generator, "GEMINI_API_KEY", ""):
            ans = generator.generate_answer("who needs rescue?", results, plan, provider="auto")
        self.assertEqual(ans.provider, "extractive")
        self.assertIn("[1]", ans.render())

    def test_groq_and_gemini_parsing(self):
        _, index = SharedSystem.get()
        results, plan = CommanderRetriever(index).retrieve("who needs rescue?", k=3)
        groq_json = {"choices": [{"message": {"content": "Family stuck on roof near XYZ bridge [1]."}}]}
        gem_json = {"candidates": [{"content": {"parts": [{"text": "Rescue needed at XYZ bridge [1]."}]}}]}
        with mock.patch.object(generator, "GROQ_API_KEY", "k"), \
                mock.patch.object(generator.requests, "post", return_value=_fake_response(200, groq_json)):
            ans = generator.generate_answer("q", results, plan, provider="groq")
        self.assertTrue(ans.provider.startswith("groq:"))
        self.assertIn("[1]", ans.text)
        with mock.patch.object(generator, "GEMINI_API_KEY", "k"), \
                mock.patch.object(generator.requests, "post", return_value=_fake_response(200, gem_json)):
            ans = generator.generate_answer("q", results, plan, provider="gemini")
        self.assertTrue(ans.provider.startswith("gemini:"))

    def test_llm_failure_falls_back(self):
        _, index = SharedSystem.get()
        results, plan = CommanderRetriever(index).retrieve("who needs rescue?", k=3)
        with mock.patch.object(generator, "GROQ_API_KEY", "k"), \
                mock.patch.object(generator.requests, "post", side_effect=requests.Timeout()):
            ans = generator.generate_answer("q", results, plan, provider="groq")
        self.assertEqual(ans.provider, "extractive")
        self.assertIn("timed out", ans.notice)


class TestIngestion(unittest.TestCase):
    RSS = b"""<?xml version="1.0"?><rss><channel>
      <item><title>Red alert: flood in Assam</title><description>&lt;p&gt;Severe flooding&lt;/p&gt;</description>
      <link>https://example.org/1</link><pubDate>Sat, 10 Oct 2026 06:00:00 GMT</pubDate></item>
    </channel></rss>"""

    def test_gdacs_parse_and_timeout(self):
        with mock.patch.object(live_sources.requests, "get", return_value=_fake_response(200, content=self.RSS)):
            reports, status = live_sources.fetch_gdacs()
        self.assertTrue(status.ok)
        self.assertEqual(reports[0].text, "Red alert: flood in Assam. Severe flooding")
        with mock.patch.object(live_sources.requests, "get", side_effect=requests.Timeout()):
            reports, status = live_sources.fetch_gdacs()
        self.assertFalse(status.ok)
        self.assertIn("timed out", status.message)

    def test_ndma_keeps_english_alerts(self):
        feed = """<?xml version="1.0"?><rss><channel>
          <item><title>Heavy Rain is likely to occur at isolated places over Bastar in next 3 hours. ..</title>
          <author>controlroom@ndma.gov.in (IMD Raipur)</author><link>https://example.org/a</link>
          <pubDate>Tue, 27 Oct 2026 11:21:00 GMT</pubDate></item>
          <item><title>మీ ప్రాంతంలో పిడుగులు పడే అవకాశం ఉంది.</title><author>x (Andhra Pradesh SDMA)</author></item>
          <item><title>KHAWCHIN: Lunglei leh Mamit District hmun hrang hrangah, Ruah sur leh Tek tla awm thei a ni.</title></item>
        </channel></rss>""".encode("utf-8")
        no_translation = ({}, "no GROQ_API_KEY or GEMINI_API_KEY to translate with")
        with mock.patch.object(live_sources.requests, "get", return_value=_fake_response(200, content=feed)), \
                mock.patch.object(live_sources, "translate_to_english", return_value=no_translation):
            reports, status = live_sources.fetch_ndma()
        self.assertTrue(status.ok)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].source, "ndma/IMD Raipur")
        self.assertEqual(reports[0].text, "Heavy Rain is likely to occur at isolated places over Bastar in next 3 hours.")
        self.assertIn("skipped 2", status.message)
        self.assertIn("no GROQ_API_KEY", status.message)
        # with an LLM available, the Telugu alert is translated and tagged instead of skipped
        telugu = "మీ ప్రాంతంలో పిడుగులు పడే అవకాశం ఉంది."
        translated = ({telugu: ("There is a possibility of lightning in your area.", "Telugu")}, "")
        with mock.patch.object(live_sources.requests, "get", return_value=_fake_response(200, content=feed)), \
                mock.patch.object(live_sources, "translate_to_english", return_value=translated):
            reports, status = live_sources.fetch_ndma()
        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[1].text, "There is a possibility of lightning in your area. [translated from Telugu]")
        self.assertEqual((reports[1].original_text, reports[1].language), (telugu, "Telugu"))
        self.assertEqual(reports[0].original_text, "")  # English alerts are not translated
        self.assertEqual(reports[1].source, "ndma/Andhra Pradesh SDMA")
        self.assertIn("translated 1 (1 Telugu)", status.message)

    def test_translator_batches_parses_and_caches(self):
        from src.ingestion import translate
        replies = ['Here you go: [{"i": 0, "language": "Hindi", "english": "Heavy rain likely in Patna."}]']
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(translate, "CACHE_PATH", Path(d) / "t.json"), \
                mock.patch.object(translate, "TRANSLATE_REGIONAL", "auto"), \
                mock.patch.object(translate, "GROQ_API_KEY", "k"), \
                mock.patch.object(generator, "_call_groq", side_effect=replies) as call:
            first, note = translate.translate_to_english(["पटना में भारी बारिश की संभावना", "पटना में भारी बारिश की संभावना"])
            again, _ = translate.translate_to_english(["पटना में भारी बारिश की संभावना"])
        self.assertEqual(call.call_count, 1)  # the duplicate was sent once, and the second call came from the cache
        self.assertEqual(first["पटना में भारी बारिश की संभावना"], ("Heavy rain likely in Patna.", "Hindi"))
        self.assertEqual(again, first)
        self.assertEqual(note, "")
        with mock.patch.object(translate, "TRANSLATE_REGIONAL", "off"):
            out, note = translate.translate_to_english(["কলকাতায় ঝড়"])
        self.assertEqual(out, {})
        self.assertIn("off", note)

    def test_restore_originals_from_translation_cache(self):
        from src.ingestion import translate
        from src.schema import Report
        old = Report(id="ndma-1", text="Heavy rain likely in Patna. [translated from Hindi]", source="ndma")
        english = Report(id="ndma-2", text="Heavy Rain likely over Bastar.", source="ndma")
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "t.json"
            cache.write_text('{"पटना में भारी बारिश की संभावना": ["Heavy rain likely in Patna.", "Hindi"]}', encoding="utf-8")
            with mock.patch.object(translate, "CACHE_PATH", cache):
                self.assertEqual(translate.restore_originals([old, english]), 1)
        self.assertEqual((old.original_text, old.language), ("पटना में भारी बारिश की संभावना", "Hindi"))
        self.assertEqual(english.original_text, "")

    def test_google_news_headlines(self):
        feed = b"""<?xml version="1.0"?><rss><channel>
          <item><title>Flood alert in Assam as Brahmaputra rises - The Hindu</title><source url="https://www.thehindu.com">The Hindu</source>
          <link>https://news.google.com/rss/articles/a</link><pubDate>Fri, 30 Oct 2026 06:27:10 GMT</pubDate></item>
          <item><title>Cyclone nears Odisha coast, 1 lakh evacuated - NDTV</title><source url="https://www.ndtv.com">NDTV</source>
          <link>https://news.google.com/rss/articles/b</link><pubDate>Fri, 30 Oct 2026 07:00:00 GMT</pubDate></item>
        </channel></rss>"""
        with mock.patch.object(live_sources.requests, "get", return_value=_fake_response(200, content=feed)) as get:
            reports, status = live_sources.fetch_google_news(queries=["flood India", "cyclone India"])
        self.assertEqual(get.call_count, 2)
        self.assertTrue(status.ok)
        self.assertEqual(len(reports), 2)  # the same two headlines came back for both searches
        self.assertEqual(reports[0].text, "Flood alert in Assam as Brahmaputra rises")
        self.assertEqual(reports[0].source, "news/The Hindu")

    def test_bluesky_keeps_english_disaster_posts(self):
        def post(rkey, text):
            return {"uri": f"at://did:plc:x/app.bsky.feed.post/{rkey}", "author": {"handle": "someone.bsky.social"},
                    "record": {"text": text, "createdAt": "2026-10-30T09:21:12.631Z", "langs": ["en"]}}
        posts = {"posts": [
            post("k1", "Water is rising fast in our street in Silchar, three families are trapped and need a boat"),
            post("k2", "A landslide of repudiation."),
            post("k3", "Wordle 1,959 3/6 and the coffee was great this morning, love it"),
            post("k4", "Nuevo sismo registrado: earthquake de magnitud 4.1 cerca de la costa hoy"),
        ]}
        refused = _fake_response(403, {"error": "forbidden"})
        with mock.patch.object(live_sources.requests, "get", side_effect=[refused, _fake_response(200, posts)]):
            reports, status = live_sources.fetch_bluesky(queries=["flood India"])
        self.assertTrue(status.ok)
        self.assertEqual([r.id for r in reports], ["bluesky-k1"])
        self.assertEqual(reports[0].url, "https://bsky.app/profile/someone.bsky.social/post/k1")
        self.assertIn("skipped 3", status.message)
        with mock.patch.object(live_sources, "BLUESKY_HANDLE", ""), \
                mock.patch.object(live_sources.requests, "get", return_value=refused):
            reports, status = live_sources.fetch_bluesky(queries=["flood India"])
        self.assertFalse(status.ok)
        self.assertIn("BLUESKY_HANDLE", status.message)

    def test_india_news_keeps_disaster_stories(self):
        feed = b"""<?xml version="1.0"?><rss><channel>
          <item><title>Cyclone Arnab: 10,000 in relief camps in Ichchapuram</title><description>NH-16 still under water.</description>
          <link>https://www.thehindu.com/a</link><pubDate>Fri, 30 Oct 2026 13:35:00 +0530</pubDate></item>
          <item><title>High Court appoints board for temple</title><description>Ruling on administration.</description></item>
        </channel></rss>"""
        with mock.patch.object(live_sources.requests, "get", return_value=_fake_response(200, content=feed)):
            reports, status = live_sources.fetch_india_news(feeds=["https://www.thehindu.com/news/national/feeder/default.rss"])
        self.assertEqual([r.text for r in reports], ["Cyclone Arnab: 10,000 in relief camps in Ichchapuram. NH-16 still under water."])
        self.assertEqual(reports[0].source, "news/The Hindu")
        self.assertIn("skipped 1", status.message)

    def test_mastodon_keeps_recent_english_disaster_posts(self):
        now = datetime.now(timezone.utc)
        def status(sid, html_text, lang="en", days_old=0):
            return {"id": sid, "content": html_text, "language": lang, "url": f"https://m.example/@a/{sid}",
                    "created_at": (now - timedelta(days=days_old)).isoformat().replace("+00:00", "Z"), "account": {"acct": "a"}}
        statuses = [
            status("1", "<p>Flash flood warning for the river towns, residents should move to higher ground now <a>#<span>flood</span></a></p>"),
            status("2", "<p>Flash flood warning for the river towns, people are stranded on the bridge</p>", days_old=10),
            status("3", "<p>Letra da música Landslide de Fleetwood Mac com a tradução completa</p>", lang="pt"),
        ]
        with mock.patch.object(live_sources.requests, "get", return_value=_fake_response(200, statuses)):
            reports, status_ = live_sources.fetch_mastodon(tags=["flood"])
        self.assertEqual([r.id for r in reports], ["mastodon-1"])
        self.assertTrue(reports[0].text.endswith("#flood"))
        self.assertIn("skipped 2", status_.message)

    def test_eonet_and_emsc_events(self):
        eonet = {"events": [
            {"id": "EONET_1", "title": "Tropical Cyclone Arnab", "categories": [{"title": "Severe Storms"}],
             "sources": [{"id": "JTWC", "url": "https://example.org/arnab"}],
             "geometry": [{"type": "Point", "date": "2026-10-28T06:00:00Z", "coordinates": [81.8, 16.4]}]},
            {"id": "EONET_2", "title": "Prescribed Fire Walker Hill", "categories": [{"title": "Wildfires"}], "sources": [],
             "geometry": [{"type": "Point", "date": "2026-10-28T11:22:00Z", "coordinates": [-111.8, 35.2]}]}]}
        with mock.patch.object(live_sources.requests, "get", return_value=_fake_response(200, eonet)):
            reports, status = live_sources.fetch_eonet()
        self.assertEqual(len(reports), 1)
        self.assertIn("Tropical Cyclone Arnab", reports[0].text)
        self.assertIn("16.4°N, 81.8°E", reports[0].text)
        emsc = {"features": [{"properties": {"unid": "20261029_0001", "time": "2026-10-29T19:30:46.0Z", "mag": 3.6,
                                             "depth": 10.0, "flynn_region": "SOUTHWESTERN KASHMIR"}}]}
        with mock.patch.object(live_sources.requests, "get", return_value=_fake_response(200, emsc)):
            reports, status = live_sources.fetch_emsc()
        self.assertTrue(reports[0].text.startswith("M3.6 earthquake in Southwestern Kashmir"))
        with mock.patch.object(live_sources.requests, "get", return_value=_fake_response(204)):
            reports, status = live_sources.fetch_emsc()
        self.assertEqual(reports, [])
        self.assertTrue(status.ok)

    def test_reliefweb_needs_appname(self):
        with mock.patch.object(live_sources, "RELIEFWEB_APPNAME", ""):
            reports, status = live_sources.fetch_reliefweb()
        self.assertFalse(status.ok)
        self.assertIn("appname", status.message.lower())

    def test_humaid_loader_mocked(self):
        df = pd.DataFrame({"tweet_text": ["Need water in Puri", "Praying for Odisha"] * 10,
                           "class_label": ["requests_or_urgent_needs", "sympathy_and_support"] * 10})
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        listing = _fake_response(200, ["https://example.org/x.parquet"])
        payload = _fake_response(200, content=buf.getvalue())
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(curated_loader, "CACHE_DIR", Path(d)), \
                mock.patch.object(curated_loader.requests, "get", side_effect=[listing, payload]):
            out = curated_loader.load_humaid("validation")
        self.assertEqual(list(out.columns), ["text", "label"])
        self.assertEqual(len(out), 2)  # duplicates removed

    def test_crisisbench_maps_labels_and_drops_overlap(self):
        df = pd.DataFrame({
            "text": ["Need food and water in Puri", "Donate to the flood relief fund", "My cousin was affected",
                     "Praying for Odisha", "Bridge washed away near Cuttack"],
            "class_label": ["requests_or_needs", "donation_and_volunteering", "affected_individual",
                            "sympathy_and_support", "infrastructure_and_utilities_damage"]})
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(curated_loader, "CACHE_DIR", Path(d)), \
                mock.patch.object(curated_loader.requests, "get", return_value=_fake_response(200, content=buf.getvalue())):
            out = curated_loader.load_crisisbench("dev", exclude_texts={"praying for odisha"})
        self.assertEqual(dict(zip(out["text"], out["label"])), {
            "Need food and water in Puri": "requests_or_urgent_needs",
            "Donate to the flood relief fund": "rescue_volunteering_or_donation_effort",
            "Bridge washed away near Cuttack": "infrastructure_and_utility_damage",
        })  # affected_individual has no HumAID class; the excluded test tweet is gone

    def test_idrisi_loader_and_learned_gazetteer(self):
        import json as _json
        import zipfile
        from src.nlp import location_eval, ner
        rows = [{"text": "Flooding in Kerala and Alappuzha", "location_mentions": [{"text": "Kerala", "type": "State"}, {"text": "Alappuzha", "type": "City/town"}]},
                {"text": "Relief camps open in Alappuzha", "location_mentions": [{"text": "Alappuzha", "type": "City/town"}]},
                {"text": "Mobile phones donated", "location_mentions": []}]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(curated_loader.IDRISI_PREFIX + "kerala_floods_2018/train.jsonl", "\n".join(_json.dumps(r) for r in rows))
            z.writestr(curated_loader.IDRISI_PREFIX + "kerala_floods_2018/dev.jsonl", _json.dumps(rows[0]))
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(curated_loader, "CACHE_DIR", Path(d)), \
                mock.patch.object(curated_loader.requests, "get", return_value=_fake_response(200, content=buf.getvalue())):
            train = curated_loader.load_idrisi("train")
            dev = curated_loader.load_idrisi("dev")
        self.assertEqual(len(train), 3)
        self.assertEqual(train[0]["event"], "kerala_floods_2018")
        self.assertEqual(len(dev), 1)
        names = ner.build_learned_gazetteer(train, min_count=2)
        self.assertEqual(names, ["Alappuzha"])  # seen twice; Kerala only once
        with ner.use_learned_gazetteer(["Alappuzha", "Mobile"]):
            self.assertIn("Alappuzha", ner.extract_locations("water rising fast at Alappuzha"))
            self.assertNotIn("Mobile", ner.extract_locations("mobile network is down"))  # names match case-sensitively
            score = location_eval.score_locations(dev, "test")
        self.assertEqual((score.precision, score.recall), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
