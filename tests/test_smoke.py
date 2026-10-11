"""Offline smoke tests. Run from the project folder:  python -m unittest -v tests.test_smoke

Network calls (HumAID, GDACS, RSS, Groq, Gemini) are mocked, so these pass with no internet.
They check parsing and graceful-failure logic, not the live services themselves.
"""

from __future__ import annotations

import io
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


if __name__ == "__main__":
    unittest.main()
