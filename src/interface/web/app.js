/* CrisisBridge AI console. Plain JS, no build step. Talks to src/interface/api.py. */
"use strict";

const PRIORITIES = ["Critical", "High", "Medium", "Low"];
const PCOLOR = { Critical: "var(--p-critical)", High: "var(--p-high)", Medium: "var(--p-medium)", Low: "var(--p-low)" };
const EXAMPLES = [
  "What is the most critical situation near XYZ bridge right now?",
  "Where are people trapped and needing rescue?",
  "Which areas need drinking water or food?",
  "Any missing people reported?",
  "Medical emergencies in Velachery?",
];
const PROC_EXAMPLES = [
  "Family of 5 stuck on a roof in Velachery, need a boat urgently",
  "Elderly woman unconscious near Tambaram station, no ambulance can reach",
  "Power has been out in Adyar since last night, water entering ground floors",
  "Praying for everyone affected by the floods, stay safe",
];
const LAYERS = [
  ["Ingestion Layer", "Pulls curated datasets and live feeds (GDACS, Reddit, RSS) into a unified, deduplicated stream, normalising format across sources.", "src/ingestion/"],
  ["NLP Processing Layer", "Classification filters informative reports; NER extracts location, need type and people count; urgency scoring assigns a priority tier.", "src/nlp/"],
  ["Indexing Layer", "Embeds structured incident reports as dense vectors in a vector store for fast nearest-neighbour retrieval.", "src/rag/embeddings.py · indexer.py"],
  ["RAG Retrieval + Generation", "Retrieves the most relevant reports for a coordinator's query and synthesises a grounded, cited answer.", "src/rag/retriever.py · generator.py"],
  ["Coordinator Interface", "Natural-language query console with a live priority feed and on-demand briefings, built for operators under time pressure.", "src/interface/"],
];

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const ico = (name) => `<svg class="ico"><use href="#i-${name}"/></svg>`;
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtInt = (n) => Number(n || 0).toLocaleString();
const pct = (a, b) => (b ? Math.round((a / b) * 100) : 0);
const label = (s) => String(s || "").replace(/_/g, " ");

const store = {
  get(key, fallback) { try { const v = localStorage.getItem("cb." + key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem("cb." + key, JSON.stringify(value)); } catch { /* storage unavailable */ } },
};

const state = { status: null, overview: null, sources: null, feed: { priorities: new Set(["Critical", "High"]), offset: 0, limit: 25 }, timers: {} };

/* ---------------------------------------------------------------- api */
async function api(path, options = {}) {
  const opts = { headers: {}, ...options };
  if (opts.body && typeof opts.body !== "string") {
    opts.body = JSON.stringify(opts.body);
    opts.headers["Content-Type"] = "application/json";
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new Error("Cannot reach the CrisisBridge server. Is `python -m src.interface.api` still running?");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new Error(detail || `Request failed (HTTP ${res.status})`);
  }
  return data;
}

function toast(msg, ms = 3200) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(state.timers.toast);
  state.timers.toast = setTimeout(() => (el.hidden = true), ms);
}

function errorBox(err) {
  return `<div class="error-box">${ico("alert")}<div>${esc(err.message || err)}</div></div>`;
}

function busy(btn, on, text) {
  if (on) {
    btn.dataset.html = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span>${esc(text || "Working…")}`;
  } else if (btn.dataset.html) {
    btn.disabled = false;
    btn.innerHTML = btn.dataset.html;
  }
}

/* ---------------------------------------------------------------- shared renderers */
function fmtTime(ts) {
  if (!ts) return "time unknown";
  const d = new Date(ts);
  if (isNaN(d)) return ts;
  return d.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

function pill(p) { return `<span class="pill ${esc(p)}">${esc(p || "n/a")}</span>`; }

function meter(value, priority) {
  const v = Math.max(0, Math.min(100, Number(value) || 0));
  return `<div class="meter ${esc(priority || "")}"><i style="width:${v}%"></i></div>`;
}

function fieldGrid(r) {
  const people = r.people_count ?? null;
  const pclass = r.priority === "Critical" ? "field-critical" : `field-${r.priority}`;
  const cell = (lbl, icon, value) =>
    `<div class="field"><span class="field-label">${ico(icon)}${lbl}</span><span class="field-value ${value ? "" : "none"}">${esc(value || "unknown")}</span></div>`;
  return `<div class="field-grid">
    ${cell("Location", "pin", r.primary_location)}
    ${cell("Need", "need", r.primary_need)}
    ${cell("People", "people", people === null ? "" : String(people))}
    <div class="field ${pclass}"><span class="field-label">${ico("prioritize")}Priority</span><span class="field-value">${esc(r.priority || "unscored")}</span></div>
  </div>`;
}

function reasonsList(reasons) {
  if (!reasons || !reasons.length) return `<p class="muted small">No urgency signals found.</p>`;
  return `<ul class="reasons">${reasons.map((txt) => {
    const m = /^(.*?)\s*\(([+-]?\d+(?:\.\d+)?)\)\s*$/.exec(txt);
    if (!m) return `<li><span>${esc(txt)}</span></li>`;
    const neg = m[2].startsWith("-");
    return `<li><span>${esc(m[1])}</span><b class="${neg ? "neg" : ""}">${esc(m[2].startsWith("+") || neg ? m[2] : "+" + m[2])}</b></li>`;
  }).join("")}</ul>`;
}

function barRows(rows, max, color) {
  if (!rows.length) return `<div class="empty">Nothing yet</div>`;
  const top = max || Math.max(...rows.map((r) => r[1]), 1);
  return rows.map(([name, n]) => `<div class="bar-row"><span title="${esc(name)}">${esc(name)}</span>
    <div class="meter"><i style="width:${(n / top) * 100}%;${color ? `background:${color}` : ""}"></i></div><b>${fmtInt(n)}</b></div>`).join("");
}

function incidentRow(r) {
  return `<div class="incident">
    ${pill(r.priority)}
    <div>
      <div class="incident-text">${esc(r.text)}</div>
      <div class="incident-meta">
        ${r.primary_location ? `<span>${ico("pin")}${esc(r.primary_location)}</span>` : ""}
        ${r.primary_need ? `<span>${ico("need")}${esc(r.primary_need)}</span>` : ""}
        ${r.people_count != null ? `<span>${ico("people")}${esc(r.people_count)}</span>` : ""}
        <span>${ico("clock")}${esc(fmtTime(r.timestamp))}</span>
        <span>${esc(r.source)}</span>
      </div>
    </div>
    <div class="score">${Math.round(r.urgency_score)}<small>urgency</small></div>
  </div>`;
}

/* ---------------------------------------------------------------- status */
async function loadStatus() {
  try {
    state.status = await api("/api/status");
  } catch (err) {
    state.status = null;
    $("#index-pill").className = "index-pill bad";
    $("#index-pill-text").textContent = "Server offline";
    return null;
  }
  const s = state.status;
  const pillEl = $("#index-pill");
  pillEl.className = "index-pill " + (s.index_ready ? "ok" : "bad");
  $("#index-pill-text").textContent = s.index_ready ? `${fmtInt(s.index.count)} reports indexed` : "No index loaded";
  $("#no-index").hidden = s.index_ready || location.hash === "#system";
  $("#no-index-msg").textContent = s.message || "Build one to start answering queries.";
  $("#topbar-meta").innerHTML = s.index_ready
    ? `${esc(s.index.embedder)} · ${esc(s.index.store)}<br>built ${esc(fmtTime(s.index.built_at))}`
    : "";
  const llm = s.llm;
  const keyNote = [llm.groq.key ? "Groq key set" : "no Groq key", llm.gemini.key ? "Gemini key set" : "no Gemini key"].join(" · ");
  $("#provider-note").textContent = keyNote;
  return s;
}

/* ---------------------------------------------------------------- overview */
async function renderOverview() {
  const s = state.status;
  renderFlow();
  if (!s || !s.index_ready) {
    $("#ov-top").innerHTML = `<div class="empty">No index yet. Build it from the System page.</div>`;
    return;
  }
  let o;
  try {
    o = state.overview = await api("/api/overview");
  } catch (err) {
    $("#ov-top").innerHTML = errorBox(err);
    return;
  }
  $("#st-total").textContent = fmtInt(o.total);
  $("#st-total-sub").textContent = `reports indexed from ${o.sources.length} source${o.sources.length === 1 ? "" : "s"}`;
  $("#st-loc").textContent = pct(o.with_location, o.total) + "%";
  $("#st-crit").textContent = fmtInt(o.priorities.Critical);
  const nav = $("#nav-critical");
  nav.hidden = !o.priorities.Critical;
  nav.textContent = o.priorities.Critical;

  $("#ov-top").innerHTML = o.top.length ? o.top.map(incidentRow).join("") : `<div class="empty">No critical or high-priority reports.</div>`;
  $("#ov-dist").innerHTML = PRIORITIES.map((p) => `<i style="width:${pct(o.priorities[p], o.total)}%;background:${PCOLOR[p]}" title="${p}: ${o.priorities[p]}"></i>`).join("");
  $("#ov-dist-legend").innerHTML = PRIORITIES.map((p) => `<div><span class="sw" style="background:${PCOLOR[p]}"></span>${p}<b>${fmtInt(o.priorities[p])}</b></div>`).join("");
  $("#ov-needs").innerHTML = barRows(o.needs.slice(0, 6), null, "var(--accent)");
  $("#ov-locs").innerHTML = o.locations.length
    ? o.locations.map(([loc, n]) => `<button class="chip" data-loc="${esc(loc)}">${ico("pin")}${esc(loc)} <b>${n}</b></button>`).join("")
    : `<div class="empty">No locations extracted</div>`;
  $$("#ov-locs [data-loc]").forEach((b) => b.addEventListener("click", () => {
    $("#f-q").value = b.dataset.loc;
    state.feed.priorities = new Set(PRIORITIES);
    location.hash = "#feed";
  }));
  renderFlow();
}

function renderFlow() {
  const s = state.status;
  const o = state.overview;
  const ready = s && s.index_ready;
  const gen = s ? (s.llm.groq.key ? "Groq LLM" : s.llm.gemini.key ? "Gemini LLM" : "Extractive (no key)") : "–";
  const srcNames = o ? o.sources.map(([n]) => n).join(" · ") : "HumAID · GDACS · Reddit · RSS";
  const nodes = [
    ["Data Sources", "HumAID · GDACS · Reddit · RSS", srcNames, !!o],
    ["Ingestion", "Unify & clean stream", o ? `${fmtInt(o.total)} reports` : "idle", !!o],
    ["NLP Processing", "Classify · NER · Urgency", o ? `${pct(o.with_need, o.total)}% with a need` : "idle", !!o],
    ["Vector Index", "Embedded incident reports", ready ? `${s.index.embedder} · ${s.index.store}` : "not built", ready],
    ["RAG Retrieval + Generation", "Top reports → grounded, cited answer", ready ? gen : "waiting", ready],
    ["Coordinator Interface", "“Critical situations near XYZ bridge?” + live priority feed", "You are here", true],
  ];
  $("#ov-flow").innerHTML = nodes.map(([t, sub, live, on]) =>
    `<div class="flow-node"><b>${esc(t)}</b><small>${esc(sub)}</small><span class="flow-live ${on ? "" : "off"}">${esc(live)}</span></div>`).join("");
}

/* ---------------------------------------------------------------- commander query */
function setupQuery() {
  $("#examples").innerHTML = EXAMPLES.map((q) => `<button type="button" class="chip" data-q="${esc(q)}">${esc(q)}</button>`).join("");
  $$("#examples [data-q]").forEach((b) => b.addEventListener("click", () => { $("#q").value = b.dataset.q; runAsk(); }));
  $("#opt-k").value = String(store.get("k", 5));
  $("#opt-min").value = store.get("min", "");
  $("#opt-provider").value = store.get("provider", "auto");
  ["k", "min", "provider"].forEach((k) => $("#opt-" + k).addEventListener("change", (e) => store.set(k, k === "k" ? Number(e.target.value) : e.target.value)));
  $("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); runAsk(); });
  $("#hero-ask").addEventListener("submit", (e) => {
    e.preventDefault();
    $("#q").value = $("#hero-q").value.trim() || $("#hero-q").placeholder;
    location.hash = "#query";
    runAsk();
  });
}

function renderAnswerText(text) {
  const cite = (s) => s.replace(/\[(\d+)\]/g, '<button type="button" class="cite" data-n="$1" title="Show report $1">$1</button>')
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  const out = [];
  let list = [];
  const flush = () => { if (list.length) { out.push(`<ul>${list.map((l) => `<li>${l}</li>`).join("")}</ul>`); list = []; } };
  esc(text).split(/\n+/).forEach((line) => {
    const t = line.trim();
    if (!t) return;
    const m = /^(?:[-*•]|\d+[.)])\s+(.*)$/.exec(t);
    if (m) list.push(cite(m[1]));
    else { flush(); out.push(`<p>${cite(t)}</p>`); }
  });
  flush();
  return out.join("");
}

function resultCard(item) {
  const r = item.report;
  const c = item.components;
  const comp = [["Similarity", c.similarity], ["Location", c.location], ["Urgency", c.urgency], ["Recency", c.recency], ["Need", c.need]];
  return `<article class="card result" id="res-${item.rank}">
    <div class="result-num">${item.rank}</div>
    <div class="result-body">
      <div class="result-top">
        ${pill(r.priority)}
        <span class="urg">${meter(r.urgency_score, r.priority)}urgency <b>${Math.round(r.urgency_score)}</b>/100</span>
        <span class="muted small">${esc(r.source)} · ${esc(fmtTime(r.timestamp))} · <code>${esc(r.id)}</code></span>
        ${r.url ? `<a class="link small" href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">Source ${ico("arrow")}</a>` : ""}
      </div>
      <p class="result-text">${esc(r.text)}</p>
      ${fieldGrid(r)}
      <details class="why">
        <summary>Why this rank? ${ico("chev")}</summary>
        <div class="why-body">
          <div>
            <div class="eyebrow">Retrieval score ${Number(item.score).toFixed(3)}</div>
            <div class="bars" style="margin-top:10px">${comp.map(([n, v]) => `<div class="bar-row"><span>${n}</span>${meter((v || 0) * 100)}<b>${Number(v || 0).toFixed(2)}</b></div>`).join("")}</div>
          </div>
          <div>
            <div class="eyebrow">Urgency reasons</div>
            <div style="margin-top:10px">${reasonsList(r.urgency_reasons)}</div>
            <p class="muted small" style="margin-top:10px">Category: ${esc(label(r.category))} (${Math.round((r.category_confidence || 0) * 100)}% confidence)</p>
          </div>
        </div>
      </details>
    </div>
  </article>`;
}

async function runAsk() {
  const query = $("#q").value.trim();
  if (!query) { $("#q").focus(); return; }
  const out = $("#ask-out");
  const btn = $("#ask-btn");
  busy(btn, true, "Asking…");
  out.innerHTML = `<div class="card loading-card"><span class="spinner"></span>Retrieving reports and generating a grounded answer…</div>`;
  try {
    const data = await api("/api/ask", {
      method: "POST",
      body: { query, k: Number($("#opt-k").value), min_priority: $("#opt-min").value, provider: $("#opt-provider").value },
    });
    const p = data.plan;
    const flag = (on, text) => `<span class="chip ${on ? "on" : "off"}">${ico(on ? "check" : "x")}${esc(text)}</span>`;
    out.innerHTML = `
      <div class="plan">
        <span class="eyebrow">Query understanding</span>
        ${p.locations.map((l) => `<span class="chip on">${ico("pin")}${esc(l)}</span>`).join("")}
        ${p.needs.map((n) => `<span class="chip on">${ico("need")}${esc(n)}</span>`).join("")}
        ${flag(p.wants_urgent, "urgency intent")}
        ${flag(p.wants_recent, "recency intent")}
        ${flag(p.location_filter_applied, "location filter")}
      </div>
      <div class="card answer-card">
        <div class="eyebrow">Grounded answer</div>
        <div class="answer-text">${renderAnswerText(data.answer.text)}</div>
        ${data.answer.notice ? `<div class="notice">${ico("alert")}<div>${esc(data.answer.notice)}</div></div>` : ""}
        <div class="answer-foot">
          <span>${ico("spark")}Generated by ${esc(data.answer.provider)}</span>
          <span>${ico("layers")}${data.results.length} cited report${data.results.length === 1 ? "" : "s"}</span>
          <span>${ico("clock")}${fmtInt(data.elapsed_ms)} ms</span>
        </div>
      </div>
      ${data.results.length ? `<div class="eyebrow">Cited reports</div><div class="results">${data.results.map(resultCard).join("")}</div>` : ""}`;
    $$(".cite", out).forEach((b) => b.addEventListener("click", () => {
      const card = $("#res-" + b.dataset.n);
      if (!card) return;
      card.scrollIntoView({ behavior: "smooth", block: "center" });
      card.classList.add("flash");
      setTimeout(() => card.classList.remove("flash"), 1400);
    }));
  } catch (err) {
    out.innerHTML = errorBox(err);
  } finally {
    busy(btn, false);
  }
}

/* ---------------------------------------------------------------- live priority feed */
function feedParams(withPaging = true) {
  const f = state.feed;
  const params = new URLSearchParams();
  if (f.priorities.size && f.priorities.size < 4) params.set("priority", [...f.priorities].join(","));
  if ($("#f-q").value.trim()) params.set("q", $("#f-q").value.trim());
  if ($("#f-need").value) params.set("need", $("#f-need").value);
  if ($("#f-source").value) params.set("source", $("#f-source").value);
  if (withPaging) { params.set("offset", f.offset); params.set("limit", f.limit); }
  return params;
}

function renderTiles() {
  const counts = state.overview ? state.overview.priorities : {};
  $("#feed-tiles").innerHTML = PRIORITIES.map((p) => {
    const on = state.feed.priorities.has(p);
    return `<button class="tile ${on ? "on" : "off"}" data-p="${p}" aria-pressed="${on}">
      <span><i style="background:${PCOLOR[p]}"></i>${p}</span><b>${fmtInt(counts[p] ?? 0)}</b></button>`;
  }).join("");
  $$("#feed-tiles .tile").forEach((t) => t.addEventListener("click", () => {
    const set = state.feed.priorities;
    set.has(t.dataset.p) ? set.delete(t.dataset.p) : set.add(t.dataset.p);
    state.feed.offset = 0;
    renderTiles();
    loadFeed();
  }));
}

function fillFeedFilters() {
  const o = state.overview;
  if (!o) return;
  const fill = (sel, items, first) => {
    const cur = sel.value;
    sel.innerHTML = `<option value="">${first}</option>` + items.map(([n, c]) => `<option value="${esc(n)}">${esc(n)} (${c})</option>`).join("");
    sel.value = items.some(([n]) => n === cur) ? cur : "";
  };
  fill($("#f-need"), o.needs, "All needs");
  fill($("#f-source"), o.sources, "All sources");
}

async function loadFeed() {
  const list = $("#feed-list");
  $("#f-csv").href = "/api/reports.csv?" + feedParams(false).toString();
  if (!state.status || !state.status.index_ready) {
    list.innerHTML = `<div class="empty">No index loaded.</div>`;
    $("#feed-pager").innerHTML = "";
    return;
  }
  try {
    const data = await api("/api/reports?" + feedParams().toString());
    list.innerHTML = data.items.length ? data.items.map((r) => `<div class="feed-row">
        <div>${pill(r.priority)}</div>
        <div class="urgcell"><b>${Math.round(r.urgency_score)}</b>${meter(r.urgency_score, r.priority)}</div>
        <div class="txt"><p title="Click to expand">${esc(r.text)}</p><small>${esc(r.source)} · ${esc(fmtTime(r.timestamp))} · ${esc(label(r.category))}</small></div>
        <div class="cell ${r.primary_location ? "" : "none"}">${esc(r.primary_location || "–")}</div>
        <div class="cell ${r.primary_need ? "" : "none"}">${esc(r.primary_need || "–")}</div>
        <div class="cell ${r.people_count != null ? "" : "none"}">${esc(r.people_count ?? "–")}</div>
      </div>`).join("") : `<div class="empty">No reports match these filters.</div>`;
    $$(".feed-row .txt p", list).forEach((p) => p.addEventListener("click", () => p.classList.toggle("open")));
    const f = state.feed;
    const end = Math.min(f.offset + f.limit, data.total);
    $("#feed-pager").innerHTML = `<span>${data.total ? `${fmtInt(f.offset + 1)}–${fmtInt(end)} of ${fmtInt(data.total)}` : "0 results"}, sorted by urgency</span>
      <div><button class="btn btn-sm" id="pg-prev" ${f.offset ? "" : "disabled"}>Previous</button>
      <button class="btn btn-sm" id="pg-next" ${end < data.total ? "" : "disabled"}>Next</button></div>`;
    $("#pg-prev").addEventListener("click", () => { f.offset = Math.max(0, f.offset - f.limit); loadFeed(); });
    $("#pg-next").addEventListener("click", () => { f.offset += f.limit; loadFeed(); });
  } catch (err) {
    list.innerHTML = `<div style="padding:16px">${errorBox(err)}</div>`;
  }
}

function setupFeed() {
  let t;
  $("#f-q").addEventListener("input", () => { clearTimeout(t); t = setTimeout(() => { state.feed.offset = 0; loadFeed(); }, 250); });
  ["#f-need", "#f-source"].forEach((s) => $(s).addEventListener("change", () => { state.feed.offset = 0; loadFeed(); }));
  $("#f-auto").checked = store.get("autoRefresh", false);
  $("#f-auto").addEventListener("change", (e) => { store.set("autoRefresh", e.target.checked); scheduleRefresh(); });
}

function scheduleRefresh() {
  clearInterval(state.timers.refresh);
  if ($("#f-auto").checked && currentView() === "feed") {
    state.timers.refresh = setInterval(async () => {
      await loadStatus();
      if (state.status && state.status.index_ready) state.overview = await api("/api/overview").catch(() => state.overview);
      renderTiles();
      loadFeed();
    }, 15000);
  }
}

/* ---------------------------------------------------------------- process a message */
function setupProcess() {
  $("#proc-examples").innerHTML = PROC_EXAMPLES.map((q) => `<button type="button" class="chip" data-t="${esc(q)}">${esc(q.length > 46 ? q.slice(0, 44) + "…" : q)}</button>`).join("");
  $$("#proc-examples [data-t]").forEach((b) => b.addEventListener("click", () => { $("#proc-text").value = b.dataset.t; runProcess(); }));
  $("#proc-form").addEventListener("submit", (e) => { e.preventDefault(); runProcess(); });
}

async function runProcess() {
  const text = $("#proc-text").value.trim();
  if (!text) { $("#proc-text").focus(); return; }
  const btn = $("#proc-btn");
  const out = $("#proc-out");
  busy(btn, true, "Processing…");
  try {
    const data = await api("/api/process", { method: "POST", body: { text, add_to_index: $("#proc-add").checked } });
    const r = data.report;
    const score = Math.max(0, Math.min(100, r.urgency_score));
    out.innerHTML = `<div class="card proc-result">
      <div class="card-head"><div><div class="eyebrow">Structured, prioritised output</div><h3>Ready to act on in seconds</h3></div>${pill(r.priority)}</div>
      <div class="proc-quote">“${esc(r.text)}”</div>
      ${fieldGrid(r)}
      <div class="proc-cols">
        <div class="gauge">
          <div class="eyebrow">Urgency score</div>
          <div class="gauge-value">${score.toFixed(1)}<small> / 100 · ${esc(r.priority)}</small></div>
          <div class="gauge-track"><span class="gauge-marker" style="left:${score}%"></span></div>
          <div class="gauge-scale"><span style="left:0;transform:none">0</span><span style="left:25%">25 Medium</span><span style="left:45%">45 High</span><span style="left:70%">70 Critical</span><span style="right:0;left:auto;transform:none">100</span></div>
          <div style="margin-top:12px">${reasonsList(r.urgency_reasons)}</div>
        </div>
        <div class="stack" style="gap:16px">
          <div>
            <div class="eyebrow">Humanitarian category</div>
            <div class="bar-row" style="margin-top:10px;grid-template-columns:minmax(0,1fr) 120px auto"><span>${esc(label(r.category))}</span>${meter((r.category_confidence || 0) * 100)}<b>${Math.round((r.category_confidence || 0) * 100)}%</b></div>
          </div>
          <div><div class="eyebrow">All locations</div><div class="chips" style="margin-top:8px">${r.locations.length ? r.locations.map((l) => `<span class="chip">${ico("pin")}${esc(l)}</span>`).join("") : `<span class="muted small">None found</span>`}</div></div>
          <div><div class="eyebrow">All needs</div><div class="chips" style="margin-top:8px">${r.needs.length ? r.needs.map((n) => `<span class="chip">${ico("need")}${esc(n)}</span>`).join("") : `<span class="muted small">None stated</span>`}</div></div>
          <code style="align-self:flex-start;white-space:normal">${esc(r.summary_line)}</code>
        </div>
      </div>
      ${data.added ? `<div class="notice" style="background:var(--ok-soft);color:var(--ok)">${ico("check")}<div>Added as <code>${esc(r.id)}</code>. The index now holds ${fmtInt(data.index_count)} reports and commander queries can cite it.</div></div>` : ""}
    </div>`;
    if (data.added) { await loadStatus(); state.overview = null; }
  } catch (err) {
    out.innerHTML = errorBox(err);
  } finally {
    busy(btn, false);
  }
}

/* ---------------------------------------------------------------- data sources */
async function renderSources() {
  try {
    state.sources = await api("/api/sources");
  } catch (err) {
    $("#src-live").innerHTML = errorBox(err);
    return;
  }
  const { curated, live } = state.sources;
  $("#src-curated").innerHTML = curated.map((c) => `<div class="source">
      <span class="bullet"></span>
      <div><b>${esc(c.name)}</b><p>${esc(c.desc)}</p></div>
      <span class="badge ${c.ok ? "ok" : "na"}">${esc(c.state)}</span>
    </div>`).join("");
  const chosen = new Set(store.get("liveSources", ["gdacs", "rss"]));
  $("#src-live").innerHTML = live.map((s) => `<label class="source">
      <input type="checkbox" value="${esc(s.key)}" ${s.unavailable ? "disabled" : ""} ${chosen.has(s.key) && !s.unavailable ? "checked" : ""}>
      <div><b>${esc(s.name)}</b><p>${esc(s.desc)}</p><div class="note">${esc(s.note)}</div></div>
      <span class="badge ${s.unavailable ? "na" : s.ready ? "ok" : "warn"}">${s.unavailable ? "not available" : s.ready ? "ready" : "needs setup"}</span>
    </label>`).join("");
  $$("#src-live input").forEach((i) => i.addEventListener("change", () => store.set("liveSources", $$("#src-live input:checked").map((x) => x.value))));
  const o = state.overview;
  $("#src-mix").innerHTML = o ? barRows(o.sources) : `<div class="empty">${state.status && state.status.index_ready ? "Loading…" : "No index loaded."}</div>`;
  if (!o && state.status && state.status.index_ready) {
    state.overview = await api("/api/overview").catch(() => null);
    if (state.overview) $("#src-mix").innerHTML = barRows(state.overview.sources);
  }
}

async function runLive() {
  const sources = $$("#src-live input:checked").map((i) => i.value);
  const out = $("#live-out");
  if (!sources.length) { out.innerHTML = errorBox("Select at least one live source."); return; }
  const btn = $("#live-btn");
  busy(btn, true, "Fetching…");
  out.innerHTML = "";
  try {
    const data = await api("/api/live", { method: "POST", body: { sources } });
    out.innerHTML = `<div class="status-list">${data.statuses.map((s) => `<div class="status-item ${s.ok ? "ok" : "skip"}">
        ${ico(s.ok ? "check" : "alert")}<div><b>${esc(s.name)}</b> · ${esc(s.message)}</div></div>`).join("")}
      <div class="status-item ok">${ico("layers")}<div>Added <b>${fmtInt(data.added)}</b> new report${data.added === 1 ? "" : "s"}. Index now holds ${fmtInt(data.index_count)}.</div></div></div>`;
    await loadStatus();
    state.overview = await api("/api/overview").catch(() => null);
    if (state.overview) $("#src-mix").innerHTML = barRows(state.overview.sources);
  } catch (err) {
    out.innerHTML = errorBox(err);
  } finally {
    busy(btn, false);
  }
}

/* ---------------------------------------------------------------- system */
function renderSystem() {
  const s = state.status;
  $("#sys-layers").innerHTML = LAYERS.map(([t, d, code], i) =>
    `<div class="layer"><span class="layer-num">0${i + 1}</span><b>${esc(t)}</b><p>${esc(d)}</p><code>${esc(code)}</code></div>`).join("");
  if (!s) {
    $("#sys-index").innerHTML = `<span class="k">Server</span><span class="v">offline</span>`;
    return;
  }
  const i = s.index;
  $("#sys-index").innerHTML = s.index_ready ? `
      <span class="k">Reports</span><span class="v">${fmtInt(i.count)}</span>
      <span class="k">Embeddings</span><span class="v">${esc(i.embedder)}${i.dim ? ` · ${i.dim}-dim` : ""}</span>
      <span class="k">Vector store</span><span class="v">${esc(i.store)}</span>
      <span class="k">Built</span><span class="v">${esc(fmtTime(i.built_at))}</span>`
    : `<span class="k">Status</span><span class="v">${esc(s.message || "Not built")}</span>`;
  const keyBadge = (on) => `<span class="badge ${on ? "ok" : "na"}">${on ? "key set" : "no key"}</span>`;
  $("#sys-llm").innerHTML = `
    <span class="k">Default</span><span class="v">${esc(s.llm.default)}</span>
    <span class="k">Groq</span><span class="v">${esc(s.llm.groq.model)}${keyBadge(s.llm.groq.key)}</span>
    <span class="k">Gemini</span><span class="v">${esc(s.llm.gemini.model)}${keyBadge(s.llm.gemini.key)}</span>
    <span class="k">Fallback</span><span class="v">Extractive, always available</span>`;
  renderBuild(s.build);
}

function renderBuild(b) {
  const out = $("#build-out");
  const btn = $("#build-btn");
  if (!b || b.state === "idle") { out.innerHTML = ""; btn.disabled = false; return; }
  if (b.state === "running") {
    btn.disabled = true;
    out.innerHTML = `<div class="build-status"><span class="spinner"></span>${esc(b.message)} Started ${esc(fmtTime(b.started))}.</div>`;
    return;
  }
  btn.disabled = false;
  if (b.state === "error") { out.innerHTML = `<div class="build-status error">${ico("alert")}${esc(b.message)}</div>`; return; }
  const m = b.summary || {};
  out.innerHTML = `<div class="build-status done">${ico("check")}<div>Built on <b>${esc(m.data_used)}</b>: ${fmtInt(m.n_indexed)} reports (${esc(m.embedder)}, ${esc(m.store)}).</div></div>
    <div class="kv">
      <span class="k">Evaluation</span><span class="v">${esc(m.eval_name)}</span>
      <span class="k">Accuracy</span><span class="v">${(m.accuracy * 100).toFixed(1)}%</span>
      <span class="k">Macro-F1</span><span class="v">${(m.macro_f1 * 100).toFixed(1)}%</span>
    </div>
    ${(m.notes || []).map((n) => `<div class="notice">${ico("alert")}<div>${esc(n)}</div></div>`).join("")}
    ${m.data_used === "sample" ? `<p class="muted small">Sample-mode scores are optimistic: the keyword prior was written alongside the sample. Use HumAID for a real evaluation.</p>` : ""}`;
}

async function runBuild(e) {
  e.preventDefault();
  const data = $("input[name=data]:checked").value;
  if (state.status && state.status.index_ready &&
      !confirm("Rebuild the index? This replaces the current index, including any live or manually added reports.")) return;
  try {
    const b = await api("/api/build", {
      method: "POST",
      body: { data, train_limit: Number($("#b-train").value) || 0, index_limit: Number($("#b-index").value) || 0 },
    });
    renderBuild(b);
    pollBuild();
  } catch (err) {
    $("#build-out").innerHTML = errorBox(err);
  }
}

function pollBuild() {
  clearTimeout(state.timers.build);
  state.timers.build = setTimeout(async () => {
    const b = await api("/api/build").catch(() => null);
    if (!b) return pollBuild();
    renderBuild(b);
    if (b.state === "running") return pollBuild();
    await loadStatus();
    state.overview = null;
    renderSystem();
    toast(b.state === "done" ? "Index rebuilt. All views now use the new index." : "Build failed. See the System page.");
  }, 1500);
}

/* ---------------------------------------------------------------- routing */
const VIEWS = ["overview", "query", "feed", "process", "sources", "system"];
const currentView = () => { const v = location.hash.slice(1); return VIEWS.includes(v) ? v : "overview"; };

async function route() {
  const view = currentView();
  $$(".view").forEach((v) => (v.hidden = v.dataset.view !== view));
  $$(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.nav === view));
  const sec = $(`.view[data-view="${view}"]`);
  $("#page-eyebrow").textContent = sec.dataset.eyebrow;
  $("#page-title").textContent = sec.dataset.title;
  document.title = view === "overview" ? "CrisisBridge AI" : `${sec.dataset.title} · CrisisBridge AI`;
  $("#sidebar").classList.remove("open");
  window.scrollTo(0, 0);
  if (state.status) $("#no-index").hidden = state.status.index_ready || view === "system";

  if (view === "overview") return renderOverview();
  if (view === "query") { setTimeout(() => $("#q").focus(), 0); return; }
  if (view === "feed") {
    if (!state.overview && state.status && state.status.index_ready) state.overview = await api("/api/overview").catch(() => null);
    renderTiles();
    fillFeedFilters();
    state.feed.offset = 0;
    await loadFeed();
    return scheduleRefresh();
  }
  if (view === "sources") return renderSources();
  if (view === "system") { renderSystem(); if (state.status && state.status.build.state === "running") pollBuild(); }
}

async function init() {
  setupQuery();
  setupFeed();
  setupProcess();
  $("#live-btn").addEventListener("click", runLive);
  $("#build-form").addEventListener("submit", runBuild);
  $("#menu-btn").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  $("#sidebar").addEventListener("click", (e) => { if (e.target === e.currentTarget) e.currentTarget.classList.remove("open"); });
  window.addEventListener("hashchange", () => { clearInterval(state.timers.refresh); route(); });
  await loadStatus();
  route();
}

init();
