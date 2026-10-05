/* Second Look console. One job: let a person decide whether a surgeon had the
 * critical view of safety before the clip, and see the frame that says so. */

const listEl = document.getElementById("run-list");
const tallyEl = document.getElementById("tally");
const detailEl = document.getElementById("detail-body");
const rowTpl = document.getElementById("tpl-run-row");

const CRIT = [
  ["C1", "Two structures"],
  ["C2", "Hepatocystic triangle"],
  ["C3", "Cystic plate"],
];
const STATE_WORD = { met: "met", partial: "partial", not_met: "not met", not_shown: "not shown" };
const ACTIONS = [["speak", "Speaks"], ["log", "Logs"], ["silent", "Silent"]];

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const cap = (s) => { s = String(s ?? ""); return s.charAt(0).toUpperCase() + s.slice(1); };
const lead = (s) => cap(String(s ?? "").replace(/^(yes|no|cannot tell):\s*/i, ""));
/* Nemotron sometimes returns just "Silent." as the note; say what the frame shows instead. */
const noteOf = (f) => (!String(f.note || "").trim() || String(f.note).replace(/[.\s]/g, "").toLowerCase() === f.action)
  ? CRIT.map(([id], i) => `${id} ${STATE_WORD[f.state[i]]}`).join(", ") : f.note;
const clock = (t) => `${Math.floor(t / 60)}:${String(Math.round(t % 60)).padStart(2, "0")}`;

async function json(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let msg = `${res.status}`;
    try { msg = (await res.json()).detail || msg; } catch (_) { /* keep status */ }
    throw new Error(msg);
  }
  return res.json();
}

/* ---- list -------------------------------------------------------------- */

function renderList(cases, selectedId, onPick) {
  listEl.textContent = "";
  tallyEl.textContent = cases.length === 1 ? "1 case" : `${cases.length} cases`;
  cases.forEach((c) => {
    const node = rowTpl.content.cloneNode(true);
    const li = node.querySelector(".run-row");
    node.querySelector(".run-repo").textContent = c.name;
    const v = node.querySelector(".run-verdict");
    const clear = c.verdict === "clear";
    v.textContent = clear ? "clear to clip" : "hold";
    v.dataset.ok = String(clear);
    node.querySelector(".run-counts").textContent = `${c.met} of 3 criteria met`;
    if (c.id === selectedId) li.setAttribute("aria-current", "true");
    node.querySelector(".run-btn").addEventListener("click", () => onPick(c.id));
    listEl.append(node);
  });
}

/* ---- pieces ------------------------------------------------------------ */

function gateHtml(c) {
  const clear = c.gate.verdict === "clear";
  const fin = c.raters.final;
  return `
    <section class="gate" aria-labelledby="gate-h">
      <div class="gate-head">
        <span id="gate-h">Clip gate</span>
        <span class="gate-verdict" data-ok="${clear}">${clear ? "Clear to clip" : "Hold — do not clip"}</span>
      </div>
      <div class="cols" aria-hidden="true"><span>Criterion</span><span>Surgeons</span><span>Second Look</span></div>
      ${CRIT.map(([id, label], i) => `
        <div class="crit">
          <span><span class="cid">${id}</span>${esc(label)}</span>
          <span class="r">${fin[i]} of 3</span>
          <span class="s" data-state="${c.gate.state[i]}">${STATE_WORD[c.gate.state[i]]}</span>
        </div>`).join("")}
      <div class="gate-foot">
        <p class="say">${esc(c.gate.message)}</p>
        ${c.debrief ? `<p class="debrief">${esc(c.debrief)}</p>` : ""}
      </div>
    </section>`;
}

function trackHtml(state) {
  const label = CRIT.map(([id], i) => `${id} ${STATE_WORD[state[i]]}`).join(", ");
  return `<span class="track" role="img" aria-label="${label}">${
    state.map((s) => `<span class="seg" data-state="${s}"></span>`).join("")}</span>`;
}

function readsHtml(state, reading) {
  const text = [reading.structures, reading.triangle, reading.gallbladder_bed];
  return `<dl class="reads">${CRIT.map(([id, label], i) => `
    <div><dt data-state="${state[i]}">${id}</dt>
    <dd><b>${esc(label)}: ${STATE_WORD[state[i]]}.</b> <span class="p">chance met ${reading.probs[i].toFixed(2)}</span> The two vision models scored it ${reading.eyes[i].map((x) => x.toFixed(0)).join(" and ")} out of 10.</dd></div>`).join("")}</dl>`;
}

function rowHtml(f, i, c) {
  const rt = f.raters;
  return `
  <li class="row" data-verdict="x" data-action="${f.action}" data-open="false" data-i="${i}" data-t="${f.video_t}">
    <button type="button" class="row-btn" aria-expanded="false" aria-controls="p-${i}">
      <svg class="chev" viewBox="0 0 16 16" aria-hidden="true" fill="currentColor"><path d="M6 3.5 10.5 8 6 12.5V3.5Z"/></svg>
      <code class="rule-id">${clock(f.t)}</code>
      <span class="row-note">${i === c.frames.length - 1 ? '<span class="final">gate</span>' : ""}<span class="act" data-action="${f.action}">${f.action}</span>${esc(noteOf(f))}</span>
      ${trackHtml(f.state)}
    </button>
    <div class="panel cols2" id="p-${i}" hidden>
      <img src="/frames/${esc(f.file)}" alt="Surgical frame at ${clock(f.t)}" loading="lazy" width="854" height="480">
      <div>
        ${readsHtml(f.state, f.reading)}
      </div>
      <div class="panel-foot">
        <p class="where">Surgeons: C1 ${rt[0]}/3 · C2 ${rt[1]}/3 · C3 ${rt[2]}/3 · Chance met: ${f.reading.probs.map((p) => p.toFixed(2)).join(" · ")}</p>
        <button type="button" class="link-btn" data-seek="${f.video_t}">Play from ${clock(f.t)}</button>
      </div>
    </div>
  </li>`;
}

function liveHtml(c, r) {
  const d = r.decision;
  const state = [d.c1, d.c2, d.c3];
  return `
  <li class="row" data-live="true" data-action="${esc(d.action)}" data-open="true" data-i="live">
    <button type="button" class="row-btn" aria-expanded="true" aria-controls="p-live">
      <svg class="chev" viewBox="0 0 16 16" aria-hidden="true" fill="currentColor"><path d="M6 3.5 10.5 8 6 12.5V3.5Z"/></svg>
      <code class="rule-id">Live</code>
      <span class="row-note"><span class="act" data-action="${esc(d.action)}">${esc(d.action)}</span>${esc(noteOf({ note: d.note, action: d.action, state }))}</span>
      ${trackHtml(state)}
    </button>
    <div class="panel cols2" id="p-live">
      <img src="/frames/${esc(r.file)}" alt="Surgical frame at ${clock(r.t)}" width="854" height="480">
      <div>${readsHtml(state, r.reading)}</div>
      <div class="panel-foot"><p class="where">Frame ${clock(r.t)} read just now · ${r.seconds.eyes}s vision, ${r.seconds.gate}s gate</p></div>
    </div>
  </li>`;
}

/* ---- detail ------------------------------------------------------------ */

function renderDetail(c) {
  const sub = c.timelapse
    ? `${esc(c.source)} · 4× time-lapse, one frame a second. The sequence ends just before the first clip or cut.`
    : `${esc(c.source)} · 90-second clip. The gate is read at ${clock(c.frames[c.frames.length - 1].t)}.`;
  const counts = Object.fromEntries(ACTIONS.map(([k]) => [k, c.frames.filter((f) => f.action === k).length]));

  detailEl.innerHTML = `
    <h1 class="run-title">${esc(c.name)}</h1>
    <p class="run-sub">${sub}</p>
    <div class="stage">
      <figure class="player">
        <video id="vid" controls playsinline muted preload="metadata" src="${esc(c.video)}#t=0.1"
               aria-label="Operation footage, ${esc(c.name)}"></video>
        <figcaption><span>${esc(c.source)}</span><span id="vid-clock">0:00</span></figcaption>
      </figure>
      ${gateHtml(c)}
    </div>
    <div class="ledger-head">
      <h2>Verification ledger</h2>
      <p>One row per key frame. Open a row for the frame and the reasoning.</p>
      <button type="button" class="btn run-it" id="run-it" title="Reads the last frame again with the vision and Nemotron models">Run it</button>
    </div>
    <div class="filters-line">
      <div class="filters" role="group" aria-label="Filter the ledger">
        <button type="button" class="chip" data-filter="all" aria-pressed="true">All <span class="n">${c.frames.length}</span></button>
        ${ACTIONS.map(([k, l]) => `<button type="button" class="chip" data-filter="${k}" aria-pressed="false">${l} <span class="n">${counts[k]}</span></button>`).join("")}
      </div>
      <div class="track-key" aria-hidden="true"><span>C1</span><span>C2</span><span>C3</span></div>
    </div>
    <p class="sr-live" id="filter-status" role="status"></p>
    <p class="error" id="run-error" role="alert" hidden></p>
    <ol class="ledger" id="ledger">${c.frames.map((f, i) => rowHtml(f, i, c)).join("")}</ol>`;

  const ledger = document.getElementById("ledger");
  const vid = document.getElementById("vid");
  const rows = () => [...ledger.querySelectorAll(".row:not([data-live])")];

  detailEl.querySelectorAll(".chip").forEach((chip) => chip.addEventListener("click", () => {
    const want = chip.dataset.filter;
    detailEl.querySelectorAll(".chip").forEach((x) => x.setAttribute("aria-pressed", String(x === chip)));
    let shown = 0;
    rows().forEach((r) => {
      const ok = want === "all" || r.dataset.action === want;
      r.hidden = !ok; if (ok) shown += 1;
    });
    document.getElementById("filter-status").textContent = `Showing ${shown} of ${c.frames.length} frames.`;
  }));

  ledger.addEventListener("click", (e) => {
    const seek = e.target.closest("[data-seek]");
    if (seek) { vid.currentTime = Number(seek.dataset.seek); vid.play().catch(() => {}); vid.scrollIntoView({ block: "nearest", behavior: "smooth" }); return; }
    const btn = e.target.closest(".row-btn");
    if (!btn) return;
    const row = btn.closest(".row");
    const open = btn.getAttribute("aria-expanded") === "true";
    btn.setAttribute("aria-expanded", String(!open));
    row.dataset.open = String(!open);
    row.querySelector(".panel").hidden = open;
  });

  detailEl.addEventListener("keydown", (event) => {
    if (event.key !== "j" && event.key !== "k") return;
    const vis = rows().filter((r) => !r.hidden).map((r) => r.querySelector(".row-btn"));
    const at = vis.indexOf(document.activeElement);
    const next = event.key === "j" ? at + 1 : at - 1;
    if (vis[next]) { vis[next].focus(); event.preventDefault(); }
  });

  /* The row whose frame the video has reached. */
  vid.addEventListener("timeupdate", () => {
    document.getElementById("vid-clock").textContent = clock(c.timelapse ? vid.currentTime * 4 : vid.currentTime);
    let now = null;
    rows().forEach((r) => { if (Number(r.dataset.t) <= vid.currentTime + 0.05) now = r; });
    rows().forEach((r) => { r.dataset.now = String(r === now && !vid.paused); });
  });
  vid.addEventListener("pause", () => rows().forEach((r) => { r.dataset.now = "false"; }));

  const runBtn = document.getElementById("run-it");
  const errEl = document.getElementById("run-error");
  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true; runBtn.textContent = "Reading…"; errEl.hidden = true;
    try {
      const r = await json(`/api/cases/${encodeURIComponent(c.id)}/run`, { method: "POST" });
      ledger.querySelector("[data-live]")?.remove();
      ledger.insertAdjacentHTML("afterbegin", liveHtml(c, r));
    } catch (err) {
      errEl.textContent = err.message; errEl.hidden = false;
    } finally {
      runBtn.disabled = false; runBtn.textContent = "Run it";
    }
  });
}

/* ---- boot -------------------------------------------------------------- */

async function select(id, cases) {
  renderList(cases, id, (next) => { history.replaceState(null, "", `#${next}`); select(next, cases); });
  detailEl.innerHTML = `<div class="empty"><p>Loading…</p></div>`;
  renderDetail(await json(`/api/cases/${encodeURIComponent(id)}`));
}

async function boot() {
  tallyEl.textContent = "loading";
  let cases;
  try { cases = await json("/api/cases"); }
  catch (err) {
    detailEl.innerHTML = `<div class="empty"><h2>Cannot reach the case store</h2><p>${esc(err.message)}</p></div>`;
    return;
  }
  const want = location.hash.slice(1);
  const first = cases.find((c) => c.id === want) || cases.find((c) => c.verdict === "hold") || cases[0];
  await select(first.id, cases);
}

boot();
