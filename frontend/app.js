(() => {
  const params = new URLSearchParams(location.search);
  const API = (params.get("api") || window.API_BASE || "").replace(/\/$/, "");
  const $ = (s) => document.querySelector(s);
  const log = $("#log"), input = $("#input"), form = $("#composer"), send = $("#send");
  const statusEl = $("#status");
  let busy = false;

  // ---------- session (kept per browser so follow-ups have context)
  const newId = () => "s" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
  let sessionId;
  try { sessionId = localStorage.getItem("pa_session") || newId(); localStorage.setItem("pa_session", sessionId); }
  catch { sessionId = newId(); }

  // ---------- helpers
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (n) => (n == null ? null : Number(n).toLocaleString("en-US", { maximumFractionDigits: 0 }));
  const scrollDown = () => { log.scrollTop = log.scrollHeight; };

  function renderMarkdown(text, sources) {
    const byN = Object.fromEntries((sources || []).map((s) => [String(s.n), s]));
    // some models cite as 【1†L1-L4】: normalise to [1], and hide a half-streamed one at the end
    text = (text || "").replace(/【\s*(\d+)\s*(?:†[^】]*)?】/g, "[$1]").replace(/【[^】]*$/, "");
    let html = DOMPurify.sanitize(marked.parse(text), { ADD_ATTR: ["target"] });
    // [1] or [1][2] or [1, 2] -> superscript links to the source
    html = html.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (m, nums) =>
      nums.split(/\s*,\s*/).map((n) => byN[n]
        ? `<a class="cite" href="${esc(byN[n].url)}" target="_blank" rel="noopener" title="${esc(byN[n].title)}">${n}</a>`
        : `<span class="cite">${n}</span>`).join(""));
    return html;
  }

  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  function addUser(text) {
    const m = el("div", "msg user");
    m.appendChild(el("div", "bubble")).textContent = text;
    log.appendChild(m); scrollDown();
  }

  function addBot() {
    const m = el("div", "msg bot");
    const route = el("div", "route"); route.hidden = true;
    const cards = el("div", "cards"); cards.hidden = true;
    const bubble = el("div", "bubble", '<span class="typing"><i></i><i></i><i></i></span>');
    const sources = el("div", "sources"); sources.hidden = true;
    const trace = el("details", "trace"); trace.hidden = true;
    m.append(route, cards, bubble, sources, trace);
    log.appendChild(m); scrollDown();
    return { m, route, cards, bubble, sources, trace };
  }

  const ROUTE_LABEL = { docs: "searched documents", listings: "searched listings", both: "listings + documents",
                        chitchat: "conversation", off_topic: "off-topic" };

  function showMeta(ui, meta) {
    ui.route.innerHTML = `<b>${esc(ROUTE_LABEL[meta.route] || meta.route)}</b>` +
      (meta.standalone && meta.route !== "chitchat" ? `<span>“${esc(meta.standalone)}”</span>` : "");
    ui.route.hidden = false;
    if (meta.listings?.length) {
      ui.cards.innerHTML = meta.listings.map((l) => {
        const price = l.price ? `${esc(l.currency || "SAR")} ${fmt(l.price)}${l.purpose === "rent" && l.rent_frequency ? " / " + esc(l.rent_frequency) : ""}` : "Price on request";
        const spec = [l.beds ? `${fmt(l.beds)} bed` : "", l.baths ? `${fmt(l.baths)} bath` : "", l.area_sqm ? `${fmt(l.area_sqm)} sqm` : ""].filter(Boolean).join(" · ");
        const where = [l.district, l.city].filter(Boolean).join(", ");
        return `<a class="card" href="${esc(l.url)}" target="_blank" rel="noopener">
          <span class="n">[${l.n}] ${esc(l.type || "")} · ${esc(l.purpose || "")}</span>
          <span class="price">${price}</span><span class="t">${esc(l.title || "")}</span>
          <span class="m">${esc(where)}${spec ? " · " + esc(spec) : ""}</span></a>`;
      }).join("");
      ui.cards.hidden = false;
    }
    const docs = (meta.sources || []).filter((s) => s.kind !== "listing");
    if (docs.length) {
      ui.sources.innerHTML = docs.map((s) => `<a class="src" href="${esc(s.url)}" target="_blank" rel="noopener" title="${esc(s.url)}">
        <span class="num">${s.n}</span><span class="lbl">${esc(s.title)}</span><span class="k">${esc(s.source)} · ${esc(s.kind)}</span></a>`).join("");
      ui.sources.hidden = false;
    }
    const t = meta.trace || {};
    const lines = [];
    if (t.retrieval) lines.push(`queries: ${t.retrieval.queries.map((q) => `"${q}"`).join(", ")}`,
      `hybrid hits ${t.retrieval.raw_hits} → RRF ${t.retrieval.fused} → rerank ${t.retrieval.reranked} → MMR ${t.retrieval.final}  (${t.retrieval.ms} ms)`);
    if (t.listings) lines.push(`SQL filters: ${JSON.stringify(Object.fromEntries(Object.entries(t.listings.filters).filter(([, v]) => v != null)))}`,
      `matches: ${t.listings.total}${t.listings.relaxed.length ? `  (relaxed: ${t.listings.relaxed.join(", ")})` : ""}`);
    if (lines.length) {
      ui.trace.innerHTML = `<summary>How I answered</summary><pre>${esc(lines.join("\n"))}</pre>`;
      ui.trace.hidden = false;
    }
    ui.meta = meta;
  }

  // ---------- typewriter: Groq streams ~500 tokens/s, so a whole answer lands in a few
  // network bursts. Buffer it and reveal at a steady reading pace instead.
  const TYPE_CPS = 90;      // base speed, characters per second
  const CATCH_UP_S = 3;     // a large backlog speeds up so it clears in about this many seconds
  function makeTyper(render) {
    let target = "", shown = 0, carry = 0, last = 0, raf = null, ended = false, onDone = null;
    const step = (t) => {
      const dt = last ? Math.min(t - last, 100) : 16; last = t;
      const backlog = target.length - shown;
      if (backlog > 0) {
        if (document.hidden) { shown = target.length; }      // background tab: no animation
        else {
          carry += Math.max(TYPE_CPS, backlog / CATCH_UP_S) * dt / 1000;
          const n = Math.floor(carry); carry -= n;
          shown = Math.min(target.length, shown + n);
        }
        render(target.slice(0, shown));
      }
      if (shown < target.length) { raf = requestAnimationFrame(step); return; }
      raf = null; last = 0;
      if (ended && onDone) onDone();
    };
    return {
      push(s) { target += s; if (!raf) raf = requestAnimationFrame(step); },
      finish() {                       // resolves once everything received has been shown
        ended = true;
        return new Promise((res) => { onDone = res; if (!raf) { if (shown < target.length) raf = requestAnimationFrame(step); else res(); } });
      },
      flush() { if (raf) cancelAnimationFrame(raf); raf = null; shown = target.length; if (target) render(target); },
      get text() { return target; },
    };
  }

  async function ask(text) {
    if (busy || !text.trim()) return;
    busy = true; send.disabled = true;
    $("#welcome")?.remove();
    addUser(text);
    const ui = addBot();
    const typer = makeTyper((t) => { ui.bubble.innerHTML = renderMarkdown(t, ui.meta?.sources); scrollDown(); });
    let answer = "", doneInfo = null;
    try {
      const res = await fetch(`${API}/api/chat`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Server returned ${res.status}`);
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const line = buf.slice(0, i).trim(); buf = buf.slice(i + 2);
          if (!line.startsWith("data:")) continue;
          const ev = JSON.parse(line.slice(5));
          if (ev.type === "meta") showMeta(ui, ev);
          else if (ev.type === "token") { answer += ev.text; typer.push(ev.text); }
          else if (ev.type === "error") throw new Error(ev.message);
          else if (ev.type === "done") doneInfo = ev;
        }
      }
      await typer.finish();
      if (!answer) ui.bubble.innerHTML = "<p>No answer came back. Please try again.</p>";
      if (doneInfo) ui.m.appendChild(el("div", "route",
        `<span>${esc(doneInfo.model || "")} · ${(doneInfo.ms / 1000).toFixed(1)} s</span>`));
    } catch (e) {
      typer.flush();
      if (!answer) ui.bubble.remove();
      ui.m.appendChild(el("div", "error", esc(e.message || "Network error. Check that the backend is running.")));
    } finally {
      busy = false; send.disabled = false; input.focus(); scrollDown();
    }
  }

  // ---------- wiring
  form.addEventListener("submit", (e) => { e.preventDefault(); const t = input.value; input.value = ""; autosize(); ask(t); });
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });
  const autosize = () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 160) + "px"; };
  input.addEventListener("input", autosize);
  $("#chips")?.addEventListener("click", (e) => { if (e.target.tagName === "BUTTON") ask(e.target.textContent); });
  $("#newChat").addEventListener("click", async () => {
    fetch(`${API}/api/session/${sessionId}`, { method: "DELETE" }).catch(() => {});
    sessionId = newId(); try { localStorage.setItem("pa_session", sessionId); } catch {}
    location.reload();
  });

  // ---------- backend status (also wakes the free-tier server, which sleeps when idle; ~1 min)
  async function checkHealth(tries = 0) {
    try {
      const r = await fetch(`${API}/health`);
      const h = await r.json();
      if (h.ready) {
        statusEl.className = "status ok"; statusEl.lastElementChild.textContent = "Online";
        const d = h.data || {};
        const s = $("#stats");
        if (s && d.n_chunks) s.textContent = `Knowledge base: ${fmt(d.n_projects)} DarGlobal projects · ${fmt(d.n_listings)} Wasalt listings · ${fmt(d.n_chunks)} indexed passages · built ${String(d.built_at || "").slice(0, 10)}`;
        return;
      }
      statusEl.lastElementChild.textContent = "Loading models…";
    } catch {
      // free host cold start: keep saying "waking" for ~90 s before calling it offline
      statusEl.lastElementChild.textContent = tries < 22 ? "Waking server (up to 1 min)…" : "Offline";
      if (tries >= 22) statusEl.className = "status err";
    }
    if (tries < 40) setTimeout(() => checkHealth(tries + 1), 4000);
  }
  checkHealth();
})();
