"""One chat turn: route (LLM call 1) -> retrieve (local) -> answer (LLM call 2, streamed)."""
from __future__ import annotations

import logging
import time
from typing import Iterator

from . import config, llm, memory
from .listings import LISTINGS, listing_line
from .prompts import ANSWER_SYSTEM, CHITCHAT_SYSTEM, ROUTER_SYSTEM
from .retrieval import KB

log = logging.getLogger("chat")

SOURCE_LABEL = {"darglobal": "DarGlobal", "wasalt": "Wasalt"}
ROUTES = {"docs", "listings", "both", "chitchat", "off_topic"}


def _history_text(history: list[dict]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content'][:800]}" for m in history) or "(no earlier messages)"


def plan(question: str, history: list[dict]) -> dict:
    msgs = [{"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": f"Conversation so far:\n{_history_text(history)}\n\n"
                                        f"Latest user message: {question}"}]
    try:
        text, model = llm.complete(
            msgs, temperature=0.0, max_tokens=1200,
            validate=lambda t: (llm.parse_json(t) or {}).get("route") in ROUTES)
        p = llm.parse_json(text) or {}
    except llm.LLMUnavailable:
        raise
    except Exception as e:  # malformed reply etc. - fall back to searching everything
        log.warning("router failed: %s", e)
        p, model = {}, None
    route = p.get("route") if p.get("route") in ROUTES else "both"
    sq = p.get("standalone_question") or question
    queries = [q for q in (p.get("search_queries") or []) if isinstance(q, str)][:3]
    filters = p.get("filters") if isinstance(p.get("filters"), dict) else {}
    return {"route": route, "standalone": sq, "queries": queries, "filters": filters,
            "router_model": model}


def build_context(route: str, standalone: str, queries: list[str], filters: dict):
    blocks, sources, cards, trace = [], [], [], {}
    n = 0
    if route in ("listings", "both"):
        res = LISTINGS.search(filters)
        trace["listings"] = {"filters": res["filters"], "relaxed": res["relaxed"], "total": res["total"]}
        if res.get("missing"):
            m = res["missing"]
            blocks.append(f"Listing search: the collected Wasalt sample has no listings in "
                          f"'{m['city']}'. Cities with listings in the sample: "
                          f"{', '.join(m['available_cities']) or 'none'}. Tell the user this and "
                          f"suggest browsing wasalt.sa directly for {m['city']}.")
        if res["rows"]:
            lines = []
            for r in res["rows"]:
                n += 1
                lines.append(listing_line(n, r))
                sources.append({"n": n, "title": r["title"], "url": r["url"], "source": "Wasalt",
                                "kind": "listing"})
                cards.append({"n": n, **{k: r.get(k) for k in (
                    "id", "title", "url", "type", "purpose", "city", "district", "price",
                    "currency", "rent_frequency", "beds", "baths", "area_sqm")}})
            note = f"Wasalt listings matching the filters (showing {len(lines)} of {res['total']} matches)"
            if res["relaxed"]:
                note += f"; no exact match, so these filters were relaxed: {', '.join(res['relaxed'])}"
            blocks.append(note + ":\n" + "\n".join(lines))
    if route in ("docs", "both") or (route == "listings" and not cards):
        res = KB.search(standalone, queries)
        trace["retrieval"] = res["trace"]
        for h in res["hits"]:
            c = h["chunk"]
            n += 1
            label = f"{SOURCE_LABEL.get(c.source, c.source)} · {c.kind}"
            blocks.append(f"[{n}] ({label}) {c.title or ''} — {c.url}\n{c.text}")
            sources.append({"n": n, "title": c.title or c.url, "url": c.url,
                            "source": SOURCE_LABEL.get(c.source, c.source), "kind": c.kind})
    return "\n\n".join(blocks), sources, cards, trace


def run_turn(session_id: str, question: str) -> Iterator[dict]:
    t0 = time.time()
    question = question.strip()[:2000]
    history = memory.recent(session_id)
    try:
        p = plan(question, history)
    except llm.LLMUnavailable as e:
        yield {"type": "error", "message": "The language model is temporarily unavailable "
               "(free-tier limit or outage). Please try again in a minute.", "detail": str(e)[:200]}
        return

    route = p["route"]
    context, sources, cards, trace = ("", [], [], {})
    if route not in ("chitchat", "off_topic"):
        context, sources, cards, trace = build_context(route, p["standalone"], p["queries"], p["filters"])
    yield {"type": "meta", "route": route, "standalone": p["standalone"], "sources": sources,
           "listings": cards, "trace": trace, "plan_ms": int((time.time() - t0) * 1000)}

    if route in ("chitchat", "off_topic"):
        msgs = [{"role": "system", "content": CHITCHAT_SYSTEM}, *history,
                {"role": "user", "content": question}]
    else:
        msgs = [{"role": "system", "content": ANSWER_SYSTEM}, *history,
                {"role": "user", "content":
                    f"Context:\n{context or '(nothing relevant was found)'}\n\n"
                    f"Question: {question}\n(Standalone form: {p['standalone']})"}]
    answer, model = [], None
    try:
        for tok, model in llm.stream(msgs):
            answer.append(tok)
            yield {"type": "token", "text": tok}
    except llm.LLMUnavailable as e:
        yield {"type": "error", "message": "The language model is temporarily unavailable "
               "(free-tier limit or outage). Please try again in a minute.", "detail": str(e)[:200]}
        return
    full = "".join(answer).strip()
    memory.add(session_id, "user", question)
    memory.add(session_id, "assistant", full)
    yield {"type": "done", "model": model, "router_model": p["router_model"],
           "ms": int((time.time() - t0) * 1000)}
