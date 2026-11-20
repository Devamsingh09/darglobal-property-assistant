from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import config, llm, memory
from .chat import run_turn
from .listings import LISTINGS
from .retrieval import KB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
app = FastAPI(title="Property Assistant API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=config.ALLOWED_ORIGINS, allow_methods=["*"],
                   allow_headers=["*"])


def _load():
    LISTINGS.load()
    KB.load()


@app.on_event("startup")
def startup():
    # load models in the background so the Space reports "running" quickly
    threading.Thread(target=_load, daemon=True).start()


class ChatIn(BaseModel):
    session_id: str = Field(..., min_length=6, max_length=64)
    message: str = Field(..., min_length=1, max_length=2000)


RATE_WINDOW_S, RATE_MAX = 600, 30           # per client IP: 30 questions / 10 min
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _rate_limit(request: Request):
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    now = time.time()
    with _hits_lock:
        q = _hits[ip]
        while q and now - q[0] > RATE_WINDOW_S:
            q.popleft()
        if len(q) >= RATE_MAX:
            raise HTTPException(429, "Too many questions in a short time - please wait a few minutes.")
        q.append(now)


def _check(req: ChatIn, request: Request):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", req.session_id):
        raise HTTPException(400, "invalid session_id")
    if not KB.ready:
        raise HTTPException(503, "Knowledge base is still loading, try again in a few seconds.")
    _rate_limit(request)


@app.get("/")
def root():
    return {"service": "Property Assistant API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    manifest = {}
    try:
        manifest = json.loads((config.DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"ready": KB.ready, "llm_configured": llm.any_configured(),
            "providers": {p: llm.configured(p + ":x") for p in llm.PROVIDERS},
            "models": [m for m in config.MODEL_CHAIN if llm.configured(m)],
            "router_models": [m for m in config.ROUTER_CHAIN if llm.configured(m)],
            "data": manifest}


@app.post("/api/chat")
def chat_stream(req: ChatIn, request: Request):
    """Server-sent events: meta -> token* -> done | error."""
    _check(req, request)

    def gen():
        try:
            for ev in run_turn(req.session_id, req.message):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:  # never leave the client hanging
            logging.exception("turn failed")
            yield "data: " + json.dumps({"type": "error", "message": "Something went wrong on the server.",
                                         "detail": str(e)[:200]}) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/chat/sync")
def chat_sync(req: ChatIn, request: Request):
    """Same as /api/chat but returns one JSON object (handy for curl and tests)."""
    _check(req, request)
    out = {"answer": ""}
    for ev in run_turn(req.session_id, req.message):
        if ev["type"] == "token":
            out["answer"] += ev["text"]
        else:
            out.update({k: v for k, v in ev.items() if k != "type"})
            if ev["type"] == "error":
                out["error"] = ev["message"]
    out["answer"] = llm.clean_citations(out["answer"])
    return out


@app.delete("/api/session/{session_id}")
def reset(session_id: str):
    memory.clear(session_id)
    return {"ok": True}
