"""LLM access over OpenAI-compatible APIs (Groq + OpenRouter) with a fallback chain.

Chain entries are "provider:model", e.g. "groq:openai/gpt-oss-120b" or
"openrouter:nvidia/nemotron-3.5-super-vl:free". Entries whose provider has no API key
are skipped, so the app runs with either key or both.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Iterator

from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError

from . import config

log = logging.getLogger("llm")

PROVIDERS = {
    "groq": (config.GROQ_BASE_URL, lambda: config.GROQ_API_KEY),
    "openrouter": (config.OPENROUTER_BASE_URL, lambda: config.OPENROUTER_API_KEY),
}
_clients: dict[str, OpenAI] = {}
_chain: list[str] = list(config.MODEL_CHAIN)          # answer chain (bigger models first)
_router_chain: list[str] = list(config.ROUTER_CHAIN)  # router chain (small, fast models first)
_cooldown: dict[str, float] = {}       # "provider:model" -> unix time it may be retried
COOLDOWN_S = 60
DAILY_COOLDOWN_S = 1800                # provider said "per day": don't retry for a while
RETRY_PAUSE_S = 3


class LLMUnavailable(RuntimeError):
    pass


def split(entry: str) -> tuple[str, str]:
    provider, _, model = entry.partition(":")
    if provider not in PROVIDERS:       # bare OpenRouter id (e.g. from MODEL_CHAIN env)
        return "openrouter", entry
    return provider, model


def configured(entry: str) -> bool:
    return bool(PROVIDERS[split(entry)[0]][1]())


def any_configured() -> bool:
    return any(configured(e) for e in _chain + _router_chain)


def client(provider: str) -> OpenAI:
    if provider not in _clients:
        base_url, key = PROVIDERS[provider][0], PROVIDERS[provider][1]()
        if not key:
            raise LLMUnavailable(f"no API key for {provider}")
        headers = ({"HTTP-Referer": "https://darglobal-property-assistant.vercel.app",
                    "X-Title": "Property Assistant (DarGlobal x Wasalt demo)"}
                   if provider == "openrouter" else None)
        _clients[provider] = OpenAI(base_url=base_url, api_key=key, timeout=config.LLM_TIMEOUT_S,
                                    max_retries=0, default_headers=headers)
    return _clients[provider]


def _extra(provider: str, model: str) -> dict:
    # gpt-oss models reason before answering; "low" keeps latency and token use down.
    # Groq returns the reasoning in a separate field, so `content` is just the answer.
    if provider == "groq" and model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    return {}


def refresh_free_models() -> None:
    """If configured OpenRouter models vanished, append whatever free ones exist now."""
    if not config.OPENROUTER_API_KEY:
        return
    try:
        models = client("openrouter").models.list().data
    except Exception as e:  # pragma: no cover - network
        log.warning("could not list models: %s", e)
        return
    for m in (m.id for m in models if m.id.endswith(":free")):
        if f"openrouter:{m}" not in _chain:
            _chain.append(f"openrouter:{m}")


def _candidates(chain: list[str]) -> list[str]:
    usable = [m for m in chain if configured(m)]
    if not usable:
        raise LLMUnavailable("no LLM API key is configured (GROQ_API_KEY or OPENROUTER_API_KEY)")
    now = time.time()
    return [m for m in usable if _cooldown.get(m, 0) <= now] or usable


def _retryable(e: Exception) -> bool:
    if isinstance(e, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(e, APIStatusError):
        return e.status_code in (400, 402, 404, 408, 413, 429) or e.status_code >= 500
    return False


def _fail(entry: str, e: Exception) -> None:
    msg = str(e)
    daily = "per-day" in msg or "per day" in msg or "tokens per day" in msg.lower()
    _cooldown[entry] = time.time() + (DAILY_COOLDOWN_S if daily else COOLDOWN_S)
    log.warning("model %s failed: %s", entry, msg[:200])


def complete(messages: list[dict], *, temperature=0.2, max_tokens=700,
             validate=None, router=False) -> tuple[str, str]:
    """Non-streaming call. Returns (text, "provider:model").

    validate(text) -> bool: a reply it rejects (e.g. truncated JSON) moves on to the next model.
    router=True uses the small-model chain."""
    chain = _router_chain if router else _chain
    last_err = None
    for attempt in range(2):
        for entry in _candidates(chain):
            provider, model = split(entry)
            try:
                r = client(provider).chat.completions.create(
                    model=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, extra_body=_extra(provider, model) or None)
                text = (r.choices[0].message.content or "").strip()
                if text and (validate is None or validate(text)):
                    return text, entry
                last_err = RuntimeError(f"{entry} returned an unusable reply")
                log.warning("model %s returned an unusable reply: %r", entry, text[:120])
            except Exception as e:
                if not _retryable(e):
                    raise
                last_err = e
                _fail(entry, e)
        if attempt == 0:
            refresh_free_models()
            time.sleep(RETRY_PAUSE_S)  # upstream 429s usually clear within seconds
    raise LLMUnavailable(f"all models failed: {last_err}")


def stream(messages: list[dict], *, temperature=0.3, max_tokens=900) -> Iterator[tuple[str, str]]:
    """Yields (token, "provider:model"). Falls back to the next model only before the first token."""
    last_err = None
    for attempt in range(2):
        if attempt:
            time.sleep(RETRY_PAUSE_S)
        for entry in _candidates(_chain):
            provider, model = split(entry)
            started = False
            try:
                resp = client(provider).chat.completions.create(
                    model=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, stream=True,
                    extra_body=_extra(provider, model) or None)
                for ev in resp:
                    if not ev.choices:
                        continue
                    delta = ev.choices[0].delta.content or ""
                    if delta:
                        started = True
                        yield delta, entry
                if started:
                    return
                last_err = RuntimeError(f"{entry} streamed nothing")
            except Exception as e:
                if started or not _retryable(e):
                    raise
                last_err = e
                _fail(entry, e)
    raise LLMUnavailable(f"all models failed: {last_err}")


CITE_RE = re.compile(r"【\s*(\d+)\s*(?:†[^】]*)?】")


def clean_citations(text: str) -> str:
    """gpt-oss sometimes cites as 【1†L1-L4】; normalise to [1]."""
    return CITE_RE.sub(lambda m: "[" + m.group(1) + "]", text)


def parse_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply (handles ```json fences, prose)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start = text.find("{")
    if start < 0:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    raw = m.group(0) if m else text[start:]
    for candidate in (raw, re.sub(r",\s*([}\]])", r"\1", raw)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return _repair_truncated(text[start:])


def _repair_truncated(s: str) -> dict | None:
    """A reply cut off by max_tokens: keep the longest prefix that ends at a structural
    character, close whatever brackets are still open, and parse that."""
    stack, cuts, in_str, esc = [], [], False, False
    for i, ch in enumerate(s):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
            cuts.append((i + 1, "".join(reversed(stack))))
        elif ch in "}]":
            if stack:
                stack.pop()
            cuts.append((i + 1, "".join(reversed(stack))))
        elif ch == ",":
            cuts.append((i, "".join(reversed(stack))))
    for end, closers in reversed(cuts[-40:]):
        try:
            out = json.loads(s[:end] + closers)
        except json.JSONDecodeError:
            continue
        if isinstance(out, dict):
            return out
    return None
