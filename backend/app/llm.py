"""OpenRouter client (OpenAI-compatible) with a free-model fallback chain."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Iterator

from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError

from . import config

log = logging.getLogger("llm")

_client: OpenAI | None = None
_chain: list[str] = list(config.MODEL_CHAIN)
_cooldown: dict[str, float] = {}       # model -> unix time it may be retried
COOLDOWN_S = 60
RETRY_PAUSE_S = 3


class LLMUnavailable(RuntimeError):
    pass


def client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENROUTER_API_KEY:
            raise LLMUnavailable("OPENROUTER_API_KEY is not set")
        _client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY,
            timeout=config.LLM_TIMEOUT_S, max_retries=0,
            default_headers={"HTTP-Referer": "https://huggingface.co/spaces",
                             "X-Title": "Property Assistant (DarGlobal x Wasalt demo)"})
    return _client


def refresh_free_models() -> None:
    """If configured models vanished from OpenRouter, append whatever free ones exist now."""
    try:
        models = client().models.list().data
    except Exception as e:  # pragma: no cover - network
        log.warning("could not list models: %s", e)
        return
    free = [m.id for m in models if m.id.endswith(":free")]
    for m in free:
        if m not in _chain:
            _chain.append(m)
    log.info("model chain now: %s", _chain)


def _candidates() -> list[str]:
    now = time.time()
    ready = [m for m in _chain if _cooldown.get(m, 0) <= now]
    return ready or list(_chain)


def _retryable(e: Exception) -> bool:
    if isinstance(e, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(e, APIStatusError):
        return e.status_code in (402, 404, 408, 429) or e.status_code >= 500
    return False


def complete(messages: list[dict], *, temperature=0.2, max_tokens=700,
             validate=None) -> tuple[str, str]:
    """Non-streaming call. Returns (text, model_used).

    validate(text) -> bool: a reply it rejects (e.g. truncated JSON) moves on to the next model."""
    last_err = None
    refreshed = False
    for attempt in range(2):
        for model in _candidates():
            try:
                r = client().chat.completions.create(
                    model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
                text = (r.choices[0].message.content or "").strip()
                if text and (validate is None or validate(text)):
                    return text, model
                last_err = RuntimeError(f"{model} returned an unusable reply")
                log.warning("model %s returned an unusable reply: %r", model, text[:120])
            except Exception as e:
                if not _retryable(e):
                    raise
                last_err = e
                log.warning("model %s failed: %s", model, str(e)[:160])
                _cooldown[model] = time.time() + COOLDOWN_S
        if not refreshed:
            refresh_free_models()
            refreshed = True
        time.sleep(RETRY_PAUSE_S)  # upstream free-tier 429s usually clear within seconds
    raise LLMUnavailable(f"all models failed: {last_err}")


def stream(messages: list[dict], *, temperature=0.3, max_tokens=900) -> Iterator[tuple[str, str]]:
    """Yields (token, model). Falls back to the next model only before the first token."""
    last_err = None
    for attempt in range(2):
        if attempt:
            time.sleep(RETRY_PAUSE_S)
        for model in _candidates():
            started = False
            try:
                resp = client().chat.completions.create(
                    model=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, stream=True)
                for ev in resp:
                    if not ev.choices:
                        continue
                    delta = ev.choices[0].delta.content or ""
                    if delta:
                        started = True
                        yield delta, model
                if started:
                    return
                last_err = RuntimeError(f"{model} streamed nothing")
            except Exception as e:
                if started or not _retryable(e):
                    raise
                last_err = e
                log.warning("stream model %s failed: %s", model, str(e)[:160])
                _cooldown[model] = time.time() + COOLDOWN_S
    raise LLMUnavailable(f"all models failed: {last_err}")


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
