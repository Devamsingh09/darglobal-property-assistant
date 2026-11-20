"""Polite Playwright fetcher shared by both site scrapers.

- One normal Chromium instance, no stealth tricks.
- Fixed delay between requests per worker, images/fonts/media not downloaded.
- Raw results appended to data/raw/<source>.jsonl; already-fetched URLs are skipped,
  so a crashed or interrupted run resumes where it stopped.
- If a bot-check page comes back, back off and retry once; after N consecutive
  blocks the scraper stops instead of pushing harder.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from playwright.async_api import async_playwright, Page

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")
NEXT_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)


class Blocked(Exception):
    """Raised when the site keeps serving bot-check pages."""


def is_challenge(title: str, html: str) -> bool:
    return ("Just a moment" in title
            or "_Incapsula_Resource" in html[:3000]
            or "Performing security verification" in html[:20000])


@dataclass
class RawStore:
    source: str
    path: Path = field(init=False)
    done: set = field(default_factory=set)

    def __post_init__(self):
        self.path = RAW_DIR / f"{self.source}.jsonl"
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("status") == 200:
                        self.done.add(rec["url"])

    def append(self, rec: dict):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if rec.get("status") == 200:
            self.done.add(rec["url"])


async def _block_heavy(route):
    if route.request.resource_type in ("image", "media", "font"):
        await route.abort()
    else:
        await route.continue_()


async def fetch_page(page: Page, url: str, settle_ms: int) -> dict:
    resp = await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    await page.wait_for_timeout(settle_ms)
    html = await page.content()
    title = await page.title()
    status = resp.status if resp else 0
    if is_challenge(title, html):
        return {"url": url, "status": 403, "challenge": True, "title": title}
    m = NEXT_RE.search(html)
    next_data = json.loads(m.group(1)) if m else None
    text = await page.evaluate(
        "() => { const m = document.querySelector('main'); return (m || document.body).innerText; }")
    return {"url": url, "status": status, "title": title, "text": text, "next_data": next_data}


async def crawl(source: str, urls: list[str], transform, *, workers: int = 1,
                delay_s: float = 2.0, settle_ms: int = 1500, max_blocks: int = 3,
                backoff_s: float = 60.0) -> int:
    """Fetch every url not already stored. `transform(rec)` slims the record
    (e.g. strips personal data / boilerplate JSON) before it is written."""
    store = RawStore(source)
    todo = [u for u in urls if u not in store.done]
    print(f"[{source}] {len(urls)} urls, {len(store.done)} cached, {len(todo)} to fetch")
    if not todo:
        return 0
    queue: asyncio.Queue = asyncio.Queue()
    for u in todo:
        queue.put_nowait(u)
    state = {"blocks": 0, "ok": 0, "stop": False}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent=UA, locale="en-US",
                                        viewport={"width": 1366, "height": 900})
        await ctx.route("**/*", _block_heavy)

        async def worker(wid: int):
            page = await ctx.new_page()
            while not state["stop"]:
                try:
                    url = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                for attempt in (1, 2):
                    try:
                        rec = await fetch_page(page, url, settle_ms)
                    except Exception as e:  # timeouts, navigation errors
                        rec = {"url": url, "status": 0, "error": str(e)[:200]}
                    if rec.get("challenge") and attempt == 1:
                        print(f"[{source}] bot check on {url}, backing off {backoff_s:.0f}s")
                        await asyncio.sleep(backoff_s)
                        continue
                    break
                if rec.get("challenge"):
                    state["blocks"] += 1
                    if state["blocks"] >= max_blocks:
                        print(f"[{source}] {max_blocks} bot checks in a row - stopping politely")
                        state["stop"] = True
                else:
                    state["blocks"] = 0
                rec["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                if rec.get("status") == 200:
                    rec = transform(rec)
                    state["ok"] += 1
                store.append(rec)
                n = state["ok"]
                if n % 20 == 0 and rec.get("status") == 200:
                    print(f"[{source}] {n}/{len(todo)} fetched")
                await asyncio.sleep(delay_s + random.uniform(0, delay_s * 0.5))
            await page.close()

        await asyncio.gather(*(worker(i) for i in range(workers)))
        await browser.close()
    print(f"[{source}] done: {state['ok']} new pages")
    if state["stop"]:
        raise Blocked(source)
    return state["ok"]
