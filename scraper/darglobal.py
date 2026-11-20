"""Scrape darglobal.co.uk: discover pages from its sitemap, fetch each, keep the
page text plus the CMS JSON the page ships with (minus site-wide boilerplate)."""
from __future__ import annotations

import asyncio
import re

from playwright.async_api import async_playwright

from .browser import UA, crawl

BASE = "https://darglobal.co.uk"

# Utility / transactional pages that carry no useful knowledge.
SKIP = re.compile(
    r"/(ar|landing-page|landingpage|thank-you|success|pay-online|bookYourUnit|"
    r"internship-application|new-home-page|win-a-trip-to-dubai|campaigns|"
    r"broker-contest|dg-circle|exclusive-membership)(/|$)", re.I)

# Site-wide JSON blobs repeated on every page.
DROP_KEYS = {"otherProjectData", "thankyouPageData", "isMobile"}


def page_type(url: str) -> str:
    path = url.replace(BASE, "") or "/"
    for prefix, kind in (("/press/", "press"), ("/blog/", "blog"), ("/insights/", "insight"),
                         ("/partners/", "partner"), ("/one-of-one/", "one_of_one")):
        if path.startswith(prefix):
            return kind
    return "page"  # project pages are detected later from their JSON


async def discover() -> list[str]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await (await browser.new_context(user_agent=UA)).new_page()
        await page.goto(BASE + "/", wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(3000)
        await page.goto(BASE + "/sitemap.xml", wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(2000)
        xml = await page.content()
        await browser.close()
    urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml) or \
        re.findall(r"https://darglobal\.co\.uk/[^\s<\"]*", xml)
    seen, out = set(), []
    for u in urls:
        u = u.rstrip("/")
        if u.startswith(BASE) and not SKIP.search(u) and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def transform(rec: dict) -> dict:
    nd = rec.pop("next_data", None) or {}
    props = (nd.get("props") or {}).get("pageProps") or {}
    gd = props.pop("globalData", None)
    # keep the project directory (name, country) - it is the only useful part of globalData
    if gd:
        attrs = (gd.get("attributes") or {})
        props["_projectList"] = attrs.get("ProjectList")
    for k in DROP_KEYS:
        props.pop(k, None)
    rec["page_props"] = props
    rec["kind"] = "project" if props.get("projectDetailsData") else page_type(rec["url"])
    return rec


async def main():
    urls = await discover()
    print(f"[darglobal] sitemap gave {len(urls)} useful urls")
    await crawl("darglobal", urls, transform, workers=2, delay_s=1.5, settle_ms=1200)


if __name__ == "__main__":
    asyncio.run(main())
