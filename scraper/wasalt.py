"""Scrape wasalt.sa (English pages only).

Discovery uses Wasalt's public sitemaps (on its CDN) and city result pages
(/en/properties-for-<sale|rent>-in-<city>), which carry ~32 listings each.
/search is disallowed in robots.txt and is never touched.

Only listing facts are kept. Agent / owner / contact fields are dropped
before anything is written to disk.
"""
from __future__ import annotations

import asyncio
import gzip
import random
import re

import httpx

from .browser import crawl

BASE = "https://wasalt.sa"
SITEMAP = "https://cdn.wasalt.sa/sitemap/{name}.xml.gz"

CITY_PAGES = {  # city slug -> how many result pages to take per purpose
    "riyadh": 8, "jeddah": 6, "makkah": 3, "madinah": 3, "dammam": 3, "al-khobar": 3,
}
DETAIL_SAMPLE = 120     # random sale listing pages (full descriptions)
RENTAL_SAMPLE = 25      # random daily-rental pages
EXTRA_PAGES = ["/en", "/en/about", "/en/corporate", "/en/auction", "/en/projects"]

INFO_KEYS = ["title", "propertyName", "city", "district", "territory", "zone", "address",
             "propertyMainType", "propertySubType", "propertyFor", "salePrice", "expectedRent",
             "rentFreq", "currencyType", "conversionPrice", "furnishingType", "facingType",
             "projectName", "description", "tags", "slug"]
EXTRA_ATTR_KEYS = {"completionYear", "propertyFacade", "electricityMeter", "waterMeter",
                   "propertyMainType"}


def sitemap_urls(name: str) -> list[str]:
    r = httpx.get(SITEMAP.format(name=name), timeout=60,
                  headers={"User-Agent": "Mozilla/5.0 (compatible; research-scraper)"})
    r.raise_for_status()
    raw = r.content
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    return re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", raw.decode("utf-8", "ignore"))


def slim_property(p: dict) -> dict:
    info = p.get("propertyInfo") or {}
    out = {k: info.get(k) for k in INFO_KEYS if info.get(k) not in (None, "", [])}
    out["id"] = p.get("id")
    out["published_at"] = p.get("publishedAt") or p.get("createdAt")
    out["is_verified"] = p.get("isVerified")
    out["rega_licensed"] = p.get("isRegaProp")
    out["attributes"] = {a.get("key"): a.get("value") for a in (p.get("attributes") or [])
                         if a.get("key")}
    if p.get("floorSize"):
        out["attributes"].setdefault("builtUpArea", p["floorSize"])
    for a in p.get("additionalAttributes") or []:
        if a.get("key") in EXTRA_ATTR_KEYS and a.get("value") not in (None, "", "-"):
            out["attributes"][a["key"]] = a["value"]
    loc = p.get("location") or {}
    if loc.get("lat"):
        out["lat"], out["lon"] = loc.get("lat"), loc.get("lon")
    amen = p.get("featureAmenities") or []
    names = [a.get("name") or a.get("title") for a in amen if isinstance(a, dict)]
    if any(names):
        out["amenities"] = [n for n in names if n]
    return out


def transform(rec: dict) -> dict:
    nd = rec.pop("next_data", None) or {}
    pp = (nd.get("props") or {}).get("pageProps") or {}
    listings = []
    if pp.get("propertyDetailsV3"):
        d = slim_property(pp["propertyDetailsV3"])
        d["_detail"] = True
        listings.append(d)
    for key in ("searchResult", "similarProperty"):
        for p in ((pp.get(key) or {}).get("properties") or []):
            if isinstance(p, dict) and p.get("propertyInfo"):
                listings.append(slim_property(p))
    rec["listings"] = listings
    meta = pp.get("metaData") or {}
    rec["meta"] = {k: meta.get(k) for k in ("meta_title", "meta_description", "h1tag") if meta.get(k)}
    faq = pp.get("FAQSchema") or {}
    rec["faq"] = [{"q": q.get("name"), "a": (q.get("acceptedAnswer") or {}).get("text")}
                  for q in faq.get("mainEntity") or [] if isinstance(q, dict)]
    if listings and not pp.get("propertyDetailsV3"):
        rec["text"] = ""  # result pages: the structured listings are the content
    rec["kind"] = ("listing" if pp.get("propertyDetailsV3") else
                   "results" if listings else "page")
    return rec


def build_url_list() -> list[str]:
    rng = random.Random(42)
    urls = []
    # most valuable first: page 1 of every city/purpose, then deeper pages
    for n in range(1, max(CITY_PAGES.values()) + 1):
        for city, pages in CITY_PAGES.items():
            if n > pages:
                continue
            for purpose in ("sale", "rent"):
                u = f"{BASE}/en/properties-for-{purpose}-in-{city}"
                urls.append(u if n == 1 else f"{u}?page={n}")
    sale = sitemap_urls("product_sitemap_en_sa")
    rental = sitemap_urls("rental_pdp_sitemap_en_sa")
    static = sitemap_urls("static_sitemap_en_sa")
    print(f"[wasalt] sitemaps: {len(sale)} sale, {len(rental)} rental, {len(static)} static")
    urls += [BASE + p for p in EXTRA_PAGES] + static
    urls += rng.sample(sale, min(DETAIL_SAMPLE, len(sale)))
    urls += rng.sample(rental, min(RENTAL_SAMPLE, len(rental)))
    # keep sitemap slugs as an offline fallback inventory
    from .browser import RAW_DIR
    (RAW_DIR / "wasalt_sitemap_sale.txt").write_text("\n".join(sale), encoding="utf-8")
    return list(dict.fromkeys(urls))


async def main():
    import os
    urls = build_url_list()
    from .browser import Blocked
    try:
        await crawl("wasalt", urls, transform, workers=1,
                    delay_s=float(os.getenv("WASALT_DELAY_S", "6")), settle_ms=2500,
                    max_blocks=3, backoff_s=90)
    except Blocked:
        print("[wasalt] stopped after repeated bot checks; keeping what was collected")


if __name__ == "__main__":
    asyncio.run(main())
