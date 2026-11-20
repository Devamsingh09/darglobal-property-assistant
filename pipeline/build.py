"""Turn raw scraped JSONL into the knowledge base the backend serves.

Outputs (backend/data/):
  kb.db          SQLite: listings, projects, chunks
  vectors.npy    float16 [n_chunks, 384], L2-normalised, row i = chunks.rowid i+1
  manifest.json  counts, build time, model names
"""
from __future__ import annotations

import collections
import json
import re
import sqlite3
import sys
import time
from html import unescape
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "backend" / "data"
MODEL_DIR = ROOT / "backend" / "models"
sys.path.insert(0, str(ROOT / "backend"))
from app.config import EMBED_MODEL  # noqa: E402  (single source of truth for model name)

CHUNK_CHARS, OVERLAP_CHARS = 1100, 150


# ---------------------------------------------------------------- helpers
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == 200:
                out[r["url"]] = r  # last successful fetch wins
    return list(out.values())


def strip_html(s: str | None) -> str:
    if not s:
        return ""
    s = re.sub(r"<(br|/p|/li|/h\d)[^>]*>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = unescape(s)
    return re.sub(r"[ \t]+", " ", s).strip()


def rich_text(blocks) -> str:
    """Strapi rich-text blocks -> plain text."""
    if isinstance(blocks, str):
        return strip_html(blocks)
    out = []
    def walk(n):
        if isinstance(n, dict):
            if "text" in n and isinstance(n["text"], str):
                out.append(n["text"])
            for c in n.get("children") or []:
                walk(c)
            if n.get("type") in ("paragraph", "list-item", "heading"):
                out.append("\n")
        elif isinstance(n, list):
            for c in n:
                walk(c)
    walk(blocks)
    return re.sub(r"\n\s*\n+", "\n", "".join(out)).strip()


def chunk_text(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > CHUNK_CHARS:  # very long paragraph: hard split on sentence
            cut = p.rfind(". ", 0, CHUNK_CHARS)
            cut = cut + 1 if cut > CHUNK_CHARS // 2 else CHUNK_CHARS
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:cut].strip())
            p = p[cut:].strip()
        if len(cur) + len(p) + 1 > CHUNK_CHARS and cur:
            chunks.append(cur)
            cur = cur[-OVERLAP_CHARS:].split(" ", 1)[-1] + "\n" + p
        else:
            cur = (cur + "\n" + p).strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) > 60]


def to_number(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"[\d,.]+", str(v))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


COUNTRY_HINTS = [
    (r"dubai|rak|ras al khaimah|marjan|abu dhabi|uae|emirates", "United Arab Emirates"),
    (r"muscat|oman|aida|sohar", "Oman"),
    (r"riyadh|jeddah|jeddha|ksa|saudi|makkah|madinah|wadi safar|dammam|khobar", "Saudi Arabia"),
    (r"doha|qatar|simaisma", "Qatar"),
    (r"marbella|spain|costa del sol|estepona|benahav|casares", "Spain"),
    (r"london|united kingdom|\buk\b", "United Kingdom"),
    (r"maldives", "Maldives"),
]


def guess_country(*texts) -> str | None:
    t = " ".join(x for x in texts if x).lower()
    for pat, country in COUNTRY_HINTS:
        if re.search(pat, t):
            return country
    return None


# ---------------------------------------------------------------- DarGlobal
def boilerplate_lines(records: list[dict], min_share=0.25) -> set[str]:
    cnt = collections.Counter()
    for r in records:
        cnt.update({ln.strip() for ln in (r.get("text") or "").splitlines() if ln.strip()})
    n = max(len(records), 1)
    return {ln for ln, c in cnt.items() if c / n >= min_share}


def clean_page_text(text: str, boiler: set[str]) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    keep = [ln for ln in lines if ln and ln not in boiler and not re.fullmatch(r"[\W\d_]{0,3}", ln)]
    return "\n".join(keep)


def parse_project(rec: dict) -> dict | None:
    a = ((rec["page_props"].get("projectDetailsData") or {}).get("attributes")) or {}
    P = a.get("ProjectDetails") or {}
    if not P:
        return None
    banner = P.get("ProjectBanner") or {}
    about = P.get("AboutProject") or {}
    facts = {d.get("name", "").strip(): (d.get("value") or "").strip()
             for d in about.get("AboutDetailsType") or [] if d.get("name")}
    amen = [x.get("name") for x in (P.get("Amenities") or {}).get("amenitiesList") or [] if x.get("name")]
    invest = [f"{(x.get('name') or '').rstrip(':')}: {x.get('value') or ''}".strip(": ")
              for x in (P.get("whyInvest") or {}).get("investlist") or []]
    faqs = []
    for block in a.get("faqContent") or []:
        for f in block.get("faqs") or []:
            q, ans = f.get("question"), rich_text(f.get("answer"))
            if q and ans:
                faqs.append((q.strip(), ans))
    seo = P.get("SEO") or {}
    name = (a.get("title") or banner.get("Heading") or rec["title"]).strip()
    location = (banner.get("Location") or facts.get("Location") or "").strip()
    sp = banner.get("StartingPrice")
    starting = sp if isinstance(sp, str) else (json.dumps(sp) if sp else "")
    return {
        "slug": a.get("slug"), "name": name, "url": rec["url"], "location": location,
        "country": guess_country(location, name),
        "property_type": facts.get("Property Type"), "status": facts.get("Status"),
        "units": facts.get("Units") or facts.get("Unit Types") or facts.get("Bedrooms"),
        "area": next((v for k, v in facts.items() if k.lower().startswith("area")), None),
        "starting_price": starting or facts.get("Starting Price") or facts.get("Price"),
        "amenities": ", ".join(amen), "why_invest": invest, "faqs": faqs,
        "summary": strip_html(about.get("description")) or seo.get("metaDescription") or "",
        "location_text": strip_html((P.get("loaction") or {}).get("description")),
        "facts": facts,
    }


def project_card(p: dict) -> str:
    lines = [f"{p['name']} — a DarGlobal project", f"Location: {p['location']}"
             + (f" ({p['country']})" if p.get("country") else "")]
    for label, key in (("Property type", "property_type"), ("Status", "status"),
                       ("Units", "units"), ("Area", "area"), ("Starting price", "starting_price")):
        if p.get(key):
            lines.append(f"{label}: {p[key]}")
    for k, v in p["facts"].items():
        if k not in ("Property Type", "Status", "Units", "Location") and not k.lower().startswith("area") and v:
            lines.append(f"{k}: {v}")
    if p["summary"]:
        lines.append(p["summary"])
    if p["amenities"]:
        lines.append(f"Amenities: {p['amenities']}")
    if p["why_invest"]:
        lines.append("Why invest: " + "; ".join(p["why_invest"]))
    if p["location_text"]:
        lines.append("About the location: " + p["location_text"])
    return "\n".join(lines)


def build_darglobal(chunks: list, projects: list):
    recs = read_jsonl(RAW / "darglobal.jsonl")
    boiler = boilerplate_lines(recs)
    print(f"[build] darglobal: {len(recs)} pages, {len(boiler)} boilerplate lines removed")
    for r in recs:
        kind = r.get("kind", "page")
        title = re.sub(r"\s*\|\s*DarGlobal.*$", "", r.get("title") or "").strip()
        if kind == "project":
            p = parse_project(r)
            if p and p["facts"]:
                projects.append(p)
                # split long cards so the reranker (512-token window) sees every section
                for part in chunk_text(project_card(p)):
                    if not part.startswith(p["name"]):
                        part = f"{p['name']} ({p['location']}):\n{part}"
                    chunks.append(dict(source="darglobal", kind="project", url=r["url"],
                                       title=p["name"], text=part))
                for q, ans in p["faqs"]:
                    chunks.append(dict(source="darglobal", kind="faq", url=r["url"],
                                       title=p["name"], text=f"Q: {q}\nA: {ans}"))
        text = clean_page_text(r.get("text"), boiler)
        if len(text) < 200:
            continue
        for c in chunk_text(text):
            chunks.append(dict(source="darglobal", kind=kind, url=r["url"], title=title, text=c))


# ---------------------------------------------------------------- Wasalt
def norm_listing(d: dict, url_hint: str | None) -> dict | None:
    if not d.get("id"):
        return None
    attrs = d.get("attributes") or {}
    purpose = (d.get("propertyFor") or "").lower() or None
    if url_hint and "/dailyrental/" in url_hint:
        purpose = "daily"
    price = to_number(d.get("salePrice")) if purpose == "sale" else to_number(d.get("expectedRent"))
    price = price or to_number(d.get("conversionPrice")) or to_number(d.get("salePrice"))
    # rentFreq is either a string or {"monthly": {"amount":..}, "yearly": {"amount":..}}
    rf, monthly = d.get("rentFreq"), None
    if isinstance(rf, dict):
        yearly = to_number((rf.get("yearly") or {}).get("amount"))
        monthly = to_number((rf.get("monthly") or {}).get("amount"))
        if yearly:
            price, rf = yearly, "year"
        elif monthly:
            price, rf = monthly, "month"
        else:
            rf = None
    elif rf and not isinstance(rf, str):
        rf = None
    slug = d.get("slug")
    url = (f"https://wasalt.sa/en/property/{'rent' if purpose == 'rent' else 'sale'}/{slug}"
           if slug else url_hint)
    return {
        "id": int(d["id"]), "url": url, "title": d.get("title") or d.get("propertyName"),
        "purpose": purpose, "type": d.get("propertySubType"), "usage": d.get("propertyMainType"),
        "city": d.get("city"), "district": d.get("district") or d.get("zone"),
        "region": d.get("territory"), "address": d.get("address"),
        "price": price, "currency": d.get("currencyType") or "SAR",
        "rent_frequency": rf or None, "monthly_rent": monthly,
        "beds": to_number(attrs.get("noOfBedrooms")), "baths": to_number(attrs.get("noOfBathrooms")),
        "area_sqm": to_number(attrs.get("builtUpArea") or attrs.get("plotArea") or attrs.get("landArea")),
        "furnishing": d.get("furnishingType"), "facing": d.get("facingType"),
        "age": attrs.get("completionYear"), "verified": 1 if d.get("is_verified") else 0,
        "rega_licensed": 1 if d.get("rega_licensed") else 0,
        "description": strip_html(d.get("description"))[:1500],
        "amenities": ", ".join(d.get("amenities") or []), "published_at": d.get("published_at"),
        "lat": to_number(d.get("lat")), "lon": to_number(d.get("lon")),
        "_detail": bool(d.get("_detail")),
    }


def listing_text(l: dict) -> str:
    what = l["title"] or f"{l['type']} property"
    where = ", ".join(x for x in (l["district"], l["region"], l["city"]) if x)
    pur = {"sale": "for sale", "rent": "for rent", "daily": "for daily rental"}.get(l["purpose"], "")
    parts = [f"{what} {pur} in {where}, Saudi Arabia (Wasalt listing #{l['id']})."]
    if l["price"]:
        freq = f" per {l['rent_frequency']}" if l["purpose"] == "rent" and l["rent_frequency"] else ""
        parts.append(f"Price: {l['currency']} {l['price']:,.0f}{freq}.")
        if l.get("monthly_rent") and l["rent_frequency"] == "year":
            parts.append(f"(about {l['currency']} {l['monthly_rent']:,.0f} per month)")
    specs = []
    if l["beds"]:
        specs.append(f"{l['beds']:.0f} bedrooms")
    if l["baths"]:
        specs.append(f"{l['baths']:.0f} bathrooms")
    if l["area_sqm"]:
        specs.append(f"{l['area_sqm']:,.0f} sqm")
    if l["type"]:
        specs.append(f"type: {l['type']}")
    if l["usage"]:
        specs.append(f"usage: {l['usage']}")
    if l["furnishing"]:
        specs.append(f"furnishing: {l['furnishing']}")
    if l["age"]:
        specs.append(f"age: {l['age']}")
    if specs:
        parts.append("; ".join(specs) + ".")
    if l["amenities"]:
        parts.append(f"Amenities: {l['amenities']}.")
    if l["description"]:
        parts.append(l["description"][:700])
    return " ".join(parts)


TYPE_WORDS = ["rest house", "apartment", "villa", "floor", "land", "building", "room", "chalet",
              "duplex", "townhouse", "office", "shop", "warehouse", "farm"]


def sitemap_overview() -> list[str]:
    """Aggregate facts from Wasalt's public listing sitemap (no price/city in the URLs,
    so these become an inventory-overview document rather than searchable listings)."""
    path = RAW / "wasalt_sitemap_sale.txt"
    if not path.exists():
        return []
    urls = path.read_text(encoding="utf-8").split()
    purpose = collections.Counter()
    types = collections.Counter()
    beds = collections.defaultdict(collections.Counter)
    for u in urls:
        m = re.search(r"/property/(sale|rent)/(.+)-\d+$", u)
        if not m:
            continue
        purpose[m.group(1)] += 1
        slug = m.group(2).replace("-", " ")
        t = next((w for w in TYPE_WORDS if w in slug), "other")
        types[(m.group(1), t)] += 1
        b = re.search(r"with (\d+) bedrooms?", slug)
        if b:
            beds[t][int(b.group(1))] += 1
    total = sum(purpose.values())
    lines = [f"Wasalt inventory overview, computed from the public wasalt.sa listing sitemap "
             f"(snapshot taken {time.strftime('%d %b %Y')}). The sitemap lists {total:,} property "
             f"listing pages: {purpose['sale']:,} for sale and {purpose['rent']:,} for rent."]
    for p in ("sale", "rent"):
        top = sorted(((t, c) for (pp, t), c in types.items() if pp == p), key=lambda x: -x[1])
        if top:
            lines.append(f"Listings for {p} by property type: " +
                         ", ".join(f"{t}s: {c:,}" for t, c in top if c >= 5) + ".")
    for t in ("apartment", "villa", "floor"):
        if beds[t]:
            dist = ", ".join(f"{k} bedrooms: {v:,}" for k, v in sorted(beds[t].items()) if v >= 20)
            lines.append(f"{t.title()} listings by bedroom count: {dist}.")
    about = (f"What is Wasalt? Wasalt (wasalt.sa, Arabic: وصلت) is a Saudi Arabian real-estate "
             f"platform and app for buying, renting and daily-renting residential and commercial "
             f"property. Its site footer and listing pages show it is licensed by REGA (the Real "
             f"Estate General Authority) as a real-estate portal, and listings carry REGA "
             f"advertising licence details. It also runs real-estate auctions, project pages and a "
             f"'Get Matched' service. It covers cities such as Riyadh, Jeddah, Makkah, Madinah, "
             f"Dammam and Al Khobar, and its public sitemap lists about {total:,} property listing "
             f"pages ({purpose['sale']:,} for sale, {purpose['rent']:,} for rent). DarGlobal has "
             f"partnered with Wasalt to advertise projects such as Trump Tower Jeddah in Saudi Arabia.")
    return [about, "\n".join(lines[:3]), "\n".join(lines[3:])]


def build_wasalt(chunks: list) -> list[dict]:
    recs = read_jsonl(RAW / "wasalt.jsonl")
    by_id: dict[int, dict] = {}
    faq_seen = set()
    for r in recs:
        for d in r.get("listings") or []:
            l = norm_listing(d, r["url"] if d.get("_detail") else None)
            if not l:
                continue
            old = by_id.get(l["id"])
            if old is None or (l["_detail"] and not old["_detail"]):
                by_id[l["id"]] = l
        for f in r.get("faq") or []:
            q, a = (f.get("q") or "").strip(), strip_html(f.get("a"))
            if q and a and q not in faq_seen:
                faq_seen.add(q)
                chunks.append(dict(source="wasalt", kind="faq", url=r["url"],
                                   title=(r.get("meta") or {}).get("h1tag") or r.get("title"),
                                   text=f"Q: {q}\nA: {a}"))
        if r.get("kind") == "page" and len(r.get("text") or "") > 200:
            for c in chunk_text(r["text"]):
                chunks.append(dict(source="wasalt", kind="page", url=r["url"],
                                   title=r.get("title"), text=c))
    listings = list(by_id.values())
    print(f"[build] wasalt: {len(recs)} pages -> {len(listings)} unique listings, "
          f"{len(faq_seen)} FAQs")
    for i, text in enumerate(sitemap_overview()):
        chunks.append(dict(source="wasalt", kind="overview", url="https://wasalt.sa/en",
                           title="About Wasalt" if i == 0 else "Wasalt inventory overview", text=text))
    for l in listings:
        chunks.append(dict(source="wasalt", kind="listing", url=l["url"],
                           title=l["title"], text=listing_text(l), listing_id=l["id"]))
    return listings


# ---------------------------------------------------------------- write
def write_db(chunks, projects, listings):
    OUT.mkdir(parents=True, exist_ok=True)
    db_path = OUT / "kb.db"
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript("""
    CREATE TABLE listings(id INTEGER PRIMARY KEY, url TEXT, title TEXT, purpose TEXT, type TEXT,
      usage TEXT, city TEXT, district TEXT, region TEXT, address TEXT, price REAL, currency TEXT,
      rent_frequency TEXT, beds REAL, baths REAL, area_sqm REAL, furnishing TEXT, facing TEXT,
      age TEXT, verified INTEGER, rega_licensed INTEGER, description TEXT, amenities TEXT,
      published_at TEXT, lat REAL, lon REAL);
    CREATE INDEX ix_l ON listings(city, purpose, type, price);
    CREATE TABLE projects(slug TEXT PRIMARY KEY, name TEXT, url TEXT, location TEXT, country TEXT,
      property_type TEXT, status TEXT, units TEXT, area TEXT, starting_price TEXT,
      amenities TEXT, summary TEXT);
    CREATE TABLE chunks(rowid INTEGER PRIMARY KEY, source TEXT, kind TEXT, url TEXT, title TEXT,
      text TEXT, listing_id INTEGER);
    """)
    cols = ["id", "url", "title", "purpose", "type", "usage", "city", "district", "region",
            "address", "price", "currency", "rent_frequency", "beds", "baths", "area_sqm",
            "furnishing", "facing", "age", "verified", "rega_licensed", "description",
            "amenities", "published_at", "lat", "lon"]
    con.executemany(f"INSERT INTO listings({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                    [tuple(l.get(c) for c in cols) for l in listings])
    seen = set()
    for p in projects:
        if p["slug"] in seen:
            continue
        seen.add(p["slug"])
        con.execute("INSERT INTO projects VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (p["slug"], p["name"], p["url"], p["location"], p["country"],
                     p["property_type"], p["status"], p["units"], p["area"],
                     p["starting_price"], p["amenities"], p["summary"]))
    con.executemany("INSERT INTO chunks(rowid,source,kind,url,title,text,listing_id) "
                    "VALUES(?,?,?,?,?,?,?)",
                    [(i + 1, c["source"], c["kind"], c["url"], c.get("title"), c["text"],
                      c.get("listing_id")) for i, c in enumerate(chunks)])
    con.commit()
    con.close()


def embed(chunks) -> np.ndarray:
    from fastembed import TextEmbedding
    import hashlib
    docs = [f"{c.get('title') or ''}\n{c['text']}" for c in chunks]
    keys = [hashlib.sha1(f"{EMBED_MODEL}|{d}".encode()).hexdigest() for d in docs]
    cache_path = RAW / "embed_cache.npz"
    cache = dict(np.load(cache_path)) if cache_path.exists() else {}
    todo = [i for i, k in enumerate(keys) if k not in cache]
    t = time.time()
    if todo:
        model = TextEmbedding(EMBED_MODEL, cache_dir=str(MODEL_DIR))
        new = np.array(list(model.embed([docs[i] for i in todo], batch_size=64)), dtype=np.float32)
        for i, v in zip(todo, new):
            cache[keys[i]] = v
        np.savez(cache_path, **{k: cache[k] for k in set(keys)})
    vecs = np.stack([cache[k] for k in keys]).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    print(f"[build] embedded {len(todo)} new / {len(docs)} chunks in {time.time() - t:.0f}s")
    return vecs.astype(np.float16)


def main():
    chunks, projects = [], []
    build_darglobal(chunks, projects)
    listings = build_wasalt(chunks)
    # exact duplicate chunk text (e.g. same blog teaser on many pages) adds nothing
    seen, uniq = set(), []
    for c in chunks:
        key = re.sub(r"\W+", "", c["text"].lower())[:400]
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    chunks = uniq
    for l in listings:
        l.pop("_detail", None)
    write_db(chunks, projects, listings)
    vecs = embed(chunks)
    np.save(OUT / "vectors.npy", vecs)
    kinds = collections.Counter(f"{c['source']}:{c['kind']}" for c in chunks)
    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embed_model": EMBED_MODEL, "n_chunks": len(chunks), "n_projects": len({p['slug'] for p in projects}),
        "n_listings": len(listings), "chunk_kinds": dict(kinds),
        "listing_cities": dict(collections.Counter(l.get("city") for l in listings).most_common(15)),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
