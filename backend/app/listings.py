"""Structured listing search. The LLM only fills filter fields; the SQL here is fixed."""
from __future__ import annotations

import difflib
import sqlite3

from . import config

SORTS = {"price_asc": "price ASC", "price_desc": "price DESC", "newest": "published_at DESC",
         "area_desc": "area_sqm DESC"}
TYPE_ALIASES = {"flat": "apartment", "apartments": "apartment", "villas": "villa",
                "house": "villa", "townhouse": "townhouse", "land": "land", "plot": "land",
                "lands": "land", "floors": "floor", "duplex": "duplex", "office": "office",
                "shop": "shop", "building": "building", "rest house": "rest house",
                "chalet": "chalet"}


class ListingStore:
    def __init__(self):
        self.con: sqlite3.Connection | None = None

    def load(self):
        self.con = sqlite3.connect(config.DATA_DIR / "kb.db", check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.cities = [r[0] for r in self.con.execute(
            "SELECT DISTINCT city FROM listings WHERE city IS NOT NULL")]
        self.types = [r[0] for r in self.con.execute(
            "SELECT DISTINCT type FROM listings WHERE type IS NOT NULL")]
        self.districts = [r[0] for r in self.con.execute(
            "SELECT DISTINCT district FROM listings WHERE district IS NOT NULL")]

    # ---- normalisation of LLM-provided values to values that exist in the DB
    @staticmethod
    def _match(value, options, cutoff=0.75):
        if not value or not options:
            return None
        v = str(value).strip().lower()
        low = {o.lower(): o for o in options}
        if v in low:
            return low[v]
        for o_l, o in low.items():           # "al khobar" vs "Al Khobar", "riyadh city"
            if v in o_l or o_l in v:
                return o
        m = difflib.get_close_matches(v, list(low), n=1, cutoff=cutoff)
        return low[m[0]] if m else None

    def normalise(self, f: dict) -> dict:
        out = {}
        out["city"] = self._match(f.get("city"), self.cities)
        out["district"] = self._match(f.get("district"), self.districts, 0.8)
        t = (f.get("type") or "").strip().lower()
        t = TYPE_ALIASES.get(t, t)
        out["type"] = self._match(t, self.types, 0.7) if t else None
        p = (f.get("purpose") or "").strip().lower()
        out["purpose"] = p if p in ("sale", "rent", "daily") else None
        for k in ("beds_min", "beds_max", "price_min", "price_max", "area_min"):
            try:
                v = float(f.get(k)) if f.get(k) not in (None, "", "null") else None
            except (TypeError, ValueError):
                v = None
            out[k] = v if v is None or v >= 0 else None
        s = f.get("sort")
        out["sort"] = s if s in SORTS else None
        return out

    def _query(self, f: dict, limit: int):
        where, args = [], []
        for col in ("city", "district", "type", "purpose"):
            if f.get(col):
                where.append(f"{col} = ?")
                args.append(f[col])
        for key, col, op in (("beds_min", "beds", ">="), ("beds_max", "beds", "<="),
                             ("price_min", "price", ">="), ("price_max", "price", "<="),
                             ("area_min", "area_sqm", ">=")):
            if f.get(key) is not None:
                where.append(f"{col} {op} ?")
                args.append(f[key])
        if (f.get("sort") or "").startswith("price"):
            where.append("price IS NOT NULL AND price > 0")
        sql = "SELECT * FROM listings"
        if where:
            sql += " WHERE " + " AND ".join(where)
        order = SORTS.get(f.get("sort") or "", "verified DESC, published_at DESC")
        sql += f" ORDER BY {order} LIMIT ?"
        rows = self.con.execute(sql, (*args, limit)).fetchall()
        count_sql = "SELECT COUNT(*) FROM listings" + (" WHERE " + " AND ".join(where) if where else "")
        total = self.con.execute(count_sql, args).fetchone()[0]
        return [dict(r) for r in rows], total

    def search(self, raw_filters: dict, limit: int = config.LISTING_LIMIT) -> dict:
        raw_filters = raw_filters or {}
        f = self.normalise(raw_filters)
        relaxed = []
        # A city we hold no listings for must not silently become "any city".
        if raw_filters.get("city") and not f["city"]:
            return {"filters": f, "relaxed": [], "total": 0, "rows": [],
                    "missing": {"city": raw_filters["city"], "available_cities": self.cities}}
        if raw_filters.get("type") and not f["type"]:
            relaxed.append(f"type ({raw_filters['type']} not in data)")
        rows, total = self._query(f, limit)
        # loosen step by step if nothing matched
        steps = [("district", None), ("beds_max", None), ("beds_min", None),
                 ("price_max", lambda v: v * 1.2 if v else None), ("area_min", None), ("type", None)]
        for key, fn in steps:
            if rows:
                break
            if f.get(key) is None:
                continue
            f[key] = fn(f[key]) if fn else None
            relaxed.append(key)
            rows, total = self._query(f, limit)
        return {"filters": f, "relaxed": relaxed, "total": total, "rows": rows}


LISTINGS = ListingStore()


def listing_line(i: int, r: dict) -> str:
    where = ", ".join(x for x in (r.get("district"), r.get("city")) if x)
    price = f"{r['currency'] or 'SAR'} {r['price']:,.0f}" if r.get("price") else "price on request"
    if r.get("purpose") == "rent" and r.get("rent_frequency"):
        price += f" / {r['rent_frequency']}"
    spec = ", ".join(x for x in (
        f"{r['beds']:.0f} bed" if r.get("beds") else None,
        f"{r['baths']:.0f} bath" if r.get("baths") else None,
        f"{r['area_sqm']:,.0f} sqm" if r.get("area_sqm") else None) if x)
    return (f"[{i}] {r.get('title')} | {r.get('type') or ''} for {r.get('purpose') or ''} | {where} | "
            f"{price} | {spec} | listing #{r['id']} | {r['url']}")
