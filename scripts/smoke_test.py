"""End-to-end check against a running backend.

    python scripts/smoke_test.py                       # local  http://localhost:7860
    python scripts/smoke_test.py https://<backend-host>
"""
import json
import sys
import time
import uuid

import httpx

API = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860").rstrip("/")

# (question, expected route(s), words that should appear in the answer)
CASES = [
    ("hi, what can you help me with?", {"chitchat"}, []),
    ("What unit sizes does The Astera offer?", {"docs", "both"}, ["astera"]),
    ("Which DarGlobal projects are with the Trump Organization?", {"docs", "both"}, ["trump"]),
    ("Show me apartments for sale in Riyadh under 1 million SAR with at least 3 bedrooms",
     {"listings", "both"}, ["riyadh"]),
    ("and which is the cheapest one?", {"listings", "both"}, []),          # follow-up
    ("What is Wasalt?", {"docs", "both"}, ["wasalt"]),
    ("When did DarGlobal list on the London Stock Exchange?", {"docs", "both"}, ["london"]),
    ("Write me a poem about cats", {"off_topic", "chitchat"}, []),
]


def main():
    sid = "smoke" + uuid.uuid4().hex[:8]
    h = httpx.get(f"{API}/health", timeout=60).json()
    print("health:", {k: h[k] for k in ("ready", "llm_configured")}, h["data"].get("n_chunks"), "chunks")
    ok = 0
    for q, routes, words in CASES:
        t = time.time()
        r = httpx.post(f"{API}/api/chat/sync", json={"session_id": sid, "message": q}, timeout=180).json()
        ans = r.get("answer", "")
        good = (r.get("route") in routes and not r.get("error")
                and all(w in ans.lower() for w in words))
        ok += good
        print(f"\n{'PASS' if good else 'FAIL'}  [{r.get('route')}] {q}  ({time.time() - t:.1f}s, {r.get('model')})")
        if r.get("error"):
            print("  error:", r["error"], r.get("detail"))
        print("  standalone:", r.get("standalone"))
        print("  sources:", [s["title"][:40] for s in r.get("sources", [])][:5])
        print("  answer:", ans[:400].replace("\n", " "))
    print(f"\n{ok}/{len(CASES)} passed")


if __name__ == "__main__":
    main()
