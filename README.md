# Property Assistant: DarGlobal × Wasalt RAG chatbot

An AI chatbot that answers questions about **DarGlobal** projects and the company, and finds **Wasalt** property listings. It uses data scraped from the two public websites and free LLMs (gpt-oss on Groq, with OpenRouter free models as fallback). The backend runs in Docker on Render's free tier and the frontend is on Vercel.

| | URL |
|---|---|
| **Chatbot (try it)** | https://REPLACE.vercel.app |
| Backend API (Docker, Render) | https://dg-property-assistant-api.onrender.com ([/health](https://dg-property-assistant-api.onrender.com/health), [/docs](https://dg-property-assistant-api.onrender.com/docs)) |

> The backend is on a free tier that sleeps after 15 idle minutes. The first message after a pause can take up to a minute while it wakes; the status pill shows "Waking server".

---

## What it does

- **Project and company Q&A**: unit sizes, amenities, locations, brand partners (Trump, Aston Martin, Pagani, Missoni…), investment highlights, press releases, the LSE listing. Answers cite numbered sources that link to the original page.
- **Listing search**: "3-bed apartments in Riyadh under 1M SAR", "cheapest rental". Price, bedrooms and city are applied as exact SQL filters, not guessed by similarity search.
- **Multi-turn**: follow-ups like *"and the cheapest one?"* or *"what about villas?"* are rewritten into standalone questions using the chat history.
- **Grounded**: the model answers only from retrieved context and says when something isn't in the data.

## Architecture

```
OFFLINE (runs once, on a laptop)
  darglobal.co.uk ─┐   Playwright    ┌─ extract CMS JSON + page text ─┐   ┌─► SQLite: listings, projects, chunks
  (sitemap, 448 pp) ├─► (polite,  ───┤                                 ├───┤
  wasalt.sa ───────┘   cached raw)   └─ strip personal data, normalise ┘   └─► vectors.npy (bge-small, 384-d)

ONLINE (Docker container, Render free web service)
  Browser (Vercel) ──POST /api/chat──► FastAPI ──► LLM call 1: rewrite + route (JSON)
                   ◄── SSE stream ───            │   route = listings | docs | both | chitchat
                                                 ├─► listings → fixed SQL with validated filters
                                                 ├─► docs → hybrid search (BM25 + dense) × 4 queries
                                                 │          → RRF fusion → cross-encoder rerank
                                                 │          → MMR redundancy filter → top 5
                                                 └─► LLM call 2: grounded answer with [n] citations (streamed)
```

Only **2 LLM calls per question**. Search, fusion, reranking and de-duplication all run locally on the CPU inside the container, so they cost no free-tier requests and add about 1–2 s.

### Retrieval pipeline
| Stage | What | Why |
|---|---|---|
| Query rewrite | Call 1 returns a standalone question plus 3 search variants | Follow-ups work; better recall |
| Hybrid search | BM25 (exact names like "DG1", IDs, numbers) + bge-small-en-v1.5 vectors, top 10 each per query | Each catches what the other misses |
| RRF fusion | Σ 1/(60 + rank) over all lists, keep 20 | No score calibration needed between retrievers |
| Rerank | ms-marco-MiniLM-L-6 cross-encoder (ONNX), keep 8 | Biggest accuracy gain per millisecond; L-6 gave the same top results as L-12 on our checks at half the time and RAM |
| MMR | λ = 0.7, drop cosine > 0.92, max 3 chunks per page, keep 5 | Stops near-duplicate passages filling the context |

### Router
Call 1 also classifies the question and extracts filters `{city, district, type, purpose, beds_min/max, price_min/max, area_min, sort}`. The model never writes SQL. Values are matched against what exists in the DB, then placed into a fixed parameterised query. If nothing matches, filters are relaxed step by step, and the answer says which ones were relaxed. A city with no data returns "no listings for X" rather than silently searching everywhere.

### Models and free-tier resilience
Two free, OpenAI-compatible providers are configured, and each call walks a fallback chain:

| Call | Chain (in order) |
|---|---|
| 1. Router (rewrite + route JSON) | Groq `openai/gpt-oss-20b` → Groq `openai/gpt-oss-120b` → OpenRouter `:free` models |
| 2. Answer (streamed) | Groq `openai/gpt-oss-120b` → Groq `openai/gpt-oss-20b` → OpenRouter `:free` models (Nemotron, Gemma, Ling…) |

On a 429, 404, 5xx, timeout or unparseable router JSON, the request moves to the next model. The failed model cools down for 60 s, or for 30 min if the provider reports a daily limit. Providers without an API key are skipped, so either key alone is enough. If the configured OpenRouter models disappear, the backend re-reads OpenRouter's catalogue and adds whatever free models exist. Chains can be overridden with the `ROUTER_CHAIN` and `MODEL_CHAIN` env vars (`provider:model,...`).

## Data collected

| Source | What | Volume |
|---|---|---|
| darglobal.co.uk | Every page in the sitemap except utility and landing pages: projects (structured CMS fields: type, status, units, size, amenities, why-invest, FAQs), partners, one-of-one collection, press releases, insights, blog | 448 pages, 27 projects with structured facts, 139 project FAQs, 73 press releases, 201 blog posts |
| wasalt.sa | City result pages and listing pages (price, city, district, type, beds, baths, area, description) | 80 listings (Riyadh, sale and rent) + 8 market FAQs |
| wasalt.sa sitemaps | "About Wasalt" plus an inventory overview computed from 60,000 public listing URLs (type mix, bedroom distribution) | 3 passages |

Total: **3,188 indexed passages**.

**Scraping approach.** A standard headless Chromium with no stealth tricks. Images, fonts and media are not downloaded. Requests are spaced 1.5–20 s apart, results are cached so nothing is fetched twice, and `robots.txt` is respected (Wasalt's `/search` is never touched). DarGlobal pages ship their CMS content as JSON, so project facts are parsed from structured data rather than scraped HTML. Agent, owner and phone fields in Wasalt listings are dropped before anything is saved.

**Limitation.** wasalt.sa sits behind a Cloudflare bot check. When it challenged the scraper, the scraper backed off and stopped instead of trying to get around it. The Wasalt listing sample is therefore smaller than DarGlobal's coverage, and the bot says so when asked about a city outside the sample.

## Project layout

```
scraper/      browser.py (polite crawler, cache, back-off), darglobal.py, wasalt.py
pipeline/     build.py: raw JSONL → SQLite + vectors + manifest
backend/      Dockerfile, FastAPI app (router, retrieval, listings, memory, llm), data/  ← Docker build root
frontend/     static chat UI (streaming, listing cards, sources, "how I answered" trace) ← Vercel root
scripts/      smoke_test.py: end-to-end checks against a running backend
```

## Run locally

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; use bin/activate on mac/linux
pip install -r backend/requirements.txt playwright beautifulsoup4 lxml httpx
python -m playwright install chromium

python -m scraper.darglobal        # ~15 min
python -m scraper.wasalt           # stops politely if blocked
python pipeline/build.py           # builds backend/data/

cd backend && GROQ_API_KEY=gsk_... OPENROUTER_API_KEY=sk-or-... uvicorn app.main:app --port 7860
# frontend: open frontend/index.html?api=http://localhost:7860  (or: python -m http.server -d frontend)
python scripts/smoke_test.py
```

With Docker:
```bash
docker build -t property-assistant backend
docker run -p 7860:7860 -e GROQ_API_KEY=gsk_... -e OPENROUTER_API_KEY=sk-or-... property-assistant
```

## Deployment
- **Backend**: `render.yaml` defines a free Docker web service built from `backend/`. Models are baked into the image at build time, so cold starts only load files. `OPENROUTER_API_KEY` is a secret set in the Render dashboard. The whole process peaks at about 430 MB, inside the 512 MB free limit (one ONNX thread, reranking in batches of 4). The same image also runs unchanged on a Hugging Face Docker Space (`scripts/deploy_hf.py`) or any Docker host; it listens on `$PORT`, defaulting to 7860.
- **Frontend**: the `frontend/` folder is a static site on Vercel. `config.js` points at the Space URL and the browser calls the API directly (CORS enabled).

## Next steps (left out on purpose to keep latency and free-tier usage low)
- LLM-as-judge evaluator that retries weak answers with a critique
- Live web-search tool for questions outside the scraped data
- Rolling conversation summary for very long chats (currently the last 8 messages)
- Contextual compression of retrieved passages
- Scheduled re-scrape plus incremental index updates
