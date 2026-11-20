---
title: Property Assistant API
emoji: 🏙️
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: RAG chatbot API over DarGlobal and Wasalt public data
---

# Property Assistant API

FastAPI backend for the DarGlobal × Wasalt property chatbot. It runs as a Docker Space.

- `GET /health`: readiness, model chain, knowledge-base stats
- `POST /api/chat`: server-sent events (`meta`, then `token`*, then `done` or `error`)
- `POST /api/chat/sync`: the same turn as a single JSON response
- `DELETE /api/session/{id}`: clear conversation memory

Required secret: `OPENROUTER_API_KEY`. Optional: `MODEL_CHAIN` (comma-separated free model ids) and `ALLOWED_ORIGINS`.

The full project README is in the GitHub repository.
