import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Local dev convenience: read KEY=VALUE lines from the repo-root .env (never committed).
for _env in (BASE_DIR / ".env", BASE_DIR.parent / ".env"):
    if _env.exists():
        for _line in _env.read_text(encoding="utf-8").splitlines():
            if "=" in _line and not _line.lstrip().startswith("#"):
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
SESSION_DB = Path(os.getenv("SESSION_DB", "/tmp/sessions.db" if os.name != "nt" else BASE_DIR / "sessions.db"))

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
ONNX_THREADS = int(os.getenv("ONNX_THREADS", "1"))   # free hosts give ~0.1-1 vCPU; 1 thread also saves RAM

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Tried in order; a 429 / 404 / 5xx / timeout moves to the next one.
MODEL_CHAIN = [m.strip() for m in os.getenv(
    "MODEL_CHAIN",
    "nvidia/nemotron-3.5-super-vl:free,google/gemma-4-31b-it:free,"
    "inclusionai/ling-3.2-flash:free,nvidia/nemotron-3.5-lightning:free,"
    "google/gemma-4-26b-a4b-it:free,baidu/ernie-5.5-flash-preview:free",
).split(",") if m.strip()]
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "45"))

# Comma-separated list of allowed browser origins, or * for any.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

HISTORY_MESSAGES = 8          # last N messages sent to the LLM
RETRIEVE_PER_QUERY = 10       # dense and sparse each
FUSED_KEEP = 20               # after RRF
RERANK_KEEP = 8
FINAL_CHUNKS = 5              # after MMR
MMR_LAMBDA = 0.7
MAX_PER_URL = 3
LISTING_LIMIT = 6
