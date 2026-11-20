"""Create/update the Hugging Face Docker Space from ./backend.

    hf auth login                      (once, with a WRITE token)
    python scripts/deploy_hf.py        (reads OPENROUTER_API_KEY from env or .env to set the Space secret)
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
SPACE_NAME = os.getenv("HF_SPACE_NAME", "property-assistant-api")


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def main():
    load_env()
    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{SPACE_NAME}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        api.add_space_secret(repo_id, "OPENROUTER_API_KEY", key)
        print("secret OPENROUTER_API_KEY set")
    else:
        print("WARNING: OPENROUTER_API_KEY not found - set it in the Space settings")
    if os.getenv("MODEL_CHAIN"):
        api.add_space_variable(repo_id, "MODEL_CHAIN", os.environ["MODEL_CHAIN"])
    api.upload_folder(
        repo_id=repo_id, repo_type="space", folder_path=str(ROOT / "backend"),
        ignore_patterns=["models/*", "models/**", "sessions.db", "__pycache__/*", "**/__pycache__/*", ".env"],
        commit_message="Deploy Property Assistant API")
    host = f"https://{user.lower()}-{SPACE_NAME}.hf.space"
    print(f"Space: https://huggingface.co/spaces/{repo_id}")
    print(f"API:   {host}  (first build takes ~5-8 min)")
    return host


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
