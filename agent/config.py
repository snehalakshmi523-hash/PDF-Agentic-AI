"""Central configuration, read from environment variables / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Chat model: "ollama" (free, local, default) | "groq" (free API tier) | "openai" | "anthropic"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
_DEFAULT_MODELS = {"ollama": "llama3.1:8b", "groq": "llama-3.3-70b-versatile",
                   "openai": "gpt-4o-mini", "anthropic": "claude-sonnet-4-5"}
LLM_MODEL = os.getenv("LLM_MODEL") or _DEFAULT_MODELS.get(LLM_PROVIDER, "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Embeddings: "fastembed" (free, local, default) | "openai"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "fastembed").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or (
    "BAAI/bge-small-en-v1.5" if EMBEDDING_PROVIDER == "fastembed" else "text-embedding-3-small")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
MAX_PDF_MB = 50


class ConfigError(RuntimeError):
    """Raised when required configuration (e.g. an API key) is missing."""


def require_keys() -> None:
    needs = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "groq": "GROQ_API_KEY"}
    if LLM_PROVIDER in needs and not os.getenv(needs[LLM_PROVIDER]):
        raise ConfigError(f"LLM_PROVIDER={LLM_PROVIDER} but {needs[LLM_PROVIDER]} is not set in .env.")
    if EMBEDDING_PROVIDER == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise ConfigError("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set in .env.")
