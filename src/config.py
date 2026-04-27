import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Project `.env` wins over any `.env` in the process cwd (Streamlit’s cwd can differ)
load_dotenv(PROJECT_ROOT / ".env", override=True)

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
UPLOADS_DIR = DATA_DIR / "uploads"

# Chat: Groq OpenAI-compatible API only (https://console.groq.com/)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile").strip()

LLM_CONFIGURED = bool(GROQ_API_KEY)

# Optional: output-token logprobs on chat (Groq may reject; RAG code falls back)
CHAT_LOGPROBS = os.getenv("CHAT_LOGPROBS", os.getenv("OPENAI_LOGPROBS", "false")).lower() in (
    "1",
    "true",
    "yes",
)
CHAT_TOP_LOGPROBS = int(os.getenv("CHAT_TOP_LOGPROBS", os.getenv("OPENAI_TOP_LOGPROBS", "0")))

# Local embeddings (proposal: sentence-BERT all-MiniLM-L6-v2)
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking (hybrid: section-aware + sliding window in characters)
CHUNK_TARGET_CHARS = int(os.getenv("CHUNK_TARGET_CHARS", "1000"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "120"))

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "6"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.6"))
MMR_SELECT = int(os.getenv("MMR_SELECT", "6"))
SUMMARY_RETRIEVAL_K = int(os.getenv("SUMMARY_RETRIEVAL_K", "15"))
SUMMARY_MMR_SELECT = int(os.getenv("SUMMARY_MMR_SELECT", "10"))
MIN_RELEVANCE_SIM = float(os.getenv("MIN_RELEVANCE_SIM", "0.08"))
