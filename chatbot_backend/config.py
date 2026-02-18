from pathlib import Path

# OpenAI
AGENT_MODELS = {
    "information_gatherer": "gpt-5-mini",
    "rag_retriever": "gpt-5-nano",
    "quality_checker": "gpt-5-mini",
    "answer_formulator": "gpt-5-mini",
    "feedback_collector": "gpt-5-nano",
    "clarification_assessor": "gpt-5-mini",
    "pdf_fallback": "gpt-5-mini",
}

# Retry limits
MAX_GATHERER_ATTEMPTS = 3
MAX_RETRIEVAL_ATTEMPTS = 3
PDF_TOC_PAGE_THRESHOLD = 10

# Qdrant RAG
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "chunks"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
RAG_TOP_K = 3

# Docker / Qdrant container
QDRANT_CONTAINER_NAME = "dell-assistant-qdrant"
QDRANT_IMAGE = "qdrant/qdrant:latest"
QDRANT_HOST_PORT = 6333
QDRANT_GRPC_PORT = 6334
QDRANT_STORAGE_PATH = Path(__file__).resolve().parent.parent / "qdrant_storage"
QDRANT_STARTUP_TIMEOUT = 30  # seconds

# File paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
SUPPORT_LOG_CSV = LOGS_DIR / "support_log.csv"
TICKETS_CSV = LOGS_DIR / "tickets.csv"
PDF_DIR = Path(r"C:\Users\busin\Documents\dell_pdfs")
