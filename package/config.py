"""
Конфигурация для внешних сервисов (без сетевых вызовов на import)
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# =========================
# Embedding models registry
# =========================

EMBEDDING_MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": {"dims": 384, "size": "420MB", "quality": "good"},
    "paraphrase-multilingual-mpnet-base-v2": {"dims": 768, "size": "1.1GB", "quality": "excellent"},
    "intfloat/multilingual-e5-large-instruct": {"dims": 1024, "size": "2.2GB", "quality": "excellent"},
    "intfloat/multilingual-e5-large": {"dims": 1024, "size": "2.2GB", "quality": "excellent"},
    "intfloat/multilingual-e5-base": {"dims": 768, "size": "1.1GB", "quality": "very-good"},
    "intfloat/multilingual-e5-small": {"dims": 384, "size": "470MB", "quality": "good"},
}

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
DEFAULT_EMBEDDING_DIMS = int(
    os.getenv(
        "EMBEDDING_DIMS",
        EMBEDDING_MODELS.get(DEFAULT_EMBEDDING_MODEL, {}).get("dims", 1024),
    )
)

# Ollama
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))

# Elasticsearch
DEFAULT_ELASTIC_HOST = os.getenv("ELASTIC_HOST", "localhost")
DEFAULT_ELASTIC_PORT = int(os.getenv("ELASTIC_PORT", "9200"))
DEFAULT_ELASTIC_INDEX = os.getenv("ELASTIC_INDEX", "psb_docs")

ELASTIC_URL = os.getenv("ELASTIC_URL", f"http://{DEFAULT_ELASTIC_HOST}:{DEFAULT_ELASTIC_PORT}")
ELASTIC_USER = os.getenv("ELASTIC_USER")         
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD") 
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")   

# Evaluation
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "3"))

# Paths
DEFAULT_DOCUMENTS_PATH = "data/documents"
DEFAULT_TESTSETS_PATH = "data/testsets"
DEFAULT_REPORTS_PATH = "data/reports"

# Unified Config
class Config:
    OLLAMA_HOST = DEFAULT_OLLAMA_HOST
    OLLAMA_MODEL = DEFAULT_OLLAMA_MODEL
    OLLAMA_TIMEOUT = DEFAULT_OLLAMA_TIMEOUT

    EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
    EMBEDDING_DIMS = DEFAULT_EMBEDDING_DIMS

    ELASTIC_URL = ELASTIC_URL
    ELASTIC_INDEX = DEFAULT_ELASTIC_INDEX
    ELASTIC_USER = ELASTIC_USER
    ELASTIC_PASSWORD = ELASTIC_PASSWORD
    ELASTIC_API_KEY = ELASTIC_API_KEY

    SIMILARITY_THRESHOLD = DEFAULT_SIMILARITY_THRESHOLD
    TOP_K = DEFAULT_TOP_K

    DOCUMENTS_PATH = DEFAULT_DOCUMENTS_PATH
    TESTSETS_PATH = DEFAULT_TESTSETS_PATH
    REPORTS_PATH = DEFAULT_REPORTS_PATH


__all__ = [
    "EMBEDDING_MODELS",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_DIMS",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_OLLAMA_TIMEOUT",
    "DEFAULT_ELASTIC_HOST",
    "DEFAULT_ELASTIC_PORT",
    "DEFAULT_ELASTIC_INDEX",
    "ELASTIC_URL",
    "ELASTIC_USER",
    "ELASTIC_PASSWORD",
    "ELASTIC_API_KEY",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_TOP_K",
    "DEFAULT_DOCUMENTS_PATH",
    "DEFAULT_TESTSETS_PATH",
    "DEFAULT_REPORTS_PATH",
    "Config",
]
