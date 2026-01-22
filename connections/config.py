"""
Конфигурация системы
Все параметры загружаются из файла 'configuration'
"""

import os
from dotenv import load_dotenv

# Загружаем из файла 'configuration' вместо '.env'
load_dotenv(dotenv_path="configuration")

# =============================================================================
# Embedding models registry
# =============================================================================

EMBEDDING_MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": {"dims": 384, "size": "420MB", "quality": "good"},
    "paraphrase-multilingual-mpnet-base-v2": {"dims": 768, "size": "1.1GB", "quality": "excellent"},
    "intfloat/multilingual-e5-large-instruct": {"dims": 1024, "size": "2.2GB", "quality": "excellent"},
    "intfloat/multilingual-e5-large": {"dims": 1024, "size": "2.2GB", "quality": "excellent"},
    "intfloat/multilingual-e5-base": {"dims": 768, "size": "1.1GB", "quality": "very-good"},
    "intfloat/multilingual-e5-small": {"dims": 384, "size": "470MB", "quality": "good"},
}

# =============================================================================
# OLLAMA
# =============================================================================

DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))

# =============================================================================
# ELASTICSEARCH
# =============================================================================

DEFAULT_ELASTIC_HOST = os.getenv("ELASTIC_HOST", "localhost")
DEFAULT_ELASTIC_PORT = int(os.getenv("ELASTIC_PORT", "9200"))
DEFAULT_ELASTIC_INDEX = os.getenv("ELASTIC_INDEX", "psb_docs")

ELASTIC_URL = os.getenv("ELASTIC_URL", f"http://{DEFAULT_ELASTIC_HOST}:{DEFAULT_ELASTIC_PORT}")
ELASTIC_USER = os.getenv("ELASTIC_USER")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")

# =============================================================================
# EMBEDDINGS
# =============================================================================

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
DEFAULT_EMBEDDING_DIMS = int(
    os.getenv(
        "EMBEDDING_DIMS",
        str(EMBEDDING_MODELS.get(DEFAULT_EMBEDDING_MODEL, {}).get("dims", 1024)),
    )
)

# =============================================================================
# RAG PARAMETERS
# =============================================================================

DEFAULT_TOP_K = int(os.getenv("TOP_K", "3"))
DEFAULT_NEED_HYDE = os.getenv("NEED_HYDE", "True").lower() == "true"
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

# =============================================================================
# PATHS
# =============================================================================

DEFAULT_DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "data/documents")
DEFAULT_TESTSETS_PATH = os.getenv("TESTSETS_PATH", "data/testsets")
DEFAULT_REPORTS_PATH = os.getenv("REPORTS_PATH", "data/reports")

# =============================================================================
# UNIFIED CONFIG CLASS
# =============================================================================

class Config:
    """Единый класс конфигурации - все параметры в одном месте"""
    
    # Ollama
    OLLAMA_HOST = DEFAULT_OLLAMA_HOST
    OLLAMA_MODEL = DEFAULT_OLLAMA_MODEL
    OLLAMA_TIMEOUT = DEFAULT_OLLAMA_TIMEOUT

    # Embeddings
    EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
    EMBEDDING_DIMS = DEFAULT_EMBEDDING_DIMS

    # Elasticsearch
    ELASTIC_URL = ELASTIC_URL
    ELASTIC_INDEX = DEFAULT_ELASTIC_INDEX
    ELASTIC_USER = ELASTIC_USER
    ELASTIC_PASSWORD = ELASTIC_PASSWORD
    ELASTIC_API_KEY = ELASTIC_API_KEY

    # RAG
    TOP_K = DEFAULT_TOP_K
    NEED_HYDE = DEFAULT_NEED_HYDE
    SIMILARITY_THRESHOLD = DEFAULT_SIMILARITY_THRESHOLD

    # Paths
    DOCUMENTS_PATH = DEFAULT_DOCUMENTS_PATH
    TESTSETS_PATH = DEFAULT_TESTSETS_PATH
    REPORTS_PATH = DEFAULT_REPORTS_PATH


__all__ = [
    "EMBEDDING_MODELS",
    "Config",
    # Экспортируем константы для обратной совместимости
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_OLLAMA_TIMEOUT",
    "DEFAULT_ELASTIC_HOST",
    "DEFAULT_ELASTIC_PORT",
    "DEFAULT_ELASTIC_INDEX",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_DIMS",
    "DEFAULT_TOP_K",
    "DEFAULT_NEED_HYDE",
    "DEFAULT_SIMILARITY_THRESHOLD",
]