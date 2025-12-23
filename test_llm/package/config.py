"""
Конфигурация для внешних сервисов
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()


@dataclass
class ElasticsearchConfig:
    """Конфигурация Elasticsearch"""
    host: str = os.getenv("ELASTIC_HOST", "localhost")
    port: int = int(os.getenv("ELASTIC_PORT", "9200"))
    index_name: str = os.getenv("ELASTIC_INDEX", "psb_docs")
    
    def get_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class OllamaConfig:
    """Конфигурация Ollama"""
    host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "gemma2:2b")  # Легкая модель для слабого ПК
    timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "300"))  # 300 секунд для CPU


@dataclass
class EmbeddingConfig:
    """Конфигурация Embeddings"""
    model_name: str = os.getenv(
        "EMBEDDING_MODEL",
        "paraphrase-multilingual-MiniLM-L12-v2"  # L6 для слабого ПК
    )


@dataclass
class EvaluationConfig:
    """Конфигурация оценки"""
    similarity_threshold: float = float(
        os.getenv("SIMILARITY_THRESHOLD", "0.7")
    )
    top_k: int = int(os.getenv("TOP_K", "3"))  # 3 для слабого ПК


# ==================================
# Главный класс Config для main.py
# ==================================
class Config:
    """Unified configuration class для обратной совместимости"""
    
    # Ollama settings
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
    
    # Embeddings settings
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL", 
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Evaluation settings
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    TOP_K = int(os.getenv("TOP_K", "3"))
    
    # Elasticsearch settings
    ELASTIC_HOST = os.getenv("ELASTIC_HOST", "localhost")
    ELASTIC_PORT = int(os.getenv("ELASTIC_PORT", "9200"))
    ELASTIC_INDEX = os.getenv("ELASTIC_INDEX", "psb_docs")
    ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD", "")
    
    # Neo4j settings
    DB_NAME = os.getenv("DB_NAME", "")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    
    # Postgres settings
    POSTGRES_DB_NAME = os.getenv("POSTGRES_DB_NAME", "")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
    
    # Paths
    DOCUMENTS_PATH = "data/documents"
    TESTSETS_PATH = "data/testsets"
    REPORTS_PATH = "data/reports"