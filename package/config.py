"""
Конфигурация для внешних сервисов
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import Optional, Dict
import requests

# Загрузка переменных окружения
load_dotenv()

EMBEDDING_MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "dims": 384,
        "size": "420MB",
        "quality": "good",
        "description": "Стандартная, быстрая, поддержка русского"
    },
    "intfloat/multilingual-e5-large-instruct": {
        "dims": 1024,
        "size": "2.2GB",
        "quality": "excellent",
        "description": "Очень высокое качество для русского"
    }
}


# ==========================================
# АВТООПРЕДЕЛЕНИЕ РАЗМЕРНОСТИ МОДЕЛИ
# ==========================================


def get_model_dims_from_hf(model_name: str) -> Optional[int]:
    """Получить размерность модели через HuggingFace API"""
    try:
        url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
        print(f"🔍 Запрос к HuggingFace API: {model_name}...")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            config = response.json()
            
            possible_keys = [
                'hidden_size',
                'embedding_size',
                'd_model',
                'dim',
                'n_embd',
            ]
            
            for key in possible_keys:
                if key in config:
                    dims = config[key]
                    print(f"Размерность найдена: {dims} (поле: {key})")
                    return dims
            
            print(f" Размерность не найдена в config.json")
            return None
        else:
            print(f"  Не удалось получить config: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f" Ошибка при запросе к HuggingFace: {e}")
        return None


def get_model_dims_from_model(model_name: str) -> Optional[int]:
    """Определить размерность загрузив модель"""
    try:
        print(f"Загрузка модели для определения размерности...")
        model = SentenceTransformer(model_name)
        test_vec = model.encode("test", show_progress_bar=False)
        dims = len(test_vec)
        print(f"Размерность определена: {dims}")
        return dims
    except Exception as e:
        print(f"Не удалось загрузить модель: {e}")
        return None


def get_model_info(model_name: str) -> Dict:
    """Получить информацию о модели (dims + метаданные)"""
    
    # 1. Проверка в словаре
    if model_name in EMBEDDING_MODELS:
        print(f"Модель найдена в конфигурации")
        return EMBEDDING_MODELS[model_name]
    
    print(f"\n{'='*70}")
    print(f"Модель '{model_name}' не найдена в EMBEDDING_MODELS")
    print(f"Автоматическое определение параметров...")
    print(f"{'='*70}\n")
    
    # 2. Попытка через API
    dims = get_model_dims_from_hf(model_name)
    
    # 3. Если не получилось - загрузка модели
    if dims is None:
        print(f"\n\API не помог, загружаем модель...")
        dims = get_model_dims_from_model(model_name)
    
    # 4. Дефолт
    if dims is None:
        print(f"\n  Не удалось определить автоматически!")
        print(f"  Используется dims=384 по умолчанию")
        dims = 384
    
    model_info = {
        "dims": dims,
        "size": "unknown",
        "quality": "unknown",
        "description": f"Автоматически определено (dims={dims})"
    }
    
    EMBEDDING_MODELS[model_name] = model_info
    
    print(f"\n{'='*70}")
    print(f" Модель добавлена в кеш: {model_name}")
    print(f"   Размерность: {dims}")
    print(f"{'='*70}\n")
    
    return model_info

# КОНСТАНТЫ (ЕДИНСТВЕННОЕ МЕСТО ОПРЕДЕЛЕНИЯ)

# Embedding модель 
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    # "google/embeddinggemma-300m"
    "intfloat/multilingual-e5-large"
)


HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# Автоматическое определение размерности
print(f"\n{'='*70}")
print(f"ИНИЦИАЛИЗАЦИЯ EMBEDDING МОДЕЛИ")
print(f"{'='*70}")

model_info = get_model_info(DEFAULT_EMBEDDING_MODEL)
DEFAULT_EMBEDDING_DIMS = model_info["dims"]

print(f"Модель: {DEFAULT_EMBEDDING_MODEL}")
print(f"Размерность: {DEFAULT_EMBEDDING_DIMS}")
if model_info["size"] != "unknown":
    print(f"Размер: {model_info['size']}")
if model_info["quality"] != "unknown":
    print(f"Качество: {model_info['quality']}")
print(f"{'='*70}\n")

# Ollama модель 
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))

# Elasticsearch
DEFAULT_ELASTIC_HOST = os.getenv("ELASTIC_HOST", "localhost")
DEFAULT_ELASTIC_PORT = int(os.getenv("ELASTIC_PORT", "9200"))
DEFAULT_ELASTIC_INDEX = os.getenv("ELASTIC_INDEX", "psb_docs")

# Evaluation
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.60"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))



# EMBEDDING MODEL (SINGLETON)
class EmbeddingModel:
    """
    Singleton класс для работы с embeddings
    Используется во всем проекте
    """
    
    _instance: Optional['EmbeddingModel'] = None
    _model: Optional[SentenceTransformer] = None
    
    def __new__(cls):
        """Singleton pattern - один экземпляр на весь проект"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Инициализация модели (только один раз)"""
        if self._model is None:
            print(f"Загрузка embedding модели: {DEFAULT_EMBEDDING_MODEL}")
            self._model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL) 
            print("Модель загружена")
    
    def encode(self, texts, show_progress_bar: bool = False, convert_to_numpy: bool = True):
        """
        Создание embeddings для текстов
        
        Args:
            texts: Строка или список строк
            show_progress_bar: Показывать прогресс бар
            convert_to_numpy: Конвертировать в numpy array
            
        Returns:
            Numpy array с embeddings или list
        """
        if self._model is None:
            raise RuntimeError("Модель не инициализирована")
        
        embeddings = self._model.encode(
            texts,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy
        )
        
        return embeddings
    
    def get_model(self) -> SentenceTransformer:
        """Получить базовую модель SentenceTransformer"""
        if self._model is None:
            raise RuntimeError("Модель не инициализирована")
        return self._model

# HELPER FUNCTION (для простого доступа)

def get_embedding_model() -> EmbeddingModel:
    """
    Получить глобальный экземпляр EmbeddingModel
    
    Usage:
        from package.config import get_embedding_model
        
        model = get_embedding_model()
        embedding = model.encode("текст")
    """
    return EmbeddingModel()


# DATACLASS КОНФИГУРАЦИИ (для обратной совместимости)

@dataclass
class ElasticsearchConfig:
    """Конфигурация Elasticsearch"""
    host: str = DEFAULT_ELASTIC_HOST          
    port: int = DEFAULT_ELASTIC_PORT          
    index_name: str = DEFAULT_ELASTIC_INDEX
    
    def get_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class OllamaConfig:
    """Конфигурация Ollama"""
    host: str = DEFAULT_OLLAMA_HOST      
    model: str = DEFAULT_OLLAMA_MODEL    
    timeout: int = DEFAULT_OLLAMA_TIMEOUT


@dataclass
class EmbeddingConfig:
    """Конфигурация Embeddings (устарело, используйте get_embedding_model())"""
    model_name: str = DEFAULT_EMBEDDING_MODEL 


@dataclass
class EvaluationConfig:
    """Конфигурация оценки"""
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD 
    top_k: int = DEFAULT_TOP_K                                 


# UNIFIED CONFIG (для обратной совместимости)

class Config:
    """Unified configuration class"""
    
    # Ollama settings
    OLLAMA_HOST = DEFAULT_OLLAMA_HOST      
    OLLAMA_MODEL = DEFAULT_OLLAMA_MODEL    
    OLLAMA_TIMEOUT = DEFAULT_OLLAMA_TIMEOUT
    
    # Embeddings settings
    EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL 
    
    # Evaluation settings
    SIMILARITY_THRESHOLD = DEFAULT_SIMILARITY_THRESHOLD 
    TOP_K = DEFAULT_TOP_K                               
    
    # Elasticsearch settings
    ELASTIC_HOST = DEFAULT_ELASTIC_HOST   
    ELASTIC_PORT = DEFAULT_ELASTIC_PORT   
    ELASTIC_INDEX = DEFAULT_ELASTIC_INDEX 
    
    # Paths
    DOCUMENTS_PATH = "data/documents"
    TESTSETS_PATH = "data/testsets"
    REPORTS_PATH = "data/reports"


    __all__ = [
    # Константы
    'DEFAULT_EMBEDDING_MODEL',
    'DEFAULT_EMBEDDING_DIMS',
    'EMBEDDING_MODELS',      
    'DEFAULT_OLLAMA_MODEL',
    'DEFAULT_OLLAMA_HOST',
    'DEFAULT_OLLAMA_TIMEOUT',
    'DEFAULT_ELASTIC_HOST',
    'DEFAULT_ELASTIC_PORT',
    'DEFAULT_ELASTIC_INDEX',
    'DEFAULT_SIMILARITY_THRESHOLD',
    'DEFAULT_TOP_K',
    # Функции
    'get_embedding_model',
    'get_model_info',       
    # Классы
    'EmbeddingModel',
    'Config',
    'ElasticsearchConfig',
    'OllamaConfig',
    'EmbeddingConfig',
    'EvaluationConfig',
]