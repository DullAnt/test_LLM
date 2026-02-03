"""
Embeddings модель для RAG системы
Поддержка различных моделей через псевдонимы + Singleton pattern
"""
from typing import Optional, List, Union, Literal
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from connections.config import Config  # Используем центральный конфиг

# Типы для type hints
EMBEDDINGS_MODEL_TYPE = Literal[
    "e5-small", "e5-base", "e5-large", "minilm-l12", "mpnet-base"
]
DEVICE_TYPE = Literal["cpu", "cuda", "mps"]

class EmbeddingModel(Embeddings):
    """Embedding модель с поддержкой LangChain интерфейса"""
    
    def __init__(self, embeddings_name: str, device: str):
        self._model_name = self._match_model_path(embeddings_name)
        self._device = device
        
        # Инициализация
        print(f"[EMBEDDINGS] Loading model: {self._model_name} (device: {self._device})...")
        self._model = SentenceTransformer(self._model_name, device=self._device)
        print(f"[EMBEDDINGS] Model loaded successfully.")

    def _match_model_path(self, model_name: str) -> str:
        model_mapping = {
            "e5-small": "intfloat/multilingual-e5-small",
            "e5-base": "intfloat/multilingual-e5-base",
            "e5-large": "intfloat/multilingual-e5-large",
            "e5-large-instruct": "intfloat/multilingual-e5-large-instruct",
            "minilm-l12": "paraphrase-multilingual-MiniLM-L12-v2",
            "mpnet-base": "paraphrase-multilingual-mpnet-base-v2",
        }
        return model_mapping.get(model_name, model_name)

    def encode(self, texts: Union[str, List[str]], show_progress_bar: bool = False):
        return self._model.encode(texts, show_progress_bar=show_progress_bar, convert_to_numpy=True)

    # LangChain Interface
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.encode(text).tolist()

# --- SINGLETON PATTERN IMPLEMENTATION ---

DEFAULT_EMBEDDING_MODEL: Optional[EmbeddingModel] = None

def get_embedding_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None
) -> EmbeddingModel:
    """
    Возвращает синглтон модели.
    Если модель еще не загружена - загружает её, используя параметры из Config.
    """
    global DEFAULT_EMBEDDING_MODEL
    
    # Если модель уже есть - возвращаем её (игнорируя новые параметры, т.к. это синглтон)
    if DEFAULT_EMBEDDING_MODEL is not None:
        return DEFAULT_EMBEDDING_MODEL
    
    # Если создаем впервые:
    target_model = model_name or Config.EMBEDDING_MODEL
    target_device = device or Config.DEVICE  # Берем device из конфига!
    
    DEFAULT_EMBEDDING_MODEL = EmbeddingModel(
        embeddings_name=target_model,
        device=target_device
    )
    
    return DEFAULT_EMBEDDING_MODEL
