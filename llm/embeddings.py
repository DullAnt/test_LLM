"""
Embeddings модель для RAG системы
Поддержка различных моделей через псевдонимы
"""

from typing import Optional, List, Union, Literal
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from connections.config import DEFAULT_EMBEDDING_MODEL


# Типы для type hints
EMBEDDINGS_MODEL_TYPE = Literal[
    "e5-small",
    "e5-base", 
    "e5-large",
    "minilm-l12",
    "mpnet-base",
]

DEVICE_TYPE = Literal["cpu", "cuda", "mps"]


class EmbeddingModel(Embeddings):
    """
    Embedding модель с поддержкой различных моделей и устройств
    """

    def __init__(
        self, 
        embeddings_name: Optional[EMBEDDINGS_MODEL_TYPE | str] = None,
        device: DEVICE_TYPE = "cpu"
    ):
        """
        Args:
            embeddings_name: Название модели (короткий псевдоним или полный путь)
            device: Устройство для вычислений
        """
        
        # Определяем модель
        model_name = embeddings_name or DEFAULT_EMBEDDING_MODEL
        
        # Инициализируем через вспомогательный метод
        self._model = self._init_embeddings(model_name, device)
        self._model_name = model_name
        self._device = device
        
        print(f"[EMBEDDINGS] Модель загружена: {self._model_name} (device: {device})")

    def _match_model_path(self, model_name: str) -> str:
        """
        Преобразование короткого имени в полный путь модели
        
        Args:
            model_name: Короткое имя или полный путь
        
        Returns:
            Полный путь к модели
        """
        # Маппинг коротких имен на полные пути
        model_mapping = {
            # E5 модели (рекомендуемые)
            "e5-small": "intfloat/multilingual-e5-small",
            "e5-base": "intfloat/multilingual-e5-base",
            "e5-large": "intfloat/multilingual-e5-large",
            "e5-large-instruct": "intfloat/multilingual-e5-large-instruct",
            
            # Paraphrase модели
            "minilm-l12": "paraphrase-multilingual-MiniLM-L12-v2",
            "mpnet-base": "paraphrase-multilingual-mpnet-base-v2",
            
            # Можно добавить свои
            "FRIDA": "ai-forever/FRIDA",
        }
        
        # Если есть в маппинге - вернуть полный путь
        # Иначе считаем что уже передан полный путь
        return model_mapping.get(model_name, model_name)

    def _init_embeddings(
        self, 
        embeddings_name: str, 
        device: DEVICE_TYPE
    ) -> SentenceTransformer:
        """
        Инициализация embedding модели
        
        Args:
            embeddings_name: Название модели
            device: Устройство
        
        Returns:
            SentenceTransformer instance
        """
        # Получаем полный путь модели
        model_path = self._match_model_path(embeddings_name)
        
        # Создаем SentenceTransformer с device
        embeddings = SentenceTransformer(
            model_path,
            device=device
        )
        
        return embeddings

    def encode(self, texts: Union[str, List[str]], show_progress_bar: bool = False):
        """Прямое кодирование через SentenceTransformer"""
        if self._model is None:
            raise RuntimeError("Embedding model is not initialized")
        return self._model.encode(
            texts, 
            show_progress_bar=show_progress_bar, 
            convert_to_numpy=True
        )

    # LangChain interface
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Кодирование списка документов (LangChain interface)"""
        vectors = self.encode(texts)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Кодирование одного запроса (LangChain interface)"""
        vector = self.encode(text)
        return vector.tolist()

    def get_model_name(self) -> str:
        """Получить название модели"""
        return self._model_name

    def get_device(self) -> str:
        """Получить устройство"""
        return self._device


def get_embedding_model(
    model_name: Optional[EMBEDDINGS_MODEL_TYPE | str] = None,
    device: DEVICE_TYPE = "cpu"
) -> EmbeddingModel:
    """
    Получить embedding модель
    
    Args:
        model_name: Название модели (короткое имя или полный путь)
        device: Устройство для вычислений
    
    Returns:
        EmbeddingModel instance
    
    """
    return EmbeddingModel(model_name, device)