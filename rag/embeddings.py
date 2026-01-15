from typing import Optional, List, Union
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


class EmbeddingModel(Embeddings):
    """
    Singleton embeddings provider (SentenceTransformer) + LangChain Embeddings interface
    """

    _instance: Optional["EmbeddingModel"] = None
    _model: Optional[SentenceTransformer] = None
    _model_name: Optional[str] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: Optional[str] = None):
        if self._model is None:
            from package.config import DEFAULT_EMBEDDING_MODEL
            self._model_name = model_name or DEFAULT_EMBEDDING_MODEL
            self._model = SentenceTransformer(self._model_name)

    def encode(self, texts: Union[str, List[str]], show_progress_bar: bool = False):
        if self._model is None:
            raise RuntimeError("Embedding model is not initialized")
        return self._model.encode(texts, show_progress_bar=show_progress_bar, convert_to_numpy=True)

    # LangChain interface
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.encode(texts)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.encode(text)
        return vector.tolist()

    def get_model_name(self) -> str:
        return self._model_name or "unknown"


def get_embedding_model(model_name: Optional[str] = None) -> EmbeddingModel:
    return EmbeddingModel(model_name)
