from pydantic import BaseModel, Field
from typing import Optional, List


class RagAskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Вопрос пользователя")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Сколько чанков извлекать")
    hyde: Optional[bool] = Field(None, description="Использовать HyDE")
    model: Optional[str] = Field(None, description="Ollama model")
    embeddings: Optional[str] = Field(None, description="Embedding model name")


class RagDocument(BaseModel):
    text: str
    source: str
    score: Optional[float] = None
    rank: Optional[int] = None
    chunk_id: Optional[int] = None
    total_chunks: Optional[int] = None


class RagAskResponse(BaseModel):
    answer: str
    docs: List[RagDocument]

class HealthResponse(BaseModel):
    ok: bool
    elastic_ok: bool
    ollama_ok: bool


class ConfigResponse(BaseModel):
    ollama_host: str
    ollama_model: str
    embedding_model: str
    elastic_url: str
    elastic_index: str
    top_k: int
    need_hyde: bool
    similarity_threshold: float
