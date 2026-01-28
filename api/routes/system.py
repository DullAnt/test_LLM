from fastapi import APIRouter, Request

from api.schemas import HealthResponse, ConfigResponse
from connections.config import Config
from api.deps import check_elasticsearch

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    rag = getattr(request.app.state, "rag", None)
    elastic_ok = check_elasticsearch()
    ollama_ok = bool(rag and rag.ollama and rag.ollama.check_connection())
    return HealthResponse(ok=(elastic_ok and ollama_ok), elastic_ok=elastic_ok, ollama_ok=ollama_ok)


@router.get("/config", response_model=ConfigResponse)
def get_config() -> ConfigResponse:
    return ConfigResponse(
        ollama_host=Config.OLLAMA_HOST,
        ollama_model=Config.OLLAMA_MODEL,
        embedding_model=Config.EMBEDDING_MODEL,
        elastic_url=Config.ELASTIC_URL,
        elastic_index=Config.ELASTIC_INDEX,
        top_k=Config.TOP_K,
        need_hyde=Config.NEED_HYDE,
        similarity_threshold=Config.SIMILARITY_THRESHOLD,
    )


@router.get("/ollama/info")
def ollama_info(request: Request):
    rag = getattr(request.app.state, "rag", None)
    if not rag:
        return {"connected": False, "error": "RAG service not initialized"}
    return rag.ollama.get_info()
