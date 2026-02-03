from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict

# Импорты для FastAPI dependency
from fastapi import Request, HTTPException

from connections.config import Config
from llm.embeddings import get_embedding_model, EmbeddingModel
from llm.ollama_client import OllamaClient
from llm.retriever import DocumentRetriever
from connections.elastic import ElasticsearchClient

@dataclass
class RAGService:
    embedding_model: EmbeddingModel
    ollama: OllamaClient
    retriever: DocumentRetriever

    def ask(
        self,
        query: str,
        top_k: Optional[int] = None,
        hyde: Optional[bool] = None,
    ) -> Dict:
        """
        Основной метод RAG: поиск + генерация
        """
        # Если параметры не переданы в запросе, берем из Config
        use_top_k = top_k if top_k is not None else Config.TOP_K
        use_hyde = hyde if hyde is not None else Config.NEED_HYDE
        
        # --- БЕЗОПАСНЫЙ ВЫЗОВ ПОИСКА ---
        retrieval_result = self.retriever.retrieve_with_scores(
            question=query,
            return_hyde_info=use_hyde,
            top_k=use_top_k,
        )
        
        chunks = []
        hyde_doc = None

        # Разбираем результат (кортеж или список)
        if isinstance(retrieval_result, tuple):
            chunks = retrieval_result[0]
            # Если включен HyDE, пытаемся найти инфо о нем
            if use_hyde:
                hyde_info = retrieval_result[-1]
                if isinstance(hyde_info, dict):
                    hyde_doc = hyde_info.get("hypothetical_document")
        else:
            # Если вернулся просто список чанков
            chunks = retrieval_result

        # Генерация ответа
        context = [c.get("text", "") for c in chunks]
        answer = self.ollama.generate(question=query, context=context)
        
        return {
            "answer": answer, 
            "chunks": chunks,
            "hyde_doc": hyde_doc
        }

# --- УБРАЛИ ЛИШНИЕ АРГУМЕНТЫ ---
def build_rag_service() -> RAGService:
    """
    Создание и настройка RAG сервиса.
    Конфигурация берется СТРОГО из connections.config.Config
    """
    
    print(f"[RAG BUILD] Init with model: {Config.OLLAMA_MODEL}, embeds: {Config.EMBEDDING_MODEL}")

    # Embeddings
    embedding_model = get_embedding_model(Config.EMBEDDING_MODEL)
    
    # Ollama
    ollama = OllamaClient(
        host=Config.OLLAMA_HOST,
        model=Config.OLLAMA_MODEL,
        timeout=Config.OLLAMA_TIMEOUT,
    )
    
    # Retriever
    retriever = DocumentRetriever(
        embedding_model=embedding_model,
        index_name=Config.ELASTIC_INDEX,
        top_k=Config.TOP_K,
        ollama_client=ollama,
        es_url=Config.ELASTIC_URL,
        es_user=Config.ELASTIC_USER,
        es_password=Config.ELASTIC_PASSWORD,
        es_api_key=Config.ELASTIC_API_KEY,
        text_field="content",
        vector_field="embedding",
    )
    
    return RAGService(
        embedding_model=embedding_model,
        ollama=ollama,
        retriever=retriever,
    )

def check_elasticsearch() -> bool:
    """Проверка доступности ES"""
    try:
        es_client = ElasticsearchClient(url=Config.ELASTIC_URL, index_name=Config.ELASTIC_INDEX)
        if not es_client.ping():
            return False
        if not es_client.index_exists():
            return False
        return es_client.get_document_count() > 0
    except Exception:
        return False

# --- Dependency для роутов ---
def get_rag_service(request: Request) -> RAGService:
    """Безопасное получение сервиса из app.state"""
    rag_service = getattr(request.app.state, "rag", None)
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service is not initialized")
    return rag_service
