from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict

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
        use_top_k = top_k if top_k is not None else Config.TOP_K
        use_hyde = hyde if hyde is not None else Config.NEED_HYDE

        chunks = self.retriever.retrieve_with_scores(
            question=query,
            return_hyde_info=use_hyde,
            top_k=use_top_k,
        )

        context = [c.get("text", "") for c in chunks]
        answer = self.ollama.generate(question=query, context=context)

        return {"answer": answer, "chunks": chunks}


def build_rag_service(
    model_override: Optional[str] = None,
    embeddings_override: Optional[str] = None,
) -> RAGService:
    # Embeddings
    embedding_name = embeddings_override or Config.EMBEDDING_MODEL
    embedding_model = get_embedding_model(embedding_name)  # device пока default cpu

    # Ollama
    ollama_model = model_override or Config.OLLAMA_MODEL
    ollama = OllamaClient(
        host=Config.OLLAMA_HOST,
        model=ollama_model,
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
    try:
        # используем твой единый клиент
        es_client = ElasticsearchClient(url=Config.ELASTIC_URL, index_name=Config.ELASTIC_INDEX)
        if not es_client.ping():
            return False
        if not es_client.index_exists():
            return False
        return es_client.get_document_count() > 0
    except Exception:
        return False
