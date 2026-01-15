from typing import List, Dict, Optional
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document

from package.config import DEFAULT_TOP_K, Config
from rag.embeddings import EmbeddingModel
from rag.ollama_client import OllamaClient


class DocumentRetriever:
    """Векторный поиск документов (ES через langchain-elasticsearch)"""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        index_name: str = Config.ELASTIC_INDEX,
        top_k: int = DEFAULT_TOP_K,
        ollama_client: Optional[OllamaClient] = None,
        es_url: Optional[str] = Config.ELASTIC_URL,
        es_user: Optional[str] = Config.ELASTIC_USER,
        es_password: Optional[str] = Config.ELASTIC_PASSWORD,
        es_api_key: Optional[str] = Config.ELASTIC_API_KEY,
        # ВАЖНО: поля индекса
        text_field: str = "content",
        vector_field: str = "embedding",
    ):
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.top_k = top_k
        self.ollama_client = ollama_client

        self.store: Optional[ElasticsearchStore] = None
        if es_url:
            self.store = ElasticsearchStore(
                es_url=es_url,
                index_name=index_name,
                embedding=embedding_model,
                es_user=es_user,
                es_password=es_password,
                es_api_key=es_api_key,
                query_field=text_field,
                vector_query_field=vector_field,
            )

    def retrieve_with_scores(
        self,
        question: str,
        return_hyde_info: bool = False,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        k = top_k if top_k is not None else self.top_k

        if return_hyde_info:
            hypothesis = self._generate_hypothesis(question)
            return self.search(hypothesis, top_k=k)

        return self.search(question, top_k=k)

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        k = top_k if top_k is not None else self.top_k

        if self.store is None:
            return self._search_local(query, k)

        return self._search_elasticsearch(query, k)

    @staticmethod
    def _doc_builder(hit: Dict) -> Document:
        src = hit.get("_source", {})
        return Document(
            page_content=src.get("content", ""),
            metadata={
                "filename": src.get("filename", "unknown"),
                "chunk_id": src.get("chunk_id", 0),
                "total_chunks": src.get("total_chunks", 0),
            },
        )

    def _search_elasticsearch(self, query: str, top_k: int) -> List[Dict]:
        # similarity_search_with_score возвращает (Document, score)
        pairs = self.store.similarity_search_with_score(
            query=query,
            k=top_k,
            doc_builder=self._doc_builder,  # для формата индекса
        )

        results: List[Dict] = []
        for rank, (doc, score) in enumerate(pairs, 1):
            results.append(
                {
                    "text": doc.page_content,
                    "score": float(score),
                    "rank": rank,
                    "source": doc.metadata.get("filename", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", 0),
                    "total_chunks": doc.metadata.get("total_chunks", 0),
                }
            )
        return results

    def _generate_hypothesis(self, question: str) -> str:
        if not self.ollama_client:
            return question

        hyde_prompt = (
            "Ты - эксперт банка ПСБ. Ответь кратко и точно на вопрос (2-3 предложения). "
            "НЕ объясняй, просто дай прямой ответ как в официальном документе.\n\n"
            f"Вопрос: {question}\n\nОтвет:"
        )

        try:
            hypothesis = self.ollama_client.generate(question=hyde_prompt, context=[])
            hypothesis = (hypothesis or "").strip()
            return hypothesis[:500]
        except Exception:
            return question

    # локальный поиск как fallback (без ES)
    def _search_local(self, query: str, top_k: int) -> List[Dict]:
        if not hasattr(self, "local_chunks"):
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        query_embedding = self.embedding_model.encode(query)
        results = []

        for chunk in self.local_chunks:
            chunk_embedding = self.embedding_model.encode(chunk["text"])
            score = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
            results.append({"text": chunk["text"], "score": float(score), "source": chunk["source"], "rank": 0})

        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results[:top_k], 1):
            r["rank"] = i
        return results[:top_k]
