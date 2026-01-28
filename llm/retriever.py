from typing import List, Dict, Optional
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document

from connections.config import DEFAULT_TOP_K, Config
from connections.elastic import ElasticsearchClient
from llm.embeddings import EmbeddingModel
from llm.ollama_client import OllamaClient


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
        text_field: str = "content",
        vector_field: str = "embedding",
    ):
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.top_k = top_k
        self.ollama_client = ollama_client

        self.store: Optional[ElasticsearchStore] = None
        if es_url:
            es_client = ElasticsearchClient(url=es_url, index_name=index_name)
            self.store = es_client.get_langchain_store(
                embedding_model=embedding_model,
                text_field=text_field,
                vector_field=vector_field,
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

    def _doc_builder(self, hit: Dict) -> Document:
        src = hit.get("_source", {}) or {}

        md = src.get("metadata", {}) or {}

        # fallback на старый формат (если в ES ещё старые документы)
        filename = md.get("filename") or src.get("filename") or "unknown"
        chunk_id = md.get("chunk_id") or src.get("chunk_id") or 0
        total_chunks = md.get("total_chunks") or src.get("total_chunks") or 0

        return Document(
            page_content=src.get("content", "") or "",
            metadata={
                "filename": filename,
                "chunk_id": int(chunk_id) if chunk_id else 0,
                "total_chunks": int(total_chunks) if total_chunks else 0,
                "_index": hit.get("_index"),
                "_id": hit.get("_id"),
            },
        )

    def _search_elasticsearch(self, query: str, top_k: int) -> List[Dict]:
        pairs = self.store.similarity_search_with_score(
            query=query,
            k=top_k,
            doc_builder=self._doc_builder,
        )

        results: List[Dict] = []
        for rank, (doc, score) in enumerate(pairs, start=1):
            metadata = doc.metadata or {}

            filename = metadata.get("filename") or "unknown"
            chunk_id = metadata.get("chunk_id") or 0
            total_chunks = metadata.get("total_chunks") or 0

            results.append(
                {
                    "text": doc.page_content,
                    "score": float(score),
                    "rank": rank,
                    "source": str(filename),
                    "chunk_id": int(chunk_id) if chunk_id else 0,
                    "total_chunks": int(total_chunks) if total_chunks else 0,
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

    def _search_local(self, query: str, top_k: int) -> List[Dict]:
        if not hasattr(self, "local_chunks"):
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        query_embedding = self.embedding_model.encode(query)
        results = []

        for chunk in self.local_chunks:
            chunk_embedding = self.embedding_model.encode(chunk["text"])
            score = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
            results.append(
                {
                    "text": chunk["text"],
                    "score": float(score),
                    "source": chunk.get("source", "unknown"),
                    "rank": 0,
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results[:top_k], 1):
            r["rank"] = i
        return results[:top_k]
