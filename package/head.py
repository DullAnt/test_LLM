from pydantic import BaseModel
from typing import Literal

from package.config import Config
from package.embeddings import get_embedding_model
from rag.ollama_client import OllamaClient
from rag.retriever import DocumentRetriever


class RagDocument(BaseModel):
    text: str
    source: str


class RunProps(BaseModel):
    answer: str
    docs: list[RagDocument]
    debug: bool = True


class HeadRAG:
    @staticmethod
    def run(
        query: str,
        db_name: Literal["all", ""] = "all",
        top_k: int = Config.TOP_K,
        need_hyde: bool = True,
        model: str = "",
        embeddings: str = "",
    ) -> RunProps:
        embedding_model = get_embedding_model(embeddings or Config.EMBEDDING_MODEL)

        ollama = OllamaClient(
            host=Config.OLLAMA_HOST,
            model=model or Config.OLLAMA_MODEL,
            timeout=Config.OLLAMA_TIMEOUT,
        )

        retriever = DocumentRetriever(
            embedding_model=embedding_model,
            index_name=Config.ELASTIC_INDEX,
            top_k=top_k,
            ollama_client=ollama,
            es_url=Config.ELASTIC_URL,
            es_user=Config.ELASTIC_USER,
            es_password=Config.ELASTIC_PASSWORD,
            es_api_key=Config.ELASTIC_API_KEY,
            text_field="content",
            vector_field="embedding",
        )

        chunks = retriever.retrieve_with_scores(query, return_hyde_info=need_hyde, top_k=top_k)
        context = [c["text"] for c in chunks]

        answer = ollama.generate(question=query, context=context)
        docs = [RagDocument(text=c["text"], source=c["source"]) for c in chunks]

        return RunProps(answer=answer, docs=docs, debug=True)
