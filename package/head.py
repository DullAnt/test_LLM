"""
Head - единое хранилище параметров RAG системы

Использование:
1. Программный API через HeadRAG.run()
2. Глобальное изменение дефолтных параметров

Примеры:
    # Изменить глобально
    from package import head
    head.top_k = 10
    head.need_hyde = False
    
    # Использовать в HeadRAG
    result = HeadRAG.run("Вопрос")  # использует head.top_k, head.need_hyde
    result = HeadRAG.run("Вопрос", top_k=5)  # переопределяет только top_k
"""

from pydantic import BaseModel
from typing import Literal, Optional

from package.config import Config
from rag.embeddings import get_embedding_model  
from rag.ollama_client import OllamaClient
from rag.retriever import DocumentRetriever


# ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ (можно менять)

top_k = 5
need_hyde = True
model = Config.OLLAMA_MODEL
embeddings = "paraphrase-multilingual-MiniLM-L12-v2"


# МОДЕЛИ ДАННЫХ

class RagDocument(BaseModel):
    """Документ в результате RAG"""
    text: str
    source: str


class RunProps(BaseModel):
    """Результат RAG запроса"""
    answer: str
    docs: list[RagDocument]
    debug: bool = True


# RAG API

class HeadRAG:
    """
    Простой API для выполнения RAG запросов.
    
    Приоритет параметров:
    1. Явно переданный в run() → используется он
    2. None в run() → используется из head модуля
    3. Пустая строка в run() → используется из Config
    """
    
    @staticmethod
    def run(
        query: str,
        db_name: Literal["all", ""] = "all",
        top_k: Optional[int] = None,
        need_hyde: Optional[bool] = None,
        model: Optional[str] = None,
        embeddings: Optional[str] = None,
    ) -> RunProps:
        """
        Выполнить RAG запрос
        
        Args:
            query: Вопрос пользователя
            db_name: Название БД (резерв для будущего)
            top_k: Количество chunks (если None → из head, если 0 → из Config)
            need_hyde: Использовать HyDE (если None → из head)
            model: Ollama модель (если None → из head, если "" → из Config)
            embeddings: Embedding модель (если None → из head, если "" → из Config)
        
        Returns:
            RunProps с ответом и документами
        
        Examples:
            # Использовать параметры из head
            result = HeadRAG.run("Что такое ИЗП?")
            
            # Переопределить конкретный параметр
            result = HeadRAG.run("Вопрос", top_k=10)
            
            # Изменить head глобально
            from package import head
            head.top_k = 5
            result = HeadRAG.run("Вопрос")  # использует top_k=5
        """
        
        # Импортируем себя как модуль для доступа к переменным
        from package import head as head_module
        
        # Определяем параметры по приоритету
        use_top_k = top_k if top_k is not None else head_module.top_k
        use_hyde = need_hyde if need_hyde is not None else head_module.need_hyde
        use_model = model if model is not None else (head_module.model or Config.OLLAMA_MODEL)
        use_embeddings = embeddings if embeddings is not None else (head_module.embeddings or Config.EMBEDDING_MODEL)
        
        # Создаем компоненты
        embedding_model = get_embedding_model(use_embeddings)
        
        ollama = OllamaClient(
            host=Config.OLLAMA_HOST,
            model=use_model,
            timeout=Config.OLLAMA_TIMEOUT,
        )
        
        retriever = DocumentRetriever(
            embedding_model=embedding_model,
            index_name=Config.ELASTIC_INDEX,
            top_k=use_top_k,
            ollama_client=ollama,
            es_url=Config.ELASTIC_URL,
            es_user=Config.ELASTIC_USER,
            es_password=Config.ELASTIC_PASSWORD,
            es_api_key=Config.ELASTIC_API_KEY,
            text_field="content",
            vector_field="embedding",
        )
        
        # Выполняем RAG
        chunks = retriever.retrieve_with_scores(
            query, 
            return_hyde_info=use_hyde, 
            top_k=use_top_k
        )
        context = [c["text"] for c in chunks]
        
        answer = ollama.generate(question=query, context=context)
        docs = [RagDocument(text=c["text"], source=c["source"]) for c in chunks]
        
        return RunProps(answer=answer, docs=docs, debug=True)