# """
# Head - тестовый интерфейс RAG системы

# Использование:
#     # Дефолтные параметры из Config
#     result = HeadRAG.run("Что такое ИЗП?")
    
#     # Переопределить конкретные параметры
#     result = HeadRAG.run("Вопрос", top_k=10, need_hyde=False)
    
#     # Изменить дефолты в head
#     from llm import head
#     head.top_k = 10
#     result = HeadRAG.run("Вопрос")  # использует top_k=10
# """

# from pydantic import BaseModel
# from typing import Optional

# from connections.config import Config
# from llm.embeddings import get_embedding_model
# from llm.ollama_client import OllamaClient
# from llm.retriever import DocumentRetriever


# # =============================================================================
# # ПАРАМЕТРЫ ПО УМОЛЧАНИЮ (можно менять)
# # =============================================================================

# top_k = Config.TOP_K
# need_hyde = Config.NEED_HYDE
# model = Config.OLLAMA_MODEL
# embeddings = Config.EMBEDDING_MODEL


# # =============================================================================
# # МОДЕЛИ ДАННЫХ
# # =============================================================================

# class RagDocument(BaseModel):
#     """Документ в результате RAG"""
#     text: str
#     source: str


# class RunProps(BaseModel):
#     """Результат RAG запроса"""
#     answer: str
#     docs: list[RagDocument]
#     debug: bool = True


# # =============================================================================
# # RAG API
# # =============================================================================

# class HeadRAG:
#     """
#     Тестовый интерфейс для RAG запросов
    
#     Приоритет параметров:
#     1. Явно переданный в run() → используется он
#     2. None в run() → используется из head модуля
#     3. В head модуле → значение из Config
#     """
    
#     @staticmethod
#     def run(
#         query: str,
#         top_k_param: Optional[int] = None,
#         need_hyde_param: Optional[bool] = None,
#         model_param: Optional[str] = None,
#         embeddings_param: Optional[str] = None,
#     ) -> RunProps:
#         """
#         Выполнить тестовый RAG запрос
        
#         Args:
#             query: Вопрос пользователя
#             top_k_param: Количество chunks (если None → из head → из Config)
#             need_hyde_param: Использовать HyDE (если None → из head → из Config)
#             model_param: Ollama модель (если None → из head → из Config)
#             embeddings_param: Embedding модель (если None → из head → из Config)
        
#         Returns:
#             RunProps с ответом и документами
        
#         Examples:
#             # Дефолтные параметры
#             result = HeadRAG.run("Что такое ИЗП?")
            
#             # Переопределить top_k
#             result = HeadRAG.run("Вопрос", top_k_param=10)
            
#             # Изменить дефолты через head
#             from llm import head
#             head.top_k = 10
#             head.need_hyde = False
#             result = HeadRAG.run("Вопрос")  # использует измененные значения
#         """
        
#         # Импортируем текущий модуль для доступа к глобальным переменным
#         import sys
#         current_module = sys.modules[__name__]
        
#         # Определяем параметры по приоритету
#         use_top_k = top_k_param if top_k_param is not None else current_module.top_k
#         use_hyde = need_hyde_param if need_hyde_param is not None else current_module.need_hyde
#         use_model = model_param if model_param is not None else current_module.model
#         use_embeddings = embeddings_param if embeddings_param is not None else current_module.embeddings
        
#         print(f"\n[HEAD] Параметры запроса:")
#         print(f"  query: {query[:50]}...")
#         print(f"  top_k: {use_top_k}")
#         print(f"  need_hyde: {use_hyde}")
#         print(f"  model: {use_model}")
#         print(f"  embeddings: {use_embeddings}")
        
#         # Создаем компоненты
#         embedding_model = get_embedding_model(use_embeddings)
        
#         ollama = OllamaClient(
#             host=Config.OLLAMA_HOST,
#             model=use_model,
#             timeout=Config.OLLAMA_TIMEOUT,
#         )
        
#         retriever = DocumentRetriever(
#             embedding_model=embedding_model,
#             index_name=Config.ELASTIC_INDEX,
#             top_k=use_top_k,
#             ollama_client=ollama,
#             es_url=Config.ELASTIC_URL,
#             es_user=Config.ELASTIC_USER,
#             es_password=Config.ELASTIC_PASSWORD,
#             es_api_key=Config.ELASTIC_API_KEY,
#             text_field="content",
#             vector_field="embedding",
#         )
        
#         # Выполняем RAG
#         print("\n[HEAD] Поиск релевантных документов...")
#         chunks = retriever.retrieve_with_scores(
#             query, 
#             return_hyde_info=use_hyde, 
#             top_k=use_top_k
#         )
        
#         context = [c["text"] for c in chunks]
        
#         print(f"[HEAD] Найдено {len(chunks)} документов")
#         print("[HEAD] Генерация ответа...")
        
#         answer = ollama.generate(question=query, context=context)
#         docs = [RagDocument(text=c["text"], source=c["source"]) for c in chunks]
        
#         print("[HEAD] Ответ сгенерирован\n")
        
#         return RunProps(answer=answer, docs=docs, debug=True)