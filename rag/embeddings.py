# """
# Модуль для работы с embeddings

# DEPRECATED: Этот модуль оставлен для обратной совместимости.
# Рекомендуется использовать напрямую:
#     from package.config import get_embedding_model
#     model = get_embedding_model()
# """

# import numpy as np
# from package.config import get_embedding_model as _get_global_model


# class EmbeddingModel:
#     """
#     Legacy wrapper для обратной совместимости
    
#     УСТАРЕЛО: Используйте напрямую config.get_embedding_model()
    
#     Usage (NEW):
#         from package.config import get_embedding_model
#         model = get_embedding_model()
#         embedding = model.encode("текст")
    
#     Usage (OLD - still works):
#         from rag.embeddings import EmbeddingModel
#         model = EmbeddingModel()
#         embedding = model.encode("текст")
#     """
    
#     def __init__(self):
#         """Инициализация (использует глобальную модель из config)"""
#         print("⚠️  [DEPRECATED] rag.embeddings.EmbeddingModel устарел")
#         print("    Используйте: from package.config import get_embedding_model")
#         self.model = _get_global_model()
    
#     def encode(self, texts, show_progress=False, show_progress_bar=False):
#         """
#         Создание embeddings для текстов
        
#         Args:
#             texts: Строка или список строк
#             show_progress: Показывать прогресс (для совместимости)
#             show_progress_bar: Показывать прогресс бар
            
#         Returns:
#             Numpy array с embeddings
#         """
#         # Используем глобальную модель
#         embeddings = self.model.encode(
#             texts, 
#             show_progress_bar=show_progress or show_progress_bar
#         )
        
#         embeddings = np.array(embeddings)
        
#         # Для одиночной строки - flatten
#         if isinstance(texts, str):
#             embeddings = embeddings.flatten()
        
#         return embeddings
