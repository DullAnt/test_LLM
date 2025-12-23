"""
Модуль для работы с embeddings
"""

from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """Класс для создания embeddings из текста"""
    
    def __init__(self):
        model_name = "ai-forever/FRIDA"
        print(f"Загрузка embedding модели: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(" Модель загружена")
    
    def encode(self, texts, show_progress=False, show_progress_bar=False):
        """
        Создание embeddings для текстов
        
        Args:
            texts: Строка или список строк
            show_progress: Показывать прогресс (для совместимости)
            show_progress_bar: Показывать прогресс бар
            
        Returns:
            Numpy array с embeddings
        """
        # Используем show_progress_bar параметр модели
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=show_progress or show_progress_bar
        )
        
        embeddings = np.array(embeddings)
        
        if isinstance(texts, str):
            embeddings = embeddings.flatten()
        
        return embeddings