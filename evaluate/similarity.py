"""
Модуль для вычисления схожести текстов
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from package.config import get_embedding_model


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Вычисление cosine similarity между двумя текстами
    
    Args:
        text1: Первый текст (сгенерированный ответ)
        text2: Второй текст (эталонный ответ)
        
    Returns:
        Коэффициент схожести от 0.0 до 1.0
    """
    try:
        # Проверка на пустые строки
        if not text1 or not text2:
            return 0.0
        
        # Получить глобальную модель из config (Singleton)
        model = get_embedding_model()
        
        # Создать embeddings
        embedding1 = model.encode(text1)
        embedding2 = model.encode(text2)
        
        # Преобразовать в 2D массивы для cosine_similarity
        embedding1 = np.array(embedding1).reshape(1, -1)
        embedding2 = np.array(embedding2).reshape(1, -1)
        
        # Вычислить cosine similarity
        similarity_matrix = cosine_similarity(embedding1, embedding2)
        
        # Извлечь скалярное значение из матрицы
        similarity = float(similarity_matrix[0][0])
        
        # Ограничить диапазон [0, 1]
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
        
    except Exception as e:
        print(f"Ошибка при вычислении схожести: {e}")
        return 0.0