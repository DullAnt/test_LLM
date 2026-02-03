"""
Модуль для вычисления схожести текстов (Cosine Similarity)
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from llm.embeddings import get_embedding_model

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Вычисление cosine similarity между двумя текстами.
    Использует глобально загруженную модель эмбеддингов.
    """
    try:
        if not text1 or not text2:
            return 0.0
        
        # Получаем уже загруженный синглтон (без повторной загрузки)
        model = get_embedding_model()
        
        # Создаем embeddings
        e1 = model.encode(text1)
        e2 = model.encode(text2)
        
        # Reshape для sklearn (1, N)
        e1 = e1.reshape(1, -1)
        e2 = e2.reshape(1, -1)
        
        # Считаем косинус
        sim = float(cosine_similarity(e1, e2)[0][0])
        
        return max(0.0, min(1.0, sim))
        
    except Exception as e:
        print(f"[ERROR] Similarity calc failed: {e}")
        return 0.0
