"""
Загрузка данных (документы, вопросы)
"""

from load.loader import (
    load_documents_local,
    ensure_elasticsearch_ready,
    setup_directories
)
from load.questions import (
    load_questions,
    extract_questions,
    extract_questions_from_elasticsearch,
    save_questions
)

__all__ = [
    # Документы
    "load_documents_local",
    "ensure_elasticsearch_ready",
    "setup_directories",
    # Вопросы
    "load_questions",
    "extract_questions",
    "extract_questions_from_elasticsearch",
    "save_questions",
]