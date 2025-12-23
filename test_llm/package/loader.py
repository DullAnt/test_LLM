"""
Загрузка документов из разных источников
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
from .elastic import ElasticsearchClient

def setup_directories():
    """Создание необходимых директорий"""
    from pathlib import Path
    
    # Создать необходимые папки
    directories = [
        "data/documents",
        "data/testsets",
        "data/reports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def load_documents_local(documents_path: str = "data/documents") -> List[Dict]:
    """
    Загрузка документов из локальных файлов
    
    Args:
        documents_path: Путь к папке с документами
        
    Returns:
        List[Dict]: Список документов
    """
    documents = []
    docs_dir = Path(documents_path)
    
    if not docs_dir.exists():
        print(f"[ERROR] Папка не найдена: {documents_path}")
        return []
    
    # Поддерживаемые форматы
    supported_formats = ['.txt', '.md']
    
    # Загрузка всех документов
    for file_path in docs_dir.rglob('*'):
        if file_path.suffix.lower() in supported_formats:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                documents.append({
                    'filename': file_path.name,
                    'content': content,
                    'path': str(file_path)
                })
            except Exception as e:
                print(f"[WARNING] Ошибка чтения {file_path.name}: {e}")
    
    if not documents:
        print(f"[WARNING] Не найдено документов в {documents_path}")
        print(f"[TIP] Поддерживаемые форматы: {', '.join(supported_formats)}")
    else:
        print(f"[DOCS] Загружено локально: {len(documents)} документов")
    
    return documents


def load_documents_elasticsearch(
    es_host: str = "localhost",
    es_port: int = 9200,
    es_index: str = "documents"
) -> Tuple[Optional[List[Dict]], Optional[ElasticsearchClient]]:
    """
    Загрузка документов из Elasticsearch
    
    Args:
        es_host: Хост Elasticsearch
        es_port: Порт Elasticsearch
        es_index: Название индекса
        
    Returns:
        Tuple[List[Dict], ElasticsearchClient]: Документы и клиент ES
    """
    try:
        # Подключение к Elasticsearch
        es_client = ElasticsearchClient(host=es_host, port=es_port, index_name=es_index)
        
        # Проверка подключения
        if not es_client.ping():
            print("[ERROR] Не удалось подключиться к Elasticsearch!")
            print("[TIP] Запустите: docker-compose up -d elasticsearch")
            print("[TIP] Или используйте: python main.py --local-files")
            return None, None
        
        # Проверка индекса
        if not es_client.index_exists():
            print(f"[ERROR] Индекс '{es_index}' не существует!")
            print("[TIP] Запустите: python load_to_elasticsearch.py")
            print("[TIP] Или используйте: python main.py --local-files")
            return None, None
        
        # Получение количества документов
        doc_count = es_client.get_document_count()
        
        if doc_count == 0:
            print(f"[WARNING] Индекс '{es_index}' пустой!")
            print("[TIP] Запустите: python load_to_elasticsearch.py")
            print("[TIP] Или используйте: python main.py --local-files")
            return None, None
        
        print(f"[DOCS] Документов в индексе: {doc_count}")
        
        # Загрузка всех документов
        documents = es_client.get_all_documents()
        
        if not documents:
            print("[ERROR] Не удалось загрузить документы из Elasticsearch")
            return None, None
        
        return documents, es_client
        
    except Exception as e:
        print(f"[ERROR] Ошибка подключения к Elasticsearch: {e}")
        print("[TIP] Запустите: docker-compose up -d elasticsearch")
        print("[TIP] Или используйте: python main.py --local-files")
        return None, None


def validate_documents(documents: List[Dict]) -> bool:
    """
    Валидация загруженных документов
    
    Args:
        documents: Список документов
        
    Returns:
        bool: True если документы валидны
    """
    if not documents:
        print("[ERROR] Список документов пустой")
        return False
    
    # Проверка структуры
    required_fields = ['content']
    
    for i, doc in enumerate(documents):
        # Проверка наличия обязательных полей
        for field in required_fields:
            if field not in doc:
                print(f"[ERROR] Документ {i}: отсутствует поле '{field}'")
                return False
        
        # Проверка что контент не пустой
        if not doc['content'] or not doc['content'].strip():
            print(f"[ERROR] Документ {i}: пустой контент")
            return False
    
    print(f"[DOCS] Валидация пройдена: {len(documents)} документов")
    return True