from typing import List, Dict
from pathlib import Path
from connections.elastic import ElasticsearchClient


def setup_directories():
    for d in ["data/documents", "data/testsets", "data/reports"]:
        Path(d).mkdir(parents=True, exist_ok=True)


def load_documents_local(documents_path: str = "data/documents") -> List[Dict]:
    documents = []
    docs_dir = Path(documents_path)

    if not docs_dir.exists():
        print(f"[ERROR] Папка не найдена: {documents_path}")
        return []

    for file_path in docs_dir.rglob("*"):
        if file_path.suffix.lower() in [".txt", ".md"]:
            try:
                content = file_path.read_text(encoding="utf-8")
                documents.append({"filename": file_path.name, "content": content, "path": str(file_path)})
            except Exception as e:
                print(f"[WARNING] Ошибка чтения {file_path.name}: {e}")

    print(f"[DOCS] Загружено локально: {len(documents)} документов")
    return documents


def ensure_elasticsearch_ready(es_host: str, es_port: int, es_index: str) -> bool:
    """
    Проверка готовности Elasticsearch
    
    Args:
        es_host: хост (например, "localhost")
        es_port: порт (например, 9200)
        es_index: название индекса
    """
    try:
        # Формируем URL из host и port
        es_url = f"http://{es_host}:{es_port}"
        
        # Создаем клиент с правильными параметрами
        es_client = ElasticsearchClient(url=es_url, index_name=es_index)

        if not es_client.ping():
            print("[ERROR] Elasticsearch недоступен")
            return False

        if not es_client.index_exists():
            print(f"[ERROR] Индекс '{es_index}' не существует")
            return False

        count = es_client.get_document_count()
        if count <= 0:
            print(f"[ERROR] Индекс '{es_index}' пустой")
            return False

        print(f"[ES] OK: index={es_index}, chunks={count}")
        return True

    except Exception as e:
        print(f"[ERROR] ES check failed: {e}")
        return False