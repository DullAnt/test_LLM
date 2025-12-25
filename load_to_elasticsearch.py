"""
Загрузка документов в Elasticsearch с векторами (embeddings)
"""
import os
from pathlib import Path
from typing import List
from elasticsearch import Elasticsearch
from package.config import get_embedding_model, DEFAULT_EMBEDDING_DIMS


def check_elasticsearch_connection(es_host: str = "localhost", es_port: int = 9200) -> Elasticsearch:
    """Проверка подключения к Elasticsearch"""
    es = Elasticsearch([f"http://{es_host}:{es_port}"])
    
    if not es.ping():
        raise ConnectionError(f"Не удалось подключиться к Elasticsearch на {es_host}:{es_port}")
    
    print(f"Подключено к Elasticsearch: {es_host}:{es_port}")
    return es


def create_index_with_vectors(es: Elasticsearch, index_name: str) -> None:
    """Создание индекса с поддержкой векторов"""
    
    if es.indices.exists(index=index_name):
        print(f"Индекс '{index_name}' уже существует. Удаляем...")
        es.indices.delete(index=index_name)
    
    # Настройки индекса с векторным полем
    settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "filename": {
                    "type": "keyword"
                },
                "chunk_id": {
                    "type": "integer"
                },
                "total_chunks": {
                    "type": "integer"
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": DEFAULT_EMBEDDING_DIMS,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    
    es.indices.create(index=index_name, body=settings)
    print(f"Индекс '{index_name}' создан с поддержкой векторного поиска")


def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Разбивает текст на chunks с перекрытием"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        
        if start >= text_length - overlap:
            break
    
    return chunks


def find_documents(docs_path: str = "data/documents") -> List[Path]:
    """Поиск всех .txt и .md файлов"""
    docs_dir = Path(docs_path)
    
    if not docs_dir.exists():
        raise FileNotFoundError(f"Папка {docs_path} не найдена")
    
    files = list(docs_dir.glob("*.txt")) + list(docs_dir.glob("*.md"))
    files = [f for f in files if f.name.lower() != "readme.md"]
    
    if not files:
        raise FileNotFoundError(f"Не найдено .txt или .md файлов в {docs_path}")
    
    print(f"Найдено документов: {len(files)}")
    for f in files:
        print(f"   - {f.name}")
    
    return files


def load_documents_with_vectors(
    es: Elasticsearch,
    files: List[Path],
    index_name: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> None:
    """Загрузка документов с векторами"""
    
    # Получить embedding модель
    print("\nИнициализация embedding модели...")
    embedding_model = get_embedding_model()
    
    total_docs = 0
    total_chunks = 0
    total_chars = 0
    
    for file_path in files:
        print(f"\nОбработка: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"   Не удалось прочитать (неверная кодировка), пропускаем")
            continue
        
        char_count = len(content)
        total_chars += char_count
        
        # Разбиваем на chunks
        chunks = split_into_chunks(content, chunk_size=chunk_size, overlap=overlap)
        chunk_count = len(chunks)
        
        print(f"   Символов: {char_count:,}")
        print(f"   Chunks: {chunk_count}")
        
        # Вычисляем векторы для ВСЕХ chunks сразу (быстрее)
        print(f"   Вычисление векторов...")
        chunk_embeddings = embedding_model.encode(
            chunks, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Загружаем каждый chunk с вектором
        print(f"   Загрузка в Elasticsearch...")
        for i, (chunk_text, embedding) in enumerate(zip(chunks, chunk_embeddings), 1):
            doc = {
                "content": chunk_text,
                "filename": file_path.name,
                "chunk_id": i,
                "total_chunks": chunk_count,
                "embedding": embedding.tolist()  # Вектор (384 числа)
            }
            
            es.index(index=index_name, document=doc)
        
        total_docs += 1
        total_chunks += chunk_count
        
        print(f"   Загружено {chunk_count} chunks с векторами")
    
    print(f"\n{'='*60}")
    print(f"ИТОГО:")
    print(f"   Документов: {total_docs}")
    print(f"   Chunks: {total_chunks}")
    print(f"   Символов: {total_chars:,}")
    print(f"   Векторов: {total_chunks} x 384 = {total_chunks * 384:,} чисел")
    print("="*60)


def verify_index(es: Elasticsearch, index_name: str) -> None:
    """Проверка индекса после загрузки"""
    es.indices.refresh(index=index_name)
    
    count = es.count(index=index_name)['count']
    stats = es.indices.stats(index=index_name)
    size_bytes = stats['indices'][index_name]['total']['store']['size_in_bytes']
    size_mb = size_bytes / (1024 * 1024)
    
    print(f"\nПРОВЕРКА ИНДЕКСА '{index_name}':")
    print(f"   Документов (chunks): {count}")
    print(f"   Размер: {size_mb:.2f} MB")
    
    # Пример документа
    result = es.search(index=index_name, body={"size": 1, "query": {"match_all": {}}})
    if result['hits']['hits']:
        doc = result['hits']['hits'][0]['_source']
        print(f"\nПример chunk:")
        print(f"   Файл: {doc['filename']}")
        print(f"   Chunk: {doc['chunk_id']}/{doc['total_chunks']}")
        print(f"   Длина текста: {len(doc['content'])} символов")
        print(f"   Длина вектора: {len(doc['embedding'])} чисел")
        print(f"   Текст: {doc['content'][:100]}...")


def main():
    """Главная функция"""
    print("="*60)
    print("ЗАГРУЗКА ДОКУМЕНТОВ С ВЕКТОРАМИ В ELASTICSEARCH")
    print("="*60)
    
    ES_HOST = "localhost"
    ES_PORT = 9200
    INDEX_NAME = "psb_docs"
    DOCS_PATH = "data/documents"
    CHUNK_SIZE = 500
    OVERLAP = 50
    
    try:
        # 1. Подключение
        es = check_elasticsearch_connection(ES_HOST, ES_PORT)
        
        # 2. Создание индекса с векторами
        create_index_with_vectors(es, INDEX_NAME)
        
        # 3. Поиск документов
        files = find_documents(DOCS_PATH)
        
        # 4. Загрузка с векторами
        load_documents_with_vectors(
            es=es,
            files=files,
            index_name=INDEX_NAME,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP
        )
        
        # 5. Проверка
        verify_index(es, INDEX_NAME)
        
        print(f"\nЗагрузка завершена успешно!")
        print(f"Проверить в Kibana: http://localhost:5601")
        print(f"Количество chunks: curl http://localhost:9200/{INDEX_NAME}/_count")
        
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())