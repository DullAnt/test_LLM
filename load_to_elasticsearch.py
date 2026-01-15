"""
Загрузка документов в Elasticsearch с векторами (embeddings)
"""

from pathlib import Path
from typing import List
from elasticsearch import Elasticsearch

from package.config import Config, DEFAULT_EMBEDDING_DIMS
from rag.embeddings import get_embedding_model


def check_elasticsearch_connection(es_url: str) -> Elasticsearch:
    """Проверка подключения к Elasticsearch"""
    es = Elasticsearch([es_url])

    if not es.ping():
        raise ConnectionError(f"Не удалось подключиться к Elasticsearch на {es_url}")

    print(f"[ES] Подключено: {es_url}")
    return es


def create_index_with_vectors(es: Elasticsearch, index_name: str, dims: int) -> None:
    """Создание индекса с поддержкой dense_vector"""

    if es.indices.exists(index=index_name):
        print(f"[ES] Индекс '{index_name}' уже существует. Удаляем...")
        es.indices.delete(index=index_name)

    body = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "content": {"type": "text", "analyzer": "standard"},
                "filename": {"type": "keyword"},
                "chunk_id": {"type": "integer"},
                "total_chunks": {"type": "integer"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    }

    es.indices.create(index=index_name, body=body)
    print(f"[ES] Индекс '{index_name}' создан (dense_vector dims={dims})")


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

    print(f"[DOCS] Найдено документов: {len(files)}")
    for f in files:
        print(f"   - {f.name}")

    return files


def load_documents_with_vectors(
    es: Elasticsearch,
    files: List[Path],
    index_name: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> None:
    """Загрузка документов с векторами"""

    print("\n[EMB] Инициализация embedding модели...")
    embedding_model = get_embedding_model()
    dims_expected = DEFAULT_EMBEDDING_DIMS

    total_docs = 0
    total_chunks = 0
    total_chars = 0

    for file_path in files:
        print(f"\n[FILE] {file_path.name}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            print("   [SKIP] Не удалось прочитать (кодировка).")
            continue

        char_count = len(content)
        total_chars += char_count

        chunks = split_into_chunks(content, chunk_size=chunk_size, overlap=overlap)
        chunk_count = len(chunks)

        print(f"   Символов: {char_count:,}")
        print(f"   Chunks: {chunk_count}")

        if chunk_count == 0:
            print("   [SKIP] Пустой файл после разбиения.")
            continue

        print("   [EMB] Вычисление векторов...")
        # Важно: твой EmbeddingModel поддерживает encode(), а для ST можно прогрессбар
        chunk_embeddings = embedding_model.encode(chunks, show_progress_bar=True)

        # sanity check dims
        try:
            vec0_len = len(chunk_embeddings[0])
            if vec0_len != dims_expected:
                print(
                    f"   [WARNING] Длина эмбеддинга={vec0_len}, "
                    f"а DEFAULT_EMBEDDING_DIMS={dims_expected}. "
                    f"Проверь EMBEDDING_MODEL/EMBEDDING_DIMS в .env"
                )
        except Exception:
            pass

        print("   [ES] Загрузка в Elasticsearch...")
        for i, (chunk_text, embedding) in enumerate(zip(chunks, chunk_embeddings), 1):
            doc = {
                "content": chunk_text,
                "filename": file_path.name,
                "chunk_id": i,
                "total_chunks": chunk_count,
                "embedding": embedding.tolist() if hasattr(embedding, "tolist") else list(embedding),
            }
            es.index(index=index_name, document=doc)

        total_docs += 1
        total_chunks += chunk_count
        print(f"   [OK] Загружено {chunk_count} chunks")

    print(f"\n{'=' * 60}")
    print("ИТОГО:")
    print(f"   Документов: {total_docs}")
    print(f"   Chunks: {total_chunks}")
    print(f"   Символов: {total_chars:,}")
    print(f"   Векторов: {total_chunks} x {DEFAULT_EMBEDDING_DIMS} = {total_chunks * DEFAULT_EMBEDDING_DIMS:,} чисел")
    print("=" * 60)


def verify_index(es: Elasticsearch, index_name: str) -> None:
    """Проверка индекса после загрузки"""
    es.indices.refresh(index=index_name)

    count = es.count(index=index_name)["count"]
    stats = es.indices.stats(index=index_name)
    size_bytes = stats["indices"][index_name]["total"]["store"]["size_in_bytes"]
    size_mb = size_bytes / (1024 * 1024)

    print(f"\n[VERIFY] Индекс '{index_name}':")
    print(f"   Документов (chunks): {count}")
    print(f"   Размер: {size_mb:.2f} MB")

    result = es.search(index=index_name, body={"size": 1, "query": {"match_all": {}}})
    if result["hits"]["hits"]:
        doc = result["hits"]["hits"][0]["_source"]
        print("\nПример chunk:")
        print(f"   Файл: {doc.get('filename')}")
        print(f"   Chunk: {doc.get('chunk_id')}/{doc.get('total_chunks')}")
        print(f"   Длина текста: {len(doc.get('content', ''))} символов")
        print(f"   Длина вектора: {len(doc.get('embedding', []))} чисел")
        print(f"   Текст: {doc.get('content','')[:100]}...")


def main():
    """Главная функция"""
    print("=" * 60)
    print("ЗАГРУЗКА ДОКУМЕНТОВ С ВЕКТОРАМИ В ELASTICSEARCH")
    print("=" * 60)

    ES_URL = Config.ELASTIC_URL          # <-- берём из .env / config
    INDEX_NAME = Config.ELASTIC_INDEX    # <-- берём из .env / config
    DOCS_PATH = Config.DOCUMENTS_PATH

    CHUNK_SIZE = 500
    OVERLAP = 50

    try:
        es = check_elasticsearch_connection(ES_URL)

        # dims берём из DEFAULT_EMBEDDING_DIMS (который берётся из env/registry)
        create_index_with_vectors(es, INDEX_NAME, dims=DEFAULT_EMBEDDING_DIMS)

        files = find_documents(DOCS_PATH)

        load_documents_with_vectors(
            es=es,
            files=files,
            index_name=INDEX_NAME,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP,
        )

        verify_index(es, INDEX_NAME)

        print("\nЗагрузка завершена успешно!")
        print("Проверка:")
        print(f"  curl {ES_URL}/{INDEX_NAME}/_count")

    except Exception as e:
        print(f"\nОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
