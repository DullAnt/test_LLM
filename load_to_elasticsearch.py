"""
Загрузка документов в Elasticsearch и автогенерация вопросов, если Q/A в ES не найдено.

Логика:
1) Создать/пересоздать индекс
2) Загрузить документы (чанки, embeddings, metadata)
3) Попробовать извлечь Q/A из ES
4) Если Q/A не найдено -> создать data/questions_autogen.json из файлов
"""

from pathlib import Path
from connections.config import Config
from connections.elastic import ElasticsearchClient
from llm.embeddings import get_embedding_model
from load.questions import extract_questions_from_elasticsearch
from load.question_generator import generate_questions_from_files, save_questions_json


DOCS_PATH = "data/documents"
QUESTIONS_PATH = "data/questions_autogen.json"


def ensure_dirs() -> None:
    Path(DOCS_PATH).mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)


def load_documents_to_es(files) -> None:
    es_client = ElasticsearchClient(url=Config.ELASTIC_URL, index_name=Config.ELASTIC_INDEX)
    emb_model = get_embedding_model()

    dims = int(getattr(Config, "EMBEDDING_DIMS", 1024))
    chunk_size = int(getattr(Config, "CHUNK_SIZE", 500))
    overlap = int(getattr(Config, "CHUNK_OVERLAP", 50))

    es_client.create_index_with_vectors(dims=dims)

    es_client.load_documents_with_vectors(
        files=files,
        embedding_model=emb_model,
        chunk_size=chunk_size,
        overlap=overlap,
        dims=dims,
    )

    es_client.verify_index()


def has_qa_in_es() -> bool:
    es_client = ElasticsearchClient(url=Config.ELASTIC_URL, index_name=Config.ELASTIC_INDEX)
    try:
        questions = extract_questions_from_elasticsearch(
            es_client=es_client.es,
            index=Config.ELASTIC_INDEX,
        )
        return bool(questions)
    except Exception as e:
        print(f"[WARN] Не удалось извлечь вопросы из ES: {e}")
        return False


def ensure_questions_file(files) -> None:
    if has_qa_in_es():
        print("[QUESTIONS] В Elasticsearch найден Q/A. Автогенерация questions_autogen.json не требуется.")
        return

    out_path = Path(QUESTIONS_PATH)

    if out_path.exists():
        print(f"[QUESTIONS] В ES нет Q/A, но файл уже существует: {QUESTIONS_PATH}")
        print("[QUESTIONS] Оставляем как есть.")
        return

    print("[QUESTIONS] В Elasticsearch не найден Q/A. Генерируем questions_autogen.json из файлов...")

    questions = generate_questions_from_files(files, max_questions_per_file=10)
    n = save_questions_json(questions, QUESTIONS_PATH)

    print(f"[QUESTIONS] Сгенерировано: {n} вопросов -> {QUESTIONS_PATH}")


def main() -> None:
    print("============================================================")
    print("ЗАГРУЗКА ДОКУМЕНТОВ В ELASTICSEARCH + AUTOGEN QUESTIONS")
    print("============================================================")

    ensure_dirs()

    es_client = ElasticsearchClient(url=Config.ELASTIC_URL, index_name=Config.ELASTIC_INDEX)

    files = es_client.find_documents(DOCS_PATH)

    load_documents_to_es(files)
    ensure_questions_file(files)

    print("\n[DONE] Готово.")


if __name__ == "__main__":
    main()
