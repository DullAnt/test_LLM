"""
Загрузка документов в Elasticsearch и автогенерация вопросов.
"""
from pathlib import Path
from connections.config import Config
from connections.elastic import ElasticsearchClient
from llm.embeddings import get_embedding_model
from load.questions import extract_questions_from_elasticsearch
from load.question_generator import generate_questions_from_files, save_questions_json
from connections.config import Config


DEFAULT_DOCUMENTS_PATH = Path(Config.DOCUMENTS_PATH)
DEFAULT_QUESTIONS_PATH = Path(Config.QUESTIONS_PATH)

def ensure_dirs() -> None:
    Path(DEFAULT_DOCUMENTS_PATH).mkdir(parents=True, exist_ok=True)
    Path("data/questions").mkdir(parents=True, exist_ok=True)
    Path("data/reports").mkdir(parents=True, exist_ok=True)

def load_documents_to_es(files) -> None:
    es_client = ElasticsearchClient(url=Config.ELASTIC_URL, index_name=Config.ELASTIC_INDEX)
    
    print("[EMBEDDINGS] Loading model to determine dimensions...")
    emb_model = get_embedding_model() # Модель загружается здесь
    
    # Получаем размерность реального вектора от модели
    # sentence-transformers модели имеют метод .get_sentence_embedding_dimension()
    if hasattr(emb_model._model, "get_sentence_embedding_dimension"):
        real_dims = emb_model._model.get_sentence_embedding_dimension()
    print("[AUTO-DETECT] Calculating embedding dimensions...")
    
    # Просто кодируем слово "test" и смотрим длину вектора.
    # Это работает всегда, независимо от того, как внутри устроен класс.
    try:
        test_emb = emb_model.encode("test")
        # Если encode возвращает список векторов (для батча), берем первый
        if hasattr(test_emb, "shape") and len(test_emb.shape) > 1:
             real_dims = test_emb.shape[1]
        else:
             real_dims = len(test_emb)
             
        print(f"[AUTO-DETECT] Success! Model dimensions: {real_dims}")
    except Exception as e:
        print(f"[WARN] Auto-detect failed ({e}). Fallback to Config.")
        real_dims = int(getattr(Config, "EMBEDDING_DIMS", 1024))

    dims = real_dims

    
    chunk_size = int(getattr(Config, "CHUNK_SIZE", 500))
    overlap = int(getattr(Config, "CHUNK_OVERLAP", 50))

    # Создаем индекс с ПРАВИЛЬНЫМИ размерами
    es_client.create_index_with_vectors(dims=dims)
    
    # Загружаем документы
    es_client.load_documents_with_vectors(
        files=files,
        embedding_model=emb_model,
        chunk_size=chunk_size,
        overlap=overlap,
        dims=dims,
    )
    es_client.verify_index()


def ensure_questions_file(files) -> None:
    out_path = Path(DEFAULT_QUESTIONS_PATH)
    
    # 1. Если файл есть - удаляем его, чтобы гарантировать генерацию новых вопросов
    if out_path.exists():
        try:
            out_path.unlink()
            print(f"[QUESTIONS] Старый файл удален: {DEFAULT_QUESTIONS_PATH}")
        except Exception as e:
            print(f"[WARN] Не удалось удалить старый файл: {e}")

    print("[QUESTIONS] Генерация вопросов")
    
    # Генерируем вопросы (теперь с использованием Ollama внутри)
    # max_questions_per_file можно настроить
    questions = generate_questions_from_files(files, max_questions_per_file=5)
    
    if questions:
        n = save_questions_json(questions, DEFAULT_QUESTIONS_PATH)
        print(f"[QUESTIONS] Успешно сохранено {n} вопросов -> {DEFAULT_QUESTIONS_PATH}")
    else:
        print("[WARN] Не удалось сгенерировать вопросы (возможно, тексты слишком короткие или LLM недоступна)")

def main() -> None:
    print("============================================================")
    print("ЗАГРУЗКА ДОКУМЕНТОВ В ELASTICSEARCH + AI QUESTIONS GEN")
    print("============================================================")
    
    ensure_dirs()
    es_client = ElasticsearchClient(url=Config.ELASTIC_URL, index_name=Config.ELASTIC_INDEX)
    
    # Ищем файлы документов
    files = es_client.find_documents(DEFAULT_DOCUMENTS_PATH)
    
    if not files:
        print(f"[ERROR] Не найдены документы в {DEFAULT_DOCUMENTS_PATH}")
        return

    # 1. Загружаем в базу
    load_documents_to_es(files)
    
    # 2. Генерируем вопросы для тестов
    ensure_questions_file(files)
    
    print("\n[DONE] Готово.")

if __name__ == "__main__":
    main()
