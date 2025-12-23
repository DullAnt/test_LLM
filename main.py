"""
Основной Main для запуска тестирования
"""

from test_llm.package.CLI import parse_arguments
from test_llm.package.loader import load_documents_local, load_documents_elasticsearch, setup_directories
from test_llm.package.evaluator import RAGEvaluator


def main():
    """Главная функция"""
    print("=" * 80)
    print("TEST_LLM - СИСТЕМА ТЕСТИРОВАНИЯ RAG")
    print("=" * 80)
    
    # Парсинг аргументов
    args = parse_arguments()
    
    # Вывод конфигурации
    print(f"\n[CONFIG] Конфигурация:")
    print(f"  Модель: {args.model}")
    print(f"  HyDE: {'Включен' if args.hyde else 'Выключен'}")
    print(f"  TOP_K: {args.top_k}")
    print(f"  Порог: {args.threshold:.0%}")
    print(f"  Макс. вопросов: {args.max_questions}")
    if args.local_files:
        print(f"  Источник: Локальные файлы ({args.documents})")
    else:
        print(f"  Источник: Elasticsearch ({args.es_host}:{args.es_port}/{args.es_index})")
    
    # Создание директорий
    setup_directories()
    
    # Загрузка документов
    documents = None
    es_client = None
    
    if args.local_files:
        # Режим локальных файлов (дополнительная опция)
        documents = load_documents_local(args.documents)
        if not documents:
            print("[ERROR] Не найдено документов")
            print("[TIP] Положите документы в data/documents/")
            return
    else:
        # Режим Elasticsearch (по умолчанию)
        documents, es_client = load_documents_elasticsearch(
            es_host=args.es_host,
            es_port=args.es_port,
            es_index=args.es_index
        )
        if documents is None:
            return
    
    # Создание evaluator
    evaluator = RAGEvaluator(
        model=args.model,
        ollama_host=args.ollama_host,
        timeout=args.timeout,
        top_k=args.top_k,
        threshold=args.threshold,
        use_hyde=args.hyde,
        random_seed=args.seed
    )
    
    # Запуск оценки
    result = evaluator.run_evaluation(
        documents=documents,
        questions_path=args.questions,
        max_questions=args.max_questions,
        extract_qa=args.extract_qa,
        es_client=es_client,
        es_index=args.es_index if not args.local_files else None
    )
    
    if result:
        print("\nТестирование успешно завершено!")
    else:
        print("\nТестирование завершено с ошибками")


if __name__ == "__main__":
    main()