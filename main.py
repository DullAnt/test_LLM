# ============================================
# main.py
# ============================================

from package.CLI import parse_arguments
from package.loader import load_documents_local, ensure_elasticsearch_ready, setup_directories
from package.evaluator import RAGEvaluator
from package.config import Config


def main():
    print("=" * 80)
    print("TEST_LLM - СИСТЕМА ТЕСТИРОВАНИЯ RAG")
    print("=" * 80)

    args = parse_arguments()
    setup_directories()

    # Определяем режим работы
    if args.local_files:
        print("\n[MODE] Local Files Mode")
        documents = load_documents_local(args.documents)
        if not documents:
            print("[ERROR] Нет документов для загрузки")
            return
    else:
        print("\n[MODE] Elasticsearch Mode (default)")
        ok = ensure_elasticsearch_ready("localhost", 9200, args.es_index)
        if not ok:
            print("[ERROR] Elasticsearch недоступен")
            return
        documents = []  # В ES-режиме документы в память не грузим

    # Создаем evaluator
    evaluator = RAGEvaluator(
        model=args.model,
        ollama_host=args.ollama_host,
        timeout=args.timeout,
        top_k=args.top_k,
        threshold=args.threshold,
        use_hyde=args.hyde,
        random_seed=args.seed,
    )

    # Запускаем оценку
    result = evaluator.run_evaluation(
        documents=documents,
        questions_path=args.questions,
        max_questions=args.max_questions,
        extract_qa=args.extract_qa,
        es_index=args.es_index,
    )

    if result:
        print("\nТестирование успешно завершено!")
    else:
        print("\nТестирование завершено с ошибками")


if __name__ == "__main__":
    main()