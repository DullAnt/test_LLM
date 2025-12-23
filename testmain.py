"""
TEST_LLM - Main Entry Point
Система тестирования LLM с поддержкой локальных документов и Elasticsearch
Оптимизирована для слабого ПК (16GB RAM + 4GB GPU)
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from package.config import Config
from rag.ollama_client import OllamaClient
from rag.mock_client import MockOllamaClient
from rag.embeddings import EmbeddingModel
from rag.retriever import DocumentRetriever
from rag.hyde import HyDEGenerator
from evaluate.questions import load_questions, extract_questions, extract_questions_from_elasticsearch
from evaluate.similarity import calculate_similarity
from evaluate.metrics import generate_html_report
from package.elastic import ElasticsearchClient


def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="TEST_LLM Evaluation System (Optimized for Weak PC)"
    )
    
    # Ollama settings
    parser.add_argument(
        "--model",
        type=str,
        default="gemma2:2b",
        help="Ollama model name (default: gemma2:2b - lightweight for CPU)"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama host URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Ollama timeout in seconds (default: 300 for CPU)"
    )
    
    # Data source settings
    parser.add_argument(
        "--documents",
        type=str,
        default="data/documents",
        help="Path to documents directory (default: data/documents)"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Path to questions JSONL file (optional)"
    )
    
    # Q&A extraction settings
    parser.add_argument(
        "--extract-qa",
        action="store_true",
        default=True,
        help="Auto-extract Q&A from documents (default: True)"
    )
    parser.add_argument(
        "--no-extract-qa",
        dest="extract_qa",
        action="store_false",
        help="Disable auto Q&A extraction"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of relevant chunks to retrieve (default: 3 for weak PC)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for correct answers (default: 0.7)"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=10,
        help="Maximum number of questions to test (default: 10)"
    )
    
    # HyDE settings
    parser.add_argument(
        "--hyde",
        action="store_true",
        help="Enable HyDE (Hypothetical Document Embeddings) for improved retrieval"
    )
    
    # Mock mode
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock Ollama client (no real LLM, for testing architecture)"
    )
    
    # Elasticsearch settings
    parser.add_argument(
        "--elasticsearch",
        action="store_true",
        help="Use Elasticsearch as document source"
    )
    parser.add_argument(
        "--es-host",
        type=str,
        default="localhost",
        help="Elasticsearch host (default: localhost)"
    )
    parser.add_argument(
        "--es-port",
        type=int,
        default=9200,
        help="Elasticsearch port (default: 9200)"
    )
    parser.add_argument(
        "--es-index",
        type=str,
        default="psb_docs",
        help="Elasticsearch index name (default: psb_docs)"
    )
    
    return parser.parse_args()


def load_documents_local(documents_path: str):
    """Загрузка документов из локальной директории"""
    print(f"[DOCS] Загрузка документов из {documents_path}...")
    
    documents = []
    path = Path(documents_path)
    
    if not path.exists():
        print(f"[FAIL] Директория {documents_path} не существует!")
        return documents
    
    # Загружаем все .md и .txt файлы
    for file_path in path.glob("**/*.md"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append({
                    "content": content,
                    "filename": file_path.name
                })
                print(f"  [OK] {file_path.name}")
        except Exception as e:
            print(f"  [ERROR] Ошибка при чтении {file_path.name}: {e}")
    
    for file_path in path.glob("**/*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append({
                    "content": content,
                    "filename": file_path.name
                })
                print(f"  [OK] {file_path.name}")
        except Exception as e:
            print(f"  [ERROR] Ошибка при чтении {file_path.name}: {e}")
    
    print(f"[SUCCESS] Загружено документов: {len(documents)}")
    return documents


def load_documents_elasticsearch(es_host: str, es_port: int, es_index: str):
    """Загрузка документов из Elasticsearch"""
    print(f"[DOCS] Загрузка документов из Elasticsearch ({es_host}:{es_port}/{es_index})...")
    
    try:
        es_client = ElasticsearchClient(host=es_host, port=es_port)
        
        # Проверяем подключение
        if not es_client.ping():
            print("[FAIL] Не удалось подключиться к Elasticsearch!")
            return []
        
        # Получаем все документы
        documents = es_client.get_all_documents(index_name=es_index)
        print(f"[SUCCESS] Загружено документов из Elasticsearch: {len(documents)}")
        
        return documents
        
    except Exception as e:
        print(f"[FAIL] Ошибка при работе с Elasticsearch: {e}")
        return []


def main():
    """Главная функция"""
    print("=" * 80)
    print("TEST_LLM - Система тестирования LLM")
    print("Оптимизирована для слабого ПК (16GB RAM + 4GB GPU)")
    print("=" * 80)
    print()
    
    # Парсинг аргументов
    args = parse_arguments()
    
    # Вывод конфигурации
    print("[CONFIG] Конфигурация:")
    if args.mock:
        print(f"  Режим: Mock (БЕЗ Ollama)")
    else:
        print(f"  Модель: {args.model}")
        print(f"  Ollama Host: {args.ollama_host}")
        print(f"  Timeout: {args.timeout}s")
    print(f"  TOP_K: {args.top_k}")
    print(f"  Порог схожести: {args.threshold}")
    print(f"  Максимум вопросов: {args.max_questions}")
    print(f"  HyDE режим: {'ВКЛЮЧЕН' if args.hyde else 'ВЫКЛЮЧЕН'}")
    print()
    
    # === ШАГ 1: Загрузка документов ===
    if args.elasticsearch:
        documents = load_documents_elasticsearch(
            es_host=args.es_host,
            es_port=args.es_port,
            es_index=args.es_index
        )
    else:
        documents = load_documents_local(args.documents)
    
    if not documents:
        print("[FAIL] Нет документов для обработки!")
        return
    
    print()
    
    # === ШАГ 2: Загрузка или извлечение вопросов ===
    questions = None
    
    if args.questions:
        # Загружаем готовые вопросы из файла
        print(f"[QUESTIONS] Загрузка вопросов из {args.questions}...")
        questions = load_questions(args.questions)
        
        # Ограничить количество
        if len(questions) > args.max_questions:
            print(f"[INFO] Загружено {len(questions)} вопросов, используем первые {args.max_questions}")
            questions = questions[:args.max_questions]
        
        print(f"[SUCCESS] Используется вопросов: {len(questions)}")
    
    elif args.extract_qa:
        # Автоматическое извлечение вопросов из документов
        if args.elasticsearch:
            print("[SEARCH] Извлечение вопросов из Elasticsearch документов...")
            es_client = ElasticsearchClient(host=args.es_host, port=args.es_port)
            questions = extract_questions_from_elasticsearch(es_client, index_name=args.es_index)
        else:
            print("[SEARCH] Автоматическое извлечение вопросов из документов...")
            questions = extract_questions(documents)
        
        # ОГРАНИЧЕНИЕ ДО СОХРАНЕНИЯ
        if questions and len(questions) > args.max_questions:
            print(f"[INFO] Извлечено {len(questions)} вопросов, используем первые {args.max_questions}")
            questions = questions[:args.max_questions]
        
        if questions:
            # Сохраняем ОГРАНИЧЕННЫЙ список
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/testsets/questions_extracted_{timestamp}.jsonl"
            
            os.makedirs("data/testsets", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                import json
                for q in questions:
                    f.write(json.dumps(q, ensure_ascii=False) + "\n")
            
            print(f"[SUCCESS] Используется вопросов: {len(questions)}")
            print(f"[SAVED] Сохранено в: {output_path}")
        else:
            print("[WARNING] Вопросы не найдены в документах")
    
    else:
        print("[WARNING] Не указан источник вопросов и отключено автоизвлечение")
        print("[TIP] Используйте --questions или --extract-qa")
        return
    
    if not questions:
        print("[FAIL] Нет вопросов для тестирования!")
        return
    
    print()
    
    # === ШАГ 3: Инициализация компонентов ===
    print("[INIT] Инициализация компонентов...")
    
    # Embedding модель
    print("  Инициализация embedding модели...")
    embedding_model = EmbeddingModel()
    
    # Ollama клиент (или Mock)
    if args.mock:
        print("  Инициализация Mock Ollama клиента (БЕЗ реального LLM)...")
        ollama_client = MockOllamaClient()
    else:
        print(f"  Инициализация Ollama клиента ({args.model})...")
        ollama_client = OllamaClient(
            host=args.ollama_host,
            model=args.model,
            timeout=args.timeout
        )
        
        # Проверка подключения к Ollama
        if not ollama_client.check_connection():
            print("[FAIL] Не удалось подключиться к Ollama!")
            print("[TIP] Используйте --mock для тестирования без Ollama")
            return
    
    # HyDE Generator (если включен)
    hyde_generator = None
    if args.hyde:
        print("  Инициализация HyDE генератора...")
        hyde_generator = HyDEGenerator(ollama_client)
        if args.mock:
            print("  [WARNING] HyDE в Mock режиме - будет использовать заглушки")
    
    # Document retriever
    print("  Инициализация retriever...")
    retriever = DocumentRetriever(
        embedding_model=embedding_model,
        documents=documents,
        top_k=args.top_k,
        use_hyde=args.hyde,
        hyde_generator=hyde_generator
    )
    
    print("[SUCCESS] Компоненты инициализированы")
    print()
    
    # === ШАГ 4: Тестирование ===
    print("[TEST] Начало тестирования...")
    print(f"  Вопросов: {len(questions)}")
    if args.mock:
        hyde_time = " (+ HyDE генерация)" if args.hyde else ""
        print(f"  Ожидаемое время: ~{len(questions) * (1.0 if args.hyde else 0.5):.1f} секунд (Mock режим{hyde_time})")
    else:
        hyde_time = " (+ HyDE генерация)" if args.hyde else ""
        print(f"  Ожидаемое время: ~{len(questions) * (3 if args.hyde else 2):.1f} минут (gemma2:2b на CPU{hyde_time})")
    print()
    
    results = []
    
    for i, question_data in enumerate(questions, 1):
        question = question_data["question"]
        expected_answer = question_data.get("answer", "")
        
        print(f"[{i}/{len(questions)}] Вопрос: {question[:80]}...")
        
        try:
            # Поиск релевантных документов С ДЕТАЛЬНОЙ ИНФОРМАЦИЕЙ
            retrieved_chunks = retriever.retrieve_with_scores(question, return_hyde_info=args.hyde)
            
            # Вывод информации о найденных chunks
            print(f"  [RAG] Найдено {len(retrieved_chunks)} chunks:")
            for chunk_info in retrieved_chunks:
                print(f"    Ранг {chunk_info['rank']}: {chunk_info['source']} (score: {chunk_info['score']:.2%})")
            if len(retrieved_chunks) > 0:
                print(f"  [DEBUG] Первый chunk (top-1):")
                top_chunk_text = retrieved_chunks[0]['text'][:200]
                print(f"    {top_chunk_text}...")
            # Извлечь только тексты для генерации
            relevant_texts = [chunk['text'] for chunk in retrieved_chunks]
            
            # Генерация ответа
            generated_answer = ollama_client.generate(question, relevant_texts)
            
            # Вычисление схожести
            similarity = calculate_similarity(generated_answer, expected_answer)
            
            # Определение правильности
            is_correct = similarity >= args.threshold
            
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "similarity": similarity,
                "is_correct": is_correct,
                "retrieved_chunks": retrieved_chunks  # С HyDE информацией если включено
            })
            
            status = "[OK]" if is_correct else "[FAIL]"
            print(f"  {status} Схожесть: {similarity:.2%} | Ответ: {generated_answer[:60]}...")
            
        except Exception as e:
            print(f"  [WARNING] Ошибка: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": f"ERROR: {str(e)}",
                "similarity": 0.0,
                "is_correct": False,
                "retrieved_chunks": []
            })
    
    print()
    
    # === ШАГ 5: Генерация отчета ===
    print("[REPORT] Генерация отчета...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"report_{'hyde_' if args.hyde else ''}{timestamp}.html"
    report_path = f"data/reports/{report_filename}"
    
    os.makedirs("data/reports", exist_ok=True)
    
    model_display = args.model if not args.mock else "Mock"
    if args.hyde:
        model_display += " + HyDE"
    
    generate_html_report(
        results=results,
        output_path=report_path,
        threshold=args.threshold,
        model_name=model_display,
        top_k=args.top_k
    )
    
    print(f"[SUCCESS] Отчет сохранен: {report_path}")
    print()
    
    # === ШАГ 6: Итоговая статистика ===
    correct_count = sum(1 for r in results if r["is_correct"])
    total_count = len(results)
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    avg_similarity = sum(r["similarity"] for r in results) / total_count if total_count > 0 else 0
    
    # RAG статистика
    all_chunks = []
    for r in results:
        chunks = r.get('retrieved_chunks', [])
        all_chunks.extend(chunks)
    
    avg_chunk_score = sum(c.get('score', 0) for c in all_chunks) / len(all_chunks) if all_chunks else 0
    
    print("=" * 80)
    print("[STATS] ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 80)
    print(f"  Всего вопросов: {total_count}")
    print(f"  Правильных ответов: {correct_count}")
    print(f"  Неправильных ответов: {total_count - correct_count}")
    print(f"  Точность: {accuracy:.1f}%")
    print(f"  Средняя схожесть: {avg_similarity:.2%}")
    print(f"  Порог схожести: {args.threshold:.2%}")
    print(f"  Качество RAG (средний score chunks): {avg_chunk_score:.2%}")
    if args.hyde:
        print(f"  HyDE: ВКЛЮЧЕН (улучшенный поиск)")
    print("=" * 80)
    print()
    print(f"[DONE] Тестирование завершено! Отчет: {report_path}")


if __name__ == "__main__":
    main()