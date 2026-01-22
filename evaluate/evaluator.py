"""
Evaluator для оценки качества RAG системы
"""

import time
import random
from datetime import datetime
from typing import List, Dict, Optional

from llm.ollama_client import OllamaClient
from llm.retriever import DocumentRetriever
from llm.embeddings import get_embedding_model
from load.questions import (
    load_questions,
    extract_questions,
    extract_questions_from_elasticsearch
)
from evaluate.similarity import calculate_similarity
from evaluate.metrics import generate_html_report
from connections.config import Config
from connections.elastic import ElasticsearchClient


class RAGEvaluator:
    def __init__(
        self,
        model: str,
        ollama_host: str,
        timeout: int,
        top_k: int,
        threshold: float,
        use_hyde: bool,
        random_seed: Optional[int] = None,
    ):
        self.model = model
        self.ollama_host = ollama_host
        self.timeout = timeout
        self.top_k = top_k
        self.threshold = threshold
        self.use_hyde = use_hyde

        if random_seed is not None:
            random.seed(random_seed)
            print(f"[RANDOM] Seed установлен: {random_seed}\n")

    def run_evaluation(
        self,
        documents: Optional[List[Dict]],
        questions_path: Optional[str],
        max_questions: int,
        extract_qa: bool,
        es_index: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Запуск оценки RAG системы.
        
        Логика загрузки вопросов:
        1. Если указан questions_path → загрузить из файла
        2. Иначе если documents пустой (ES режим) → извлечь из ES
        3. Иначе если extract_qa=True (local режим) → извлечь из documents
        4. Иначе → ошибка
        """

        questions = self._load_questions(
            documents=documents,
            questions_path=questions_path,
            max_questions=max_questions,
            extract_qa=extract_qa,
            es_index=es_index,
        )

        if not questions:
            print("[ERROR] Нет вопросов для тестирования")
            if documents:
                print("[TIP] Используй --extract-qa для извлечения из локальных документов")
            else:
                print("[TIP] Используй --questions <file> или убедись что в ES есть документы с Q&A")
            return None

        print("\n[INIT] Инициализация компонентов...")

        embedding_model = get_embedding_model()
        print("  Embedding модель OK")

        ollama_client = OllamaClient(host=self.ollama_host, model=self.model, timeout=self.timeout)
        if not ollama_client.check_connection():
            print("[ERROR] Не удалось подключиться к Ollama!")
            return None

        print(f"  Ollama клиент ({self.model}) OK")

        retriever = DocumentRetriever(
            embedding_model=embedding_model,
            index_name=es_index or Config.ELASTIC_INDEX,
            top_k=self.top_k,
            ollama_client=ollama_client,
            es_url=Config.ELASTIC_URL,
            es_user=Config.ELASTIC_USER,
            es_password=Config.ELASTIC_PASSWORD,
            es_api_key=Config.ELASTIC_API_KEY,
        )
        print("  Document retriever OK")

        results = self._run_tests(questions, retriever, ollama_client)

        report_path = self._generate_report(results)

        stats = self._calculate_stats(results)
        stats["report_path"] = report_path

        self._print_final_stats(stats)
        return stats

    def _load_questions(
        self,
        documents: Optional[List[Dict]],
        questions_path: Optional[str],
        max_questions: int,
        extract_qa: bool,
        es_index: Optional[str] = None,
    ) -> List[Dict]:
        """Загрузка вопросов"""

        # Приоритет 1: Явно указанный файл
        if questions_path:
            print(f"\n[QUESTIONS] Загрузка из {questions_path}...")
            questions = load_questions(questions_path)
            if len(questions) > max_questions:
                questions = random.sample(questions, max_questions)
            return questions

        # Приоритет 2: Извлечение из Elasticsearch
        if not documents or len(documents) == 0:
            print("\n[QUESTIONS] Извлечение из Elasticsearch...")
            
            # Используем единый ES клиент
            es_client = ElasticsearchClient.from_config(Config)
            
            # Извлекаем вопросы
            questions = extract_questions_from_elasticsearch(
                es_client=es_client.es,  # Передаем нативный ES клиент
                index=es_index or Config.ELASTIC_INDEX,
            )
            
            if len(questions) > max_questions:
                questions = random.sample(questions, max_questions)
            
            return questions

        # Приоритет 3: Извлечение из локальных документов
        if extract_qa:
            print("\n[QUESTIONS] Извлечение из локальных документов...")
            questions = extract_questions(documents)
            if len(questions) > max_questions:
                questions = random.sample(questions, max_questions)
            return questions

        return []

    def _run_tests(self, questions: List[Dict], retriever: DocumentRetriever, ollama_client: OllamaClient) -> List[Dict]:
        print(f"\n[TESTS] Начало тестирования ({len(questions)} вопросов)...")
        results = []
        
        for i, qd in enumerate(questions, 1):
            q = qd["question"]
            expected = qd.get("answer", "") or ""

            print(f"\n[{i}/{len(questions)}] {q[:80]}...")

            t0 = time.time()
            chunks = retriever.retrieve_with_scores(q, return_hyde_info=self.use_hyde, top_k=self.top_k)
            ctx = [c["text"] for c in chunks]

            ans = ollama_client.generate(q, ctx)
            dt = time.time() - t0

            sim = calculate_similarity(ans, expected) if expected else 0.0
            ok = sim >= self.threshold if expected else True

            print(f"  Время: {dt:.2f}s")
            print(f"  Similarity: {sim:.1%}")
            print(f"  {'✅' if ok else '❌'} {'Правильно' if ok else 'Неправильно'}")

            results.append({
                "question": q,
                "expected_answer": expected,
                "generated_answer": ans,
                "similarity": sim,
                "is_correct": ok,
                "retrieved_chunks": chunks,
                "response_time": dt,
            })
        
        return results

    def _generate_report(self, results: List[Dict]) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{Config.REPORTS_PATH}/report_{'hyde_' if self.use_hyde else ''}{ts}.html"
        generate_html_report(results, report_path, self.threshold, self.model, self.top_k)
        return report_path

    def _calculate_stats(self, results: List[Dict]) -> Dict:
        total = len(results)
        correct = sum(1 for r in results if r["is_correct"])
        acc = (correct / total * 100) if total else 0.0
        avg_sim = sum(r["similarity"] for r in results) / total if total else 0.0
        return {
            "accuracy": acc,
            "avg_similarity": avg_sim,
            "correct_count": correct,
            "total_count": total,
            "results": results
        }

    def _print_final_stats(self, stats: Dict):
        print("\n" + "=" * 80)
        print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print("=" * 80)
        print(f"Правильных ответов: {stats['correct_count']}/{stats['total_count']} ({stats['accuracy']:.1f}%)")
        print(f"Средняя схожесть: {stats['avg_similarity']:.1%}")
        print(f"Отчет: {stats['report_path']}")
        print("=" * 80)