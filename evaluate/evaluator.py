"""
Evaluator для оценки качества RAG системы
"""

import time
import random
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from llm.ollama_client import OllamaClient
from llm.retriever import DocumentRetriever
from llm.embeddings import get_embedding_model
from load.questions import (
    extract_questions,
    extract_questions_from_elasticsearch,
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

    # -----------------------------
    # Robust loader for questions
    # -----------------------------
    def _robust_load_questions_file(self, path: str) -> List[Dict]:
        """
        Устойчиво загружает вопросы из файла.
        Поддерживает:
          1) JSON-массив: [{"question":..., "answer":...}, ...]
          2) JSON-объект: {"questions":[...]}
          3) JSONL: по одному JSON-объекту на строку
        Корректно читает UTF-8 с BOM (utf-8-sig).
        """
        p = Path(path)

        if not p.exists():
            print(f"[ERROR] Файл вопросов не найден: {path}")
            return []

        raw = p.read_text(encoding="utf-8-sig", errors="replace").strip()
        preview = raw[:160].replace("\n", "\\n")

        if not raw:
            print("[ERROR] Файл вопросов пустой.")
            return []

        # 1) Пробуем как полноценный JSON
        try:
            data = json.loads(raw)

            if isinstance(data, list):
                return data

            if isinstance(data, dict) and isinstance(data.get("questions"), list):
                return data["questions"]

            print("[ERROR] Неподдерживаемый JSON-формат файла вопросов.")
            print("        Ожидаю: список объектов или {'questions':[...]} .")
            print(f"        Preview: {preview}")
            return []

        except json.JSONDecodeError:
            # 2) Пробуем как JSONL
            out: List[Dict] = []
            try:
                for ln in raw.splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    out.append(json.loads(ln))

                if out:
                    print("[QUESTIONS] Файл распознан как JSONL (1 JSON на строку).")
                    return out

                print("[ERROR] Файл не является валидным JSON и не содержит JSONL-строк.")
                print(f"        Preview: {preview}")
                return []

            except Exception as e:
                print("[ERROR] Не удалось распарсить файл вопросов ни как JSON, ни как JSONL.")
                print(f"        Причина: {e}")
                print(f"        Preview: {preview}")
                return []

    def _normalize_questions(self, questions: List[Dict]) -> List[Dict]:
        """
        Приводим элементы к формату:
        {"question": str, "answer": str(optional), "source": str(optional)}
        """
        normalized: List[Dict] = []

        for q in questions:
            if not isinstance(q, dict):
                continue

            question = q.get("question") or q.get("text") or q.get("q") or ""
            answer = q.get("answer") or q.get("expected_answer") or q.get("a") or ""
            source = q.get("source")

            question = str(question).strip()
            answer = str(answer).strip()

            if not question:
                continue

            item = {"question": question}
            if answer:
                item["answer"] = answer
            if source:
                item["source"] = source

            normalized.append(item)

        return normalized

    def _load_questions(
        self,
        documents: Optional[List[Dict]],
        questions_path: Optional[str],
        max_questions: int,
        extract_qa: bool,
        es_index: Optional[str] = None,
    ) -> List[Dict]:
        """
        Логика загрузки вопросов:
        1) Если задан questions_path -> читаем файл (JSON/JSONL)
        2) Иначе если ES-режим (documents пустой) -> пытаемся извлечь Q/A из ES
        3) Иначе если extract_qa=True -> извлекаем из локальных документов
        4) Иначе -> []
        """

        # 1) Приоритет: файл
        if questions_path:
            print(f"\n[QUESTIONS] Загрузка из {questions_path}...")
            qs = self._robust_load_questions_file(questions_path)
            qs = self._normalize_questions(qs)

            if len(qs) > max_questions:
                qs = random.sample(qs, max_questions)
            return qs

        # 2) ES-режим (documents пустой)
        if not documents:
            print("\n[QUESTIONS] Извлечение из Elasticsearch...")

            es_client = ElasticsearchClient(
                url=Config.ELASTIC_URL,
                index_name=es_index or Config.ELASTIC_INDEX,
            )

            qs = extract_questions_from_elasticsearch(
                es_client=es_client.es,
                index=es_index or Config.ELASTIC_INDEX,
            )
            qs = self._normalize_questions(qs)

            if len(qs) > max_questions:
                qs = random.sample(qs, max_questions)
            return qs

        # 3) Локальные документы
        if extract_qa:
            print("\n[QUESTIONS] Извлечение из локальных документов...")
            qs = extract_questions(documents)
            qs = self._normalize_questions(qs)

            if len(qs) > max_questions:
                qs = random.sample(qs, max_questions)
            return qs

        return []

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

    def _run_tests(
        self,
        questions: List[Dict],
        retriever: DocumentRetriever,
        ollama_client: OllamaClient
    ) -> List[Dict]:
        print(f"\n[TESTS] Начало тестирования ({len(questions)} вопросов)...")
        results: List[Dict] = []

        for i, qd in enumerate(questions, 1):
            q = (qd.get("question") or "").strip()
            expected = (qd.get("answer") or "").strip()

            if not q:
                print(f"\n[{i}/{len(questions)}] [SKIP] Пустой question в записи: {qd}")
                continue

            print(f"\n[{i}/{len(questions)}] {q[:80]}...")

            t0 = time.time()
            chunks = retriever.retrieve_with_scores(q, return_hyde_info=self.use_hyde, top_k=self.top_k)

            best_chunk = None
            best_score = 0.0
            if chunks:
                best_chunk = max(chunks, key=lambda c: c.get("score", 0.0))
                best_score = float(best_chunk.get("score", 0.0))

            ctx = [c.get("text", "") for c in chunks if c.get("text")]

            ans = ollama_client.generate(q, ctx)
            dt = time.time() - t0

            sim = calculate_similarity(ans, expected) if expected else 0.0
            ok = sim >= self.threshold if expected else True

            print(f"  Время: {dt:.2f}s")
            print(f"  Similarity: {sim:.1%}")
            print(f"  {'Правильно' if ok else 'Неправильно'}")

            results.append({
                "question": q,
                "expected_answer": expected,
                "generated_answer": ans,
                "similarity": sim,
                "is_correct": ok,
                "retrieved_chunks": chunks,
                "response_time": dt,
                "best_chunk": best_chunk,
                "best_chunk_score": best_score,
                "best_chunk_source": (best_chunk.get("source", "unknown") if best_chunk else "unknown"),
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
            "results": results,
        }

    def _print_final_stats(self, stats: Dict):
        print("\n" + "=" * 80)
        print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print("=" * 80)
        print(f"Правильных ответов: {stats['correct_count']}/{stats['total_count']} ({stats['accuracy']:.1f}%)")
        print(f"Средняя схожесть: {stats['avg_similarity']:.1%}")
        print(f"Отчет: {stats['report_path']}")
        print("=" * 80)
