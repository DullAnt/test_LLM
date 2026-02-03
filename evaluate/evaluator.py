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
            print(f"[RANDOM] Seed: {random_seed}\n")

    def _robust_load_questions_file(self, path: str) -> List[Dict]:
        p = Path(path)
        if not p.exists():
            print(f"[ERROR] File not found: {path}")
            return []
        raw = p.read_text(encoding="utf-8-sig", errors="replace").strip()
        
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and isinstance(data.get("questions"), list):
                return data["questions"]
            return []
        except json.JSONDecodeError:
            out = []
            for ln in raw.splitlines():
                if ln.strip():
                    try:
                        out.append(json.loads(ln))
                    except:
                        pass
            return out

    def _normalize_questions(self, questions: List[Dict]) -> List[Dict]:
        """
        Normalize questions and keep reference chunk for retrieval evaluation.
        """
        normalized: List[Dict] = []
        for q in questions:
            if not isinstance(q, dict):
                continue
            
            question = q.get("question") or q.get("text") or q.get("q") or ""
            answer = q.get("answer") or q.get("expected_answer") or q.get("a") or ""
            source = q.get("source") or "unknown"
            # Extract reference chunk for retrieval metric
            reference_chunk = q.get("chunk") or q.get("context") or ""
            question = str(question).strip()
            answer = str(answer).strip()
            if not question:
                continue
            item = {
                "question": question,
                "answer": answer,
                "source": source,
                "reference_chunk": reference_chunk
            }
            normalized.append(item)
        return normalized

    def _load_questions(self, documents, questions_path, max_questions, extract_qa, es_index=None) -> List[Dict]:
        if questions_path:
            print(f"\n[QUESTIONS] Loading from {questions_path}...")
            qs = self._robust_load_questions_file(questions_path)
            qs = self._normalize_questions(qs)
            if len(qs) > max_questions:
                qs = random.sample(qs, max_questions)
            return qs
        
        if not documents:
            print("\n[QUESTIONS] Extracting from Elasticsearch...")
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

        if extract_qa:
            print("\n[QUESTIONS] Extracting from local documents...")
            qs = extract_questions(documents)
            qs = self._normalize_questions(qs)
            if len(qs) > max_questions:
                qs = random.sample(qs, max_questions)
            return qs
            
        return []

    def run_evaluation(self, documents, questions_path, max_questions, extract_qa, es_index=None) -> Optional[Dict]:
        questions = self._load_questions(documents, questions_path, max_questions, extract_qa, es_index)
        if not questions:
            print("[ERROR] No questions for testing.")
            return None

        print("\n[INIT] Initializing components...")
        embedding_model = get_embedding_model()
        
        ollama_client = OllamaClient(host=self.ollama_host, model=self.model, timeout=self.timeout)
        if not ollama_client.check_connection():
            print("[ERROR] Ollama connection failed!")
            return None

        retriever = DocumentRetriever(
            embedding_model=embedding_model,
            index_name=es_index or Config.ELASTIC_INDEX,
            top_k=self.top_k,
            ollama_client=ollama_client,
            es_url=Config.ELASTIC_URL,
            es_api_key=Config.ELASTIC_API_KEY
        )
        results = self._run_tests(questions, retriever, ollama_client)
        report_path = self._generate_report(results)
        
        stats = self._calculate_stats(results)
        stats["report_path"] = report_path
        self._print_final_stats(stats)
        return stats

    def _run_tests(self, questions: List[Dict], retriever: DocumentRetriever, ollama_client: OllamaClient) -> List[Dict]:
        print(f"\n[TESTS] Starting evaluation ({len(questions)} questions)...")
        results: List[Dict] = []
        for i, qd in enumerate(questions, 1):
            q = qd.get("question", "")
            expected_ans = qd.get("answer", "")
            ref_chunk = qd.get("reference_chunk", "") 
            source_file = qd.get("source", "unknown")
            print(f"\n[{i}/{len(questions)}] {q[:80]}...")
            
            t0 = time.time()
            
            # 1. Retrieval
            # --- ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ: Безопасная обработка результата ---
            retrieval_result = retriever.retrieve_with_scores(
                q, return_hyde_info=self.use_hyde, top_k=self.top_k
            )
            
            hyde_info = None
            
            # Проверяем, вернула ли функция несколько значений
            if isinstance(retrieval_result, tuple):
                # Если да, чанки - это первый элемент
                retrieved_chunks = retrieval_result[0]
                # А информация о HyDE (если она была запрошена) - это последний элемент
                if self.use_hyde:
                    hyde_info = retrieval_result[-1]
            else:
                # Если вернулось только одно значение - это и есть наши чанки
                retrieved_chunks = retrieval_result
            
            if hyde_info and isinstance(hyde_info, dict) and hyde_info.get("hypothetical_document"):
                print(f'  [HYDE] Generated doc: {hyde_info["hypothetical_document"][:100]}...')

            # 2. Generation
            ctx = [c.get("text", "") for c in retrieved_chunks if c.get("text")]
            generated_ans = ollama_client.generate(q, ctx)
            dt = time.time() - t0

            # 3. Answer Similarity
            sim_answer = calculate_similarity(generated_ans, expected_ans) if expected_ans else 0.0
            is_correct = sim_answer >= self.threshold if expected_ans else True

            # 4. Retrieval Quality (Reference vs Retrieved)
            retrieval_quality = 0.0
            if retrieved_chunks and ref_chunk:
                similarities = [
                    calculate_similarity(retrieved.get("text", ""), ref_chunk)
                    for retrieved in retrieved_chunks
                ]
                if similarities:
                    retrieval_quality = max(similarities)
                    
            print(f"  Time: {dt:.2f}s")
            print(f"  Ans Sim: {sim_answer:.1%}, Retr Quality: {retrieval_quality:.1%}")
            
            results.append({
                "question": q,
                "expected_answer": expected_ans,
                "generated_answer": generated_ans,
                "similarity": sim_answer,
                "retrieval_quality": retrieval_quality,
                "is_correct": is_correct,
                "retrieved_chunks": retrieved_chunks,
                "response_time": dt,
                "reference_chunk": ref_chunk,
                "source_file": source_file
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
        }

    def _print_final_stats(self, stats: Dict):
        print("\n" + "=" * 80)
        print(f"Correct Answers: {stats['correct_count']}/{stats['total_count']} ({stats['accuracy']:.1f}%)")
        print(f"Avg Similarity: {stats['avg_similarity']:.1%}")
        print(f"Report: {stats['report_path']}")
        print("=" * 80)
