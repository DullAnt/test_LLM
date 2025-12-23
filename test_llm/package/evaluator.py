"""
Основная логика тестирования RAG системы
"""

import time
import random
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from ..rag.ollama_client import OllamaClient
from ..rag.embeddings import EmbeddingModel
from ..rag.retriever import DocumentRetriever
from ..rag.hyde import HyDEGenerator
from ..evaluate.questions import load_questions, extract_questions
from ..evaluate.similarity import calculate_similarity
from ..evaluate.metrics import generate_html_report

try:
    from package.ollama_detector import get_ollama_host_with_fallback
    HAS_OLLAMA_DETECTOR = True
except ImportError:
    HAS_OLLAMA_DETECTOR = False

class RAGEvaluator:
    """Класс для тестирования RAG системы"""
    
    def __init__(
        self,
        model: str,
        ollama_host: str,
        timeout: int,
        top_k: int,
        threshold: float,
        use_hyde: bool,
        random_seed: Optional[int] = None
    ):
        self.model = model
        self.ollama_host = ollama_host
        self.timeout = timeout
        self.top_k = top_k
        self.threshold = threshold
        self.use_hyde = use_hyde
        self.random_seed = random_seed
        
        # Установить seed
        if random_seed is not None:
            random.seed(random_seed)
            print(f"[RANDOM] Seed установлен: {random_seed}\n")
    
    def run_evaluation(
        self,
        documents: List[Dict],
        questions_path: Optional[str],
        max_questions: int,
        extract_qa: bool,
        es_client=None,
        es_index: Optional[str] = None
    ) -> Optional[Dict]:
        """Запуск полного цикла оценки"""
        
        # Загрузка или извлечение вопросов
        questions = self._load_questions(
            documents=documents,
            questions_path=questions_path,
            max_questions=max_questions,
            extract_qa=extract_qa
        )
        
        if not questions:
            print("[ERROR] Нет вопросов для тестирования")
            return None
        
        # Инициализация компонентов
        print("\n[INIT] Инициализация компонентов...")
        
        embedding_model = EmbeddingModel()
        print(" Embedding модель")
        
        ollama_client = OllamaClient(
            host=self.ollama_host,
            model=self.model,
            timeout=self.timeout
        )
        
        if not ollama_client.check_connection():
            print("[ERROR] Не удалось подключиться к Ollama!")
            return None
        
        print(f"  Ollama клиент ({self.model})")
        
        hyde_generator = None
        if self.use_hyde:
            hyde_generator = HyDEGenerator(ollama_client)
            print(" HyDE генератор")
        
        retriever = DocumentRetriever(
            embedding_model=embedding_model,
            top_k=self.top_k,
            es_client=es_client,
            es_index=es_index,
            documents=documents
        )
        print("  Document retriever")
        
        print("[SUCCESS] Компоненты инициализированы\n")
        
        # Тестирование
        results = self._run_tests(questions, retriever, ollama_client)
        
        # Генерация отчета
        report_path = self._generate_report(results)
        
        # Статистика
        stats = self._calculate_stats(results)
        stats['report_path'] = report_path
        
        self._print_final_stats(stats)
        
        return stats
    
    def _load_questions(
        self,
        documents: List[Dict],
        questions_path: Optional[str],
        max_questions: int,
        extract_qa: bool
    ) -> Optional[List[Dict]]:
        """Загрузка или извлечение вопросов"""
        
        if questions_path:
            print(f"\n[QUESTIONS] Загрузка вопросов из {questions_path}...")
            questions = load_questions(questions_path)
            
            if len(questions) > max_questions:
                print(f"[INFO] Загружено {len(questions)} вопросов")
                print(f"[RANDOM] Случайный выбор {max_questions} вопросов")
                questions = random.sample(questions, max_questions)
            
            self._print_selected_questions(questions)
            return questions
        
        elif extract_qa:
            print("\n[SEARCH] Автоматическое извлечение вопросов...")
            questions = extract_questions(documents)
            
            if questions and len(questions) > max_questions:
                print(f"[INFO] Извлечено {len(questions)} вопросов")
                print(f"[RANDOM] Случайный выбор {max_questions} вопросов")
                questions = random.sample(questions, max_questions)
            
            if questions:
                self._save_questions(questions)
                self._print_selected_questions(questions)
                return questions
            else:
                print("[WARNING] Вопросы не найдены в документах")
                return None
        
        return None
    
    def _save_questions(self, questions: List[Dict]):
        """Сохранение извлеченных вопросов"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"data/testsets/questions_extracted_{timestamp}.jsonl"
        
        with open(save_path, "w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        
        print(f"[SAVED] Сохранено в: {save_path}")
    
    def _print_selected_questions(self, questions: List[Dict]):
        """Вывод выбранных вопросов"""
        print(f"\n[INFO] Выбранные вопросы ({len(questions)}):")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q['question'][:70]}...")
    
    def _run_tests(
        self,
        questions: List[Dict],
        retriever: DocumentRetriever,
        ollama_client: OllamaClient
    ) -> List[Dict]:
        """Запуск тестов"""
        
        print("\n[TEST] Начало тестирования...")
        print(f"Вопросов: {len(questions)}")
        
        hyde_time = " + HyDE" if self.use_hyde else ""
        est_time = len(questions) * (3 if self.use_hyde else 2)
        print(f"Примерное время: ~{est_time:.1f} минут ({self.model}{hyde_time})\n")
        
        results = []
        
        for i, question_data in enumerate(questions, 1):
            question = question_data["question"]
            expected_answer = question_data.get("answer", "")
            
            print(f"[{i}/{len(questions)}] {question[:80]}...")
            
            try:
                start_time = time.time()
                
                # Поиск chunks
                retrieved_chunks = retriever.retrieve_with_scores(
                    question,
                    return_hyde_info=self.use_hyde
                )
                
                # DEBUG вывод
                print(f"  [RAG] Найдено {len(retrieved_chunks)} chunks:")
                for chunk in retrieved_chunks:
                    print(f"    #{chunk['rank']}: {chunk['source']} (score: {chunk['score']:.2%})")
                
                if retrieved_chunks:
                    top_chunk = retrieved_chunks[0]['text'][:120]
                    print(f"  [DEBUG] Top-1: {top_chunk}...")
                
                # Генерация ответа
                relevant_texts = [chunk['text'] for chunk in retrieved_chunks]
                answer = ollama_client.generate(question, relevant_texts)
                
                elapsed = time.time() - start_time
                print(f"  [LLM] Ответ ({elapsed:.1f}s): {answer[:80]}...")
                
                # Оценка
                similarity = calculate_similarity(answer, expected_answer, retriever.embedding_model.model)
                is_correct = similarity >= self.threshold
                
                status = "[OK]" if is_correct else "[FAIL]"
                print(f"  {status} Схожесть: {similarity:.2%}\n")
                
                results.append({
                    "question": question,
                    "expected_answer": expected_answer,
                    "generated_answer": answer,
                    "similarity": similarity,
                    "is_correct": is_correct,
                    "retrieved_chunks": retrieved_chunks,
                    "response_time": elapsed
                })
                
            except Exception as e:
                print(f"  [ERROR] {e}\n")
                results.append({
                    "question": question,
                    "expected_answer": expected_answer,
                    "generated_answer": f"ERROR: {str(e)}",
                    "similarity": 0.0,
                    "is_correct": False,
                    "retrieved_chunks": [],
                    "response_time": 0
                })
        
        return results
    
    def _generate_report(self, results: List[Dict]) -> str:
        """Генерация HTML отчета"""
        print("\n[REPORT] Генерация HTML отчета...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{'hyde_' if self.use_hyde else ''}{timestamp}.html"
        report_path = f"data/reports/{report_filename}"
        
        model_display = self.model
        if self.use_hyde:
            model_display += " + HyDE"
        
        generate_html_report(
            results=results,
            output_path=report_path,
            threshold=self.threshold,
            model_name=model_display,
            top_k=self.top_k
        )
        
        print(f"[SUCCESS] Отчет сохранен: {report_path}")
        return report_path
    
    def _calculate_stats(self, results: List[Dict]) -> Dict:
        """Расчет статистики"""
        total_count = len(results)
        correct_count = sum(1 for r in results if r["is_correct"])
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        avg_similarity = sum(r["similarity"] for r in results) / total_count if total_count > 0 else 0
        
        all_chunks = []
        for r in results:
            all_chunks.extend(r.get('retrieved_chunks', []))
        
        avg_chunk_score = sum(c.get('score', 0) for c in all_chunks) / len(all_chunks) if all_chunks else 0
        
        return {
            "accuracy": accuracy,
            "avg_similarity": avg_similarity,
            "avg_chunk_score": avg_chunk_score,
            "correct_count": correct_count,
            "total_count": total_count,
            "results": results
        }
    
    def _print_final_stats(self, stats: Dict):
        """Вывод итоговой статистики"""
        print("\n" + "=" * 80)
        print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print("=" * 80)
        print(f"Правильных ответов: {stats['correct_count']}/{stats['total_count']} ({stats['accuracy']:.1f}%)")
        print(f"Средняя схожесть: {stats['avg_similarity']:.1%}")
        print(f"Качество RAG: {stats['avg_chunk_score']:.1%}")
        if self.use_hyde:
            print("HyDE: ВКЛЮЧЕН")
        print(f"Отчет: {stats['report_path']}")
        print("=" * 80)