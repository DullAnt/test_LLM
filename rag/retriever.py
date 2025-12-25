"""
Модуль для векторного поиска релевантных документов
"""

from typing import List, Dict, Optional
from elasticsearch import Elasticsearch
import numpy as np
import requests
from package.config import get_embedding_model, EmbeddingModel, DEFAULT_TOP_K 
from rag.ollama_client import OllamaClient

class DocumentRetriever:
    """Класс для векторного поиска документов"""
    
    def __init__(
            self,
            embedding_model: EmbeddingModel,
            es_client: Optional[Elasticsearch] = None,
            index_name: str = "psb_docs",
            top_k: int = DEFAULT_TOP_K,
            ollama_client: Optional[OllamaClient] = None
        ):
            self.embedding_model = embedding_model
            self.es_client = es_client
            self.index_name = index_name
            self.top_k = top_k
            self.ollama_client = ollama_client

            # ES URL для прямых HTTP запросов
            if es_client:
                # Получаем хост из клиента
                self.es_url = "http://localhost:9200"
            else:
                self.es_url = None
    
    def retrieve_with_scores(
        self, 
        question: str, 
        return_hyde_info: bool = False,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Поиск релевантных chunks с метаданными
        
        Args:
            question: Вопрос для поиска
            return_hyde_info: Вернуть информацию о HyDE (не используется пока)
            top_k: Количество результатов (если None - использует self.top_k)
            
        Returns:
            Список chunks с score и метаданными
        """
        # Используем переданный top_k или self.top_k
        k = top_k if top_k is not None else self.top_k
        
        if return_hyde_info:
            # 1. Генерация гипотезы
            print(f"[HYDE] Генерация гипотезы для: {question[:80]}...")
            hypothesis = self._generate_hypothesis(question)
            print(f"[HYDE] Гипотеза: {hypothesis[:100]}...")
            return self.search(hypothesis, top_k=k)

        # Вызываем основной метод поиска
        return self.search(question, top_k=k)
            
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Поиск релевантных chunks через векторное сходство
        
        Args:
            query: Вопрос для поиска
            top_k: Количество результатов (если None - использует self.top_k)
            
        Returns:
            Список chunks с косинусным сходством
        """
        # Используем переданный top_k или self.top_k
        k = top_k if top_k is not None else self.top_k
        
        if self.es_client:
            return self._search_elasticsearch_knn(query, k)
        else:
            return self._search_local(query, k)
    
    def _search_elasticsearch_knn(self, query: str, top_k: int) -> List[Dict]:
        """
        kNN поиск в Elasticsearch (векторный поиск)
        Использует прямой HTTP запрос для совместимости со всеми версиями
        """
        # Вычисляем вектор вопроса
        query_vector = self.embedding_model.encode(query).tolist()
        
        # DEBUG
        print(f"\n{'='*60}")
        print(f"[DEBUG kNN SEARCH]")
        print(f"Вопрос: {query[:80]}")
        print(f"Вектор длина: {len(query_vector)}")
        print(f"Первые 5 чисел: {query_vector[:5]}")
        print(f"TOP-K запрошено: {top_k}")
        print(f"Index: {self.index_name}")
        print(f"{'='*60}")
        
        try:
            print(f"[DEBUG] Отправка kNN запроса...")
            
            # ПРЯМОЙ HTTP ЗАПРОС (работает с любой версией ES клиента)
            url = f"{self.es_url}/{self.index_name}/_search"
            
            body = {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": 100
                },
                "_source": ["content", "filename", "chunk_id", "total_chunks"]
            }
            
            response = requests.post(url, json=body, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            data = response.json()
            
            hits_count = len(data['hits']['hits'])
            print(f"[DEBUG] Получено hits: {hits_count}")
            
            if hits_count == 0:
                print(f"[WARNING] Ничего не найдено!")
                print(f"[DEBUG] Проверьте что вектора загружены: curl http://localhost:9200/{self.index_name}/_search?size=1")
                return []
            
            results = []
            for rank, hit in enumerate(data['hits']['hits'], 1):
                source = hit['_source']
                score = hit['_score']
                
                print(f"\n[DEBUG] Chunk #{rank}:")
                print(f"  Score: {score:.4f}")
                print(f"  File: {source['filename']}")
                print(f"  Chunk: {source.get('chunk_id', '?')}/{source.get('total_chunks', '?')}")
                print(f"  Text preview: {source['content'][:100]}...")
                
                results.append({
                    'text': source['content'],
                    'score': score,
                    'rank': rank,
                    'source': source['filename'],
                    'chunk_id': source.get('chunk_id', 0),
                    'total_chunks': source.get('total_chunks', 0)
                })
            
            print(f"\n[DEBUG] Возвращаем {len(results)} chunks")
            print(f"{'='*60}\n")
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"\n[ERROR kNN] HTTP ошибка: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"[ERROR] Response: {e.response.text}")
            return []
        except Exception as e:
            print(f"\n[ERROR kNN] Ошибка при поиске: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _generate_hypothesis(self, question: str) -> str:
        """
        Генерация гипотетического ответа для HyDE
        
        Args:
            question: Вопрос пользователя
            
        Returns:
            Гипотетический ответ (используется вместо вопроса для поиска)
        """
        if not self.ollama_client:
            print(f"[HYDE] Ollama client не инициализирован, используем вопрос")
            return question
        
        # Промпт для генерации гипотезы
        hyde_prompt = f"""Ты - эксперт банка ПСБ. Ответь кратко и точно на вопрос (2-3 предложения).
            НЕ объясняй, просто дай прямой ответ как в официальном документе.

            Вопрос: {question}

            Ответ:"""
            
        try:
            print(f"[HYDE] Вызов LLM для генерации гипотезы...")
            
            # Генерация через Ollama
            hypothesis = self.ollama_client.generate(
                question=hyde_prompt,  
                context=[]  
                )
            
            
            # Очистка ответа
            hypothesis = hypothesis.strip()
            
            # Ограничение длины (макс 500 символов)
            if len(hypothesis) > 500:
                hypothesis = hypothesis[:500]
            
            print(f"[HYDE] Гипотеза: {hypothesis[:100]}...")
            return hypothesis
            
        except Exception as e:
            print(f"[HYDE] ⚠️  Ошибка генерации: {e}")
            print(f"[HYDE] Используем оригинальный вопрос")
            return question

    def _search_local(self, query: str, top_k: int) -> List[Dict]:
            """
            Локальный поиск (без Elasticsearch)
            Для совместимости с --local-files
            """
            if not hasattr(self, 'local_chunks'):
                return []
            
            # Вектор вопроса
            query_embedding = self.embedding_model.encode(query)
            
            # Вычисляем косинусное сходство со всеми chunks
            from sklearn.metrics.pairwise import cosine_similarity
            
            results = []
            for chunk in self.local_chunks:
                chunk_embedding = self.embedding_model.encode(chunk['text'])
                
                # Косинусное сходство
                score = cosine_similarity(
                    [query_embedding],
                    [chunk_embedding]
                )[0][0]
                
                results.append({
                    'text': chunk['text'],
                    'score': float(score),
                    'source': chunk['source'],
                    'rank': 0
                })
            
            # Сортировка по score
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # TOP-K
            for i, result in enumerate(results[:top_k], 1):
                result['rank'] = i
            
            return results[:top_k]
    
    def load_local_files(self, docs_path: str = "data/documents"):
        """
        Загрузка локальных файлов для поиска без Elasticsearch
        """
        from pathlib import Path
        
        docs_dir = Path(docs_path)
        if not docs_dir.exists():
            raise FileNotFoundError(f"Папка {docs_path} не найдена")
        
        files = list(docs_dir.glob("*.txt")) + list(docs_dir.glob("*.md"))
        files = [f for f in files if f.name.lower() != "readme.md"]
        
        self.local_chunks = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Разбиваем на chunks
                chunks = self._split_text(content)
                
                for chunk in chunks:
                    self.local_chunks.append({
                        'text': chunk,
                        'source': file_path.name
                    })
            except Exception as e:
                print(f"Ошибка при чтении {file_path.name}: {e}")
        
        print(f"Загружено {len(self.local_chunks)} chunks из локальных файлов")
    
    def _split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Разбивка текста на chunks"""
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