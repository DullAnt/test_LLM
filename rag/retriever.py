"""
Document Retriever - поиск релевантных документов
"""

from typing import List, Dict, Optional
from rag.embeddings import EmbeddingModel
from package.elastic import ElasticsearchClient
import numpy as np


class DocumentRetriever:
    """Поиск релевантных документов"""
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        top_k: int = 3,
        es_client: Optional[ElasticsearchClient] = None,
        es_index: Optional[str] = None,
        documents: Optional[List[Dict]] = None
    ):
        """
        Args:
            embedding_model: Модель для векторизации
            top_k: Количество результатов
            es_client: Клиент Elasticsearch (опционально)
            es_index: Название индекса (опционально)
            documents: Локальные документы (опционально)
        """
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.es_client = es_client
        self.es_index = es_index
        self.documents = documents
        
        # Определить режим работы
        self.use_elasticsearch = es_client is not None and es_index is not None
        
        # Инициализация
        if self.use_elasticsearch:
            self._init_elasticsearch()
        else:
            self._init_local()
    
    def _init_elasticsearch(self):
        """Инициализация Elasticsearch retriever"""
        try:
            # Проверка подключения
            if not self.es_client.ping():
                raise Exception("Не удалось подключиться к Elasticsearch")
            
            # Проверка индекса
            if not self.es_client.index_exists():
                raise Exception(f"Индекс '{self.es_index}' не существует")
            
            # Получение количества документов
            doc_count = self.es_client.get_document_count()
            
            if doc_count == 0:
                raise Exception(f"Индекс '{self.es_index}' пустой")
            
            print(f"[RETRIEVER] Режим: Elasticsearch (индекс: {self.es_index})")
            print(f"[RETRIEVER] Документов: {doc_count}")
            
        except Exception as e:
            print(f"[ERROR] Ошибка инициализации Elasticsearch: {e}")
            raise
    
    def _init_local(self):
        """Инициализация локального retriever"""
        if not self.documents:
            raise ValueError("Документы не предоставлены для локального режима")
        
        print(f"[RETRIEVER] Режим: Локальные документы")
        print(f"[RETRIEVER] Документов: {len(self.documents)}")
        
        # Разбить документы на chunks
        self.chunks = []
        for doc in self.documents:
            doc_chunks = self._split_into_chunks(doc['content'])
            for chunk in doc_chunks:
                self.chunks.append({
                    'text': chunk,
                    'filename': doc.get('filename', 'unknown'),
                    'source_doc': doc
                })
        
        print(f"[RETRIEVER] Chunks: {len(self.chunks)}")
        
        # Векторизация всех chunks
        print(f"[RETRIEVER] Векторизация chunks...")
        chunk_texts = [c['text'] for c in self.chunks]
        self.chunk_embeddings = self.embedding_model.encode_batch(chunk_texts)
        print(f"[RETRIEVER] ✅ Векторизация завершена")
    
    def _split_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Разбить текст на chunks
        
        Args:
            text: Исходный текст
            chunk_size: Размер chunk в символах
            overlap: Перекрытие между chunks
            
        Returns:
            List[str]: Список chunks
        """
        # Разбить по параграфам
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Если параграф + текущий chunk меньше лимита
            if len(current_chunk) + len(paragraph) <= chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                # Сохранить текущий chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Начать новый chunk
                current_chunk = paragraph + "\n\n"
        
        # Добавить последний chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def retrieve(self, query: str) -> List[str]:
        """
        Поиск релевантных chunks
        
        Args:
            query: Поисковый запрос
            
        Returns:
            List[str]: Список релевантных chunks
        """
        if self.use_elasticsearch:
            return self._retrieve_elasticsearch(query)
        else:
            return self._retrieve_local(query)
    
    def _retrieve_elasticsearch(self, query: str) -> List[str]:
        """Поиск в Elasticsearch"""
        try:
            # Поиск через Elasticsearch
            response = self.es_client.es.search(
                index=self.es_index,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["content", "chunks"]
                        }
                    },
                    "size": self.top_k * 2
                }
            )
            
            # Извлечь chunks из результатов
            chunks = []
            for hit in response['hits']['hits']:
                doc = hit['_source']
                
                # Если есть chunks - взять их
                if 'chunks' in doc and doc['chunks']:
                    if isinstance(doc['chunks'], list):
                        chunks.extend(doc['chunks'][:self.top_k])
                    else:
                        chunks.append(doc['chunks'])
                else:
                    # Иначе разбить content на chunks
                    doc_chunks = self._split_into_chunks(doc['content'])
                    chunks.extend(doc_chunks[:2])
            
            # Ограничить результат
            chunks = chunks[:self.top_k]
            
            return chunks
            
        except Exception as e:
            print(f"[ERROR] Ошибка поиска в Elasticsearch: {e}")
            return []
    
    def _retrieve_local(self, query: str) -> List[str]:
        """Поиск в локальных документах"""
        # Векторизация запроса
        query_embedding = self.embedding_model.encode(query)
        
        # Вычислить similarity для всех chunks
        similarities = []
        for chunk_embedding in self.chunk_embeddings:
            # Косинусное сходство
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append(similarity)
        
        # Отсортировать по similarity
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        
        # Извлечь top chunks
        top_chunks = [self.chunks[i]['text'] for i in top_indices]
        
        return top_chunks
    
    def retrieve_with_scores(self, query: str, return_hyde_info: bool = False) -> List[Dict]:
        """
        Поиск релевантных chunks с scores
        
        Args:
            query: Поисковый запрос
            return_hyde_info: Вернуть информацию о HyDE (не используется)
            
        Returns:
            List[Dict]: Список словарей с 'text', 'score', 'rank', 'source'
        """
        if self.use_elasticsearch:
            return self._retrieve_elasticsearch_with_scores(query)
        else:
            return self._retrieve_local_with_scores(query)
    
    def _retrieve_elasticsearch_with_scores(self, query: str) -> List[Dict]:
        """Поиск в Elasticsearch с scores"""
        try:
            # Поиск через Elasticsearch
            response = self.es_client.es.search(
                index=self.es_index,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["content", "chunks"]
                        }
                    },
                    "size": self.top_k * 2
                }
            )
            
            # Извлечь chunks с scores
            results = []
            for hit in response['hits']['hits']:
                doc = hit['_source']
                score = hit['_score']
                
                # Нормализовать score (ES scores обычно 0-10)
                normalized_score = min(score / 10.0, 1.0)
                
                # Если есть chunks - взять их
                if 'chunks' in doc and doc['chunks']:
                    if isinstance(doc['chunks'], list):
                        for chunk in doc['chunks'][:self.top_k]:
                            results.append({
                                'text': chunk,
                                'score': normalized_score,
                                'rank': len(results) + 1,
                                'source': doc.get('filename', 'unknown')
                            })
                    else:
                        results.append({
                            'text': doc['chunks'],
                            'score': normalized_score,
                            'rank': len(results) + 1,
                            'source': doc.get('filename', 'unknown')
                        })
                else:
                    # Иначе разбить content на chunks
                    doc_chunks = self._split_into_chunks(doc['content'])
                    for chunk in doc_chunks[:2]:
                        results.append({
                            'text': chunk,
                            'score': normalized_score,
                            'rank': len(results) + 1,
                            'source': doc.get('filename', 'unknown')
                        })
            
            # Ограничить результат
            results = results[:self.top_k]
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Ошибка поиска в Elasticsearch: {e}")
            return []
    
    def _retrieve_local_with_scores(self, query: str) -> List[Dict]:
        """Поиск в локальных документах с scores"""
        # Векторизация запроса
        query_embedding = self.embedding_model.encode(query)
        
        # Вычислить similarity для всех chunks
        similarities = []
        for chunk_embedding in self.chunk_embeddings:
            # Косинусное сходство
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append(float(similarity))
        
        # Отсортировать по similarity
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        
        # Извлечь top chunks с scores
        results = []
        for rank, idx in enumerate(top_indices, 1):
            chunk_text = self.chunks[idx]['text']
            score = similarities[idx]
            source = self.chunks[idx].get('filename', 'unknown')
            
            results.append({
                'text': chunk_text,
                'score': score,
                'rank': rank,
                'source': source
            })
        
        return results
    
    def __repr__(self):
        mode = "Elasticsearch" if self.use_elasticsearch else "Local"
        return f"DocumentRetriever(mode={mode}, top_k={self.top_k})"