"""
Единый клиент для работы с Elasticsearch (локальный Docker)
Singleton паттерн для переиспользования подключения
"""

from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchStore
from typing import List, Dict, Optional
from pathlib import Path


class ElasticsearchClient:
    """Единый клиент для работы с Elasticsearch"""
    
    _instance: Optional["ElasticsearchClient"] = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton паттерн"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, url: str, index_name: str):
        """
        Args:
            url: URL Elasticsearch (http://localhost:9200)
            index_name: Название индекса
        """
        if self._initialized:
            return
        
        self.url = url
        self.index_name = index_name
        
        # Создаем нативный Elasticsearch клиент (без аутентификации)
        self.es = Elasticsearch(
            [url],
            verify_certs=False,
            request_timeout=30
        )
        
        self._initialized = True
        print(f"[ES] Подключен: {url} (index: {index_name})")
    
    @classmethod
    def from_config(cls, config) -> "ElasticsearchClient":
        """Создать клиент из Config объекта"""
        return cls(
            url=config.ELASTIC_URL,      # ИСПРАВЛЕНО
            index_name=config.ELASTIC_INDEX,  # ИСПРАВЛЕНО
        )
    
    def get_langchain_store(self, embedding_model, text_field: str = "content", vector_field: str = "embedding") -> ElasticsearchStore:
        """
        Получить ElasticsearchStore для langchain
        
        Args:
            embedding_model: Модель embeddings
            text_field: Поле для текста
            vector_field: Поле для векторов
        
        Returns:
            ElasticsearchStore instance
        """
        return ElasticsearchStore(
            es_url=self.url,
            index_name=self.index_name,
            embedding=embedding_model,
            query_field=text_field,
            vector_query_field=vector_field,
        )
    
    # =============================================================================
    # БАЗОВЫЕ МЕТОДЫ
    # =============================================================================
    
    def ping(self) -> bool:
        """Проверка подключения"""
        try:
            return self.es.ping()
        except Exception as e:
            print(f"[ERROR] Ошибка ping: {e}")
            return False
    
    def index_exists(self) -> bool:
        """Проверка существования индекса"""
        try:
            return self.es.indices.exists(index=self.index_name)
        except Exception as e:
            print(f"[ERROR] Ошибка проверки индекса: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Получить количество документов"""
        try:
            result = self.es.count(index=self.index_name)
            return result['count']
        except Exception as e:
            print(f"[ERROR] Ошибка подсчета документов: {e}")
            return 0
    
    def delete_index(self) -> bool:
        """Удалить индекс"""
        try:
            if self.index_exists():
                self.es.indices.delete(index=self.index_name)
                print(f"[ES] Индекс {self.index_name} удален")
                return True
            return False
        except Exception as e:
            print(f"[ERROR] Ошибка удаления индекса: {e}")
            return False
    
    # =============================================================================
    # РАБОТА С ДОКУМЕНТАМИ
    # =============================================================================
    
    def get_all_documents(self) -> List[Dict]:
        """Получить все документы из индекса"""
        documents = []
        
        try:
            response = self.es.search(
                index=self.index_name,
                body={"query": {"match_all": {}}, "size": 100},
                scroll='5m'
            )
            
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
            
            for hit in hits:
                doc = hit['_source']
                doc['_id'] = hit['_id']
                documents.append(doc)
            
            while len(hits) > 0:
                response = self.es.scroll(scroll_id=scroll_id, scroll='5m')
                scroll_id = response['_scroll_id']
                hits = response['hits']['hits']
                
                for hit in hits:
                    doc = hit['_source']
                    doc['_id'] = hit['_id']
                    documents.append(doc)
            
            self.es.clear_scroll(scroll_id=scroll_id)
            print(f"[ES] Загружено документов: {len(documents)}")
            return documents
            
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки документов: {e}")
            return []
    
    # =============================================================================
    # РАБОТА С ВЕКТОРАМИ
    # =============================================================================
    
    def create_index_with_vectors(self, dims: int) -> None:
        """
        Создание индекса с поддержкой dense_vector
        
        Args:
            dims: Размерность векторов
        """
        if self.index_exists():
            print(f"[ES] Индекс '{self.index_name}' существует. Удаляем...")
            self.delete_index()
        
        body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "standard"},
                    "filename": {"type": "keyword"},
                    "chunk_id": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            },
        }
        
        self.es.indices.create(index=self.index_name, body=body)
        print(f"[ES] Индекс '{self.index_name}' создан (dense_vector dims={dims})")
    
    def load_documents_with_vectors(
        self,
        files: List[Path],
        embedding_model,
        chunk_size: int = 500,
        overlap: int = 50,
        dims: int = 384,
    ) -> Dict[str, int]:
        """
        Загрузка документов с векторами
        
        Args:
            files: Список путей к файлам
            embedding_model: Модель embeddings
            chunk_size: Размер чанка
            overlap: Перекрытие между чанками
            dims: Размерность векторов
        
        Returns:
            Статистика загрузки
        """
        total_docs = 0
        total_chunks = 0
        total_chars = 0
        
        for file_path in files:
            print(f"\n[FILE] {file_path.name}")
            
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                print("   [SKIP] Ошибка кодировки")
                continue
            
            char_count = len(content)
            total_chars += char_count
            
            chunks = self.split_into_chunks(content, chunk_size, overlap)
            chunk_count = len(chunks)
            
            print(f"   Символов: {char_count:,}")
            print(f"   Chunks: {chunk_count}")
            
            if chunk_count == 0:
                print("   [SKIP] Пустой файл")
                continue
            
            print("   [EMB] Вычисление векторов...")
            chunk_embeddings = embedding_model.encode(chunks, show_progress_bar=True)
            
            # Проверка размерности
            vec0_len = len(chunk_embeddings[0])
            if vec0_len != dims:
                print(f"   [WARNING] Размерность {vec0_len} != {dims}")
            
            print("   [ES] Загрузка...")
            for i, (chunk_text, embedding) in enumerate(zip(chunks, chunk_embeddings), 1):
                doc = {
                    "content": chunk_text,
                    "filename": file_path.name,
                    "chunk_id": i,
                    "total_chunks": chunk_count,
                    "embedding": embedding.tolist() if hasattr(embedding, "tolist") else list(embedding),
                }
                self.es.index(index=self.index_name, document=doc)
            
            total_docs += 1
            total_chunks += chunk_count
            print(f"   [OK] Загружено {chunk_count} chunks")
        
        print(f"\n{'=' * 60}")
        print("ИТОГО:")
        print(f"   Документов: {total_docs}")
        print(f"   Chunks: {total_chunks}")
        print(f"   Символов: {total_chars:,}")
        print("=" * 60)
        
        return {
            "total_docs": total_docs,
            "total_chunks": total_chunks,
            "total_chars": total_chars,
        }
    
    def verify_index(self) -> None:
        """Проверка индекса после загрузки"""
        self.es.indices.refresh(index=self.index_name)
        
        count = self.es.count(index=self.index_name)["count"]
        stats = self.es.indices.stats(index=self.index_name)
        size_bytes = stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"]
        size_mb = size_bytes / (1024 * 1024)
        
        print(f"\n[VERIFY] Индекс '{self.index_name}':")
        print(f"   Документов: {count}")
        print(f"   Размер: {size_mb:.2f} MB")
        
        result = self.es.search(
            index=self.index_name,
            body={"size": 1, "query": {"match_all": {}}}
        )
        
        if result["hits"]["hits"]:
            doc = result["hits"]["hits"][0]["_source"]
            print("\n   Пример chunk:")
            print(f"   Файл: {doc.get('filename')}")
            print(f"   Chunk: {doc.get('chunk_id')}/{doc.get('total_chunks')}")
            print(f"   Текст: {doc.get('content', '')[:100]}...")
    
    @staticmethod
    def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Разбивка текста на chunks с перекрытием"""
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
    
    @staticmethod
    def find_documents(docs_path: str = "data/documents") -> List[Path]:
        """Поиск всех .txt и .md файлов"""
        docs_dir = Path(docs_path)
        
        if not docs_dir.exists():
            raise FileNotFoundError(f"Папка {docs_path} не найдена")
        
        files = list(docs_dir.glob("*.txt")) + list(docs_dir.glob("*.md"))
        files = [f for f in files if f.name.lower() != "readme.md"]
        
        if not files:
            raise FileNotFoundError(f"Нет файлов в {docs_path}")
        
        print(f"[DOCS] Найдено: {len(files)} файлов")
        for f in files:
            print(f"   - {f.name}")
        
        return files
    
    def close(self):
        """Закрыть соединение"""
        try:
            self.es.close()
        except:
            pass
    
    def __repr__(self):
        return f"ElasticsearchClient(url='{self.url}', index='{self.index_name}')"