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
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, url: str, index_name: str):
        if getattr(self, "_initialized", False):
            return

        self.url = url
        self.index_name = index_name

        self.es = Elasticsearch(
            [url],
            verify_certs=False,
            request_timeout=30,
        )

        self._initialized = True
        print(f"[ES] Подключен: {url} (index: {index_name})")

    def reinit(self, url: str, index_name: str) -> "ElasticsearchClient":
        """
        Переинициализировать singleton (если хочешь сменить url/index без пересоздания объекта).
        """
        self.url = url
        self.index_name = index_name
        self.es = Elasticsearch([url], verify_certs=False, request_timeout=30)
        print(f"[ES] Re-init: {url} (index: {index_name})")
        return self

    def get_langchain_store(
        self,
        embedding_model,
        text_field: str = "content",
        vector_field: str = "embedding",
    ) -> ElasticsearchStore:
        """
        Получить ElasticsearchStore для langchain

        ВАЖНО:
        - НЕ передаем metadata_field, потому что твоя версия ElasticsearchStore его не поддерживает.
        - Метаданные достаем через doc_builder в retriever.py.
        """
        return ElasticsearchStore(
            es_url=self.url,
            index_name=self.index_name,
            embedding=embedding_model,
            query_field=text_field,
            vector_query_field=vector_field,
        )

    # -------------------------------------------------------------------------
    # БАЗОВЫЕ МЕТОДЫ (они нужны для "ES check" в main.py/cli)
    # -------------------------------------------------------------------------

    def ping(self) -> bool:
        try:
            return bool(self.es.ping())
        except Exception as e:
            print(f"[ERROR] Ошибка ping: {e}")
            return False

    def index_exists(self) -> bool:
        try:
            return bool(self.es.indices.exists(index=self.index_name))
        except Exception as e:
            print(f"[ERROR] Ошибка проверки индекса: {e}")
            return False

    def get_document_count(self) -> int:
        """
        ВАЖНО: этот метод у тебя явно вызывается при проверке ES.
        """
        try:
            res = self.es.count(index=self.index_name)
            return int(res.get("count", 0))
        except Exception as e:
            print(f"[ERROR] Ошибка подсчета документов: {e}")
            return 0

    def delete_index(self) -> bool:
        try:
            if self.index_exists():
                self.es.indices.delete(index=self.index_name)
                print(f"[ES] Индекс {self.index_name} удален")
                return True
            return False
        except Exception as e:
            print(f"[ERROR] Ошибка удаления индекса: {e}")
            return False

    # -------------------------------------------------------------------------
    # ВЕКТОРНЫЙ ИНДЕКС + ЗАГРУЗКА
    # -------------------------------------------------------------------------

    def create_index_with_vectors(self, dims: int) -> None:
        """
        Создание индекса с поддержкой dense_vector

        ВАЖНО:
        - metadata хранится как вложенный объект metadata.*
        """
        if self.index_exists():
            print(f"[ES] Индекс '{self.index_name}' существует. Удаляем...")
            self.delete_index()

        body = {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "standard"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "metadata": {
                        "properties": {
                            "filename": {"type": "keyword"},
                            "chunk_id": {"type": "integer"},
                            "total_chunks": {"type": "integer"},
                        }
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
        dims: int = 1024,
    ) -> Dict[str, int]:
        """
        Загрузка документов с векторами.

        ВАЖНО:
        - filename/chunk_id/total_chunks кладем внутрь metadata (вложенный объект)
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
            except Exception as e:
                print(f"   [SKIP] Ошибка чтения: {e}")
                continue

            char_count = len(content)
            total_chars += char_count

            chunks = self.split_into_chunks(content, chunk_size, overlap)
            chunk_count = len(chunks)

            print(f"   Символов: {char_count}")
            print(f"   Chunks: {chunk_count}")

            if chunk_count == 0:
                print("   [SKIP] Пустой файл")
                continue

            print("   [EMB] Вычисление векторов...")
            chunk_embeddings = embedding_model.encode(chunks, show_progress_bar=True)

            if len(chunk_embeddings[0]) != dims:
                print(f"   [WARNING] Размерность {len(chunk_embeddings[0])} != {dims}")

            print("   [ES] Загрузка...")
            for i, (chunk_text, embedding) in enumerate(zip(chunks, chunk_embeddings), 1):
                doc = {
                    "content": chunk_text,
                    "embedding": embedding.tolist() if hasattr(embedding, "tolist") else list(embedding),
                    "metadata": {
                        "filename": file_path.name,
                        "chunk_id": i,
                        "total_chunks": chunk_count,
                    },
                }
                self.es.index(index=self.index_name, document=doc)

            total_docs += 1
            total_chunks += chunk_count
            print(f"   [OK] Загружено {chunk_count} chunks")

        self.es.indices.refresh(index=self.index_name)

        print("\n============================================================")
        print("ИТОГО:")
        print(f"   Документов: {total_docs}")
        print(f"   Chunks: {total_chunks}")
        print(f"   Символов: {total_chars}")
        print("============================================================")

        return {"total_docs": total_docs, "total_chunks": total_chunks, "total_chars": total_chars}

    def verify_index(self) -> None:
        """
        Проверка индекса после загрузки
        """
        self.es.indices.refresh(index=self.index_name)

        count = self.get_document_count()
        stats = self.es.indices.stats(index=self.index_name)
        size_bytes = stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"]
        size_mb = size_bytes / (1024 * 1024)

        print(f"\n[VERIFY] Индекс '{self.index_name}':")
        print(f"   Документов: {count}")
        print(f"   Размер: {size_mb:.2f} MB")

        result = self.es.search(index=self.index_name, body={"size": 1, "query": {"match_all": {}}})
        if result.get("hits", {}).get("hits"):
            doc = result["hits"]["hits"][0].get("_source", {})
            md = doc.get("metadata", {}) or {}
            print("\n   Пример chunk:")
            print(f"   Файл: {md.get('filename')}")
            print(f"   Chunk: {md.get('chunk_id')}/{md.get('total_chunks')}")
            print(f"   Текст: {(doc.get('content', '') or '')[:120]}...")

    def find_documents(self, docs_path: str = "data/documents") -> List[Path]:
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

    def split_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
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

    def close(self):
        try:
            self.es.close()
        except Exception:
            pass

    def __repr__(self):
        return f"ElasticsearchClient(url='{self.url}', index='{self.index_name}')"
