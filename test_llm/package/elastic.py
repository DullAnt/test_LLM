"""
Клиент для работы с Elasticsearch
"""
import os
from elasticsearch import Elasticsearch
from typing import List, Dict, Optional


class ElasticsearchClient:
    """Клиент для работы с Elasticsearch"""
    
    def __init__(self, host: str = "localhost", port: int = 9200, index_name: str = "documents"):
        """
        Args:
            host: Хост Elasticsearch
            port: Порт Elasticsearch
            index_name: Название индекса
        """
        self.host = host
        self.port = port
        self.index_name = index_name
        
        # Подключение к Elasticsearch
        # self.es = Elasticsearch(
        #     [f"http://{host}:{port}"],
        #     verify_certs=False,
        #     request_timeout=30
        # )

        self.es = Elasticsearch(
            f"https://{host}:{port}",
            ca_certs = "data/certs/ca.crt",
            basic_auth = ("elastic", os.environ["ELASTIC_PASSWORD"])
        )
        
        print(f"Elasticsearch подключен ({host}:{port})")
    
    def ping(self) -> bool:
        """
        Проверка подключения к Elasticsearch
        
        Returns:
            bool: True если подключение успешно
        """
        try:
            return self.es.ping()
        except Exception as e:
            print(f"[ERROR] Ошибка ping Elasticsearch: {e}")
            return False
    
    def index_exists(self) -> bool:
        """
        Проверка существования индекса
        
        Returns:
            bool: True если индекс существует
        """
        try:
            return self.es.indices.exists(index=self.index_name)
        except Exception as e:
            print(f"[ERROR] Ошибка проверки индекса: {e}")
            return False
    
    def get_document_count(self) -> int:
        """
        Получить количество документов в индексе
        
        Returns:
            int: Количество документов
        """
        try:
            result = self.es.count(index=self.index_name)
            return result['count']
        except Exception as e:
            print(f"[ERROR] Ошибка подсчета документов: {e}")
            return 0
    
    def get_all_documents(self) -> List[Dict]:
        """
        Получить все документы из индекса
        
        Returns:
            List[Dict]: Список документов
        """
        documents = []
        
        try:
            # Использовать scroll API для больших объемов
            response = self.es.search(
                index=self.index_name,
                body={
                    "query": {"match_all": {}},
                    "size": 100  # Получать по 100 за раз
                },
                scroll='5m'
            )
            
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
            
            # Первая порция
            for hit in hits:
                doc = hit['_source']
                doc['_id'] = hit['_id']
                documents.append(doc)
            
            # Остальные порции
            while len(hits) > 0:
                response = self.es.scroll(scroll_id=scroll_id, scroll='5m')
                scroll_id = response['_scroll_id']
                hits = response['hits']['hits']
                
                for hit in hits:
                    doc = hit['_source']
                    doc['_id'] = hit['_id']
                    documents.append(doc)
            
            # Очистить scroll
            self.es.clear_scroll(scroll_id=scroll_id)
            
            print(f"[ES] Загружено документов: {len(documents)}")
            return documents
            
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки документов: {e}")
            return []
    
    def search(self, query: str, size: int = 10) -> List[Dict]:
        """
        Поиск документов
        
        Args:
            query: Поисковый запрос
            size: Количество результатов
            
        Returns:
            List[Dict]: Найденные документы
        """
        try:
            response = self.es.search(
                index=self.index_name,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["content", "filename"]
                        }
                    },
                    "size": size
                }
            )
            
            documents = []
            for hit in response['hits']['hits']:
                doc = hit['_source']
                doc['_id'] = hit['_id']
                doc['_score'] = hit['_score']
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"[ERROR] Ошибка поиска: {e}")
            return []
    
    def index_document(self, document: Dict, doc_id: Optional[str] = None) -> bool:
        """
        Индексировать документ
        
        Args:
            document: Документ для индексации
            doc_id: ID документа (опционально)
            
        Returns:
            bool: True если успешно
        """
        try:
            self.es.index(
                index=self.index_name,
                id=doc_id,
                body=document
            )
            return True
        except Exception as e:
            print(f"[ERROR] Ошибка индексации: {e}")
            return False
    
    def delete_index(self) -> bool:
        """
        Удалить индекс
        
        Returns:
            bool: True если успешно
        """
        try:
            if self.index_exists():
                self.es.indices.delete(index=self.index_name)
                print(f"[ES] Индекс {self.index_name} удален")
                return True
            return False
        except Exception as e:
            print(f"[ERROR] Ошибка удаления индекса: {e}")
            return False
    
    def create_index(self, settings: Optional[Dict] = None, mappings: Optional[Dict] = None) -> bool:
        """
        Создать индекс
        
        Args:
            settings: Настройки индекса
            mappings: Маппинги полей
            
        Returns:
            bool: True если успешно
        """
        try:
            body = {}
            if settings:
                body['settings'] = settings
            if mappings:
                body['mappings'] = mappings
            
            self.es.indices.create(index=self.index_name, body=body)
            print(f"[ES] Индекс {self.index_name} создан")
            return True
        except Exception as e:
            print(f"[ERROR] Ошибка создания индекса: {e}")
            return False
    
    def close(self):
        """Закрыть соединение"""
        try:
            self.es.close()
        except:
            pass
    
    def __repr__(self):
        return f"ElasticsearchClient(host='{self.host}', port={self.port}, index='{self.index_name}')"