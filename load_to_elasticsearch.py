"""
Загрузка документов в Elasticsearch с векторами
Использует единый ElasticsearchClient
"""

from connections.config import Config, DEFAULT_EMBEDDING_DIMS
from connections.elastic import ElasticsearchClient
from llm.embeddings import get_embedding_model


def main():
    """Главная функция"""
    print("=" * 60)
    print("ЗАГРУЗКА ДОКУМЕНТОВ В ELASTICSEARCH")
    print("=" * 60)

    try:
        # Создаем единый ES клиент
        es_client = ElasticsearchClient(
        url=Config.ELASTIC_URL,
        index_name=Config.ELASTIC_INDEX
        )

        
        # Проверяем подключение
        if not es_client.ping():
            raise ConnectionError(f"Не удалось подключиться к {Config.ELASTIC_URL}")
        
        # Создаем индекс с векторами
        es_client.create_index_with_vectors(dims=DEFAULT_EMBEDDING_DIMS)
        
        # Ищем документы
        files = es_client.find_documents(Config.DOCUMENTS_PATH)
        
        # Инициализируем embedding модель
        print("\n[EMB] Инициализация embedding модели...")
        embedding_model = get_embedding_model()
        
        # Загружаем документы с векторами
        stats = es_client.load_documents_with_vectors(
            files=files,
            embedding_model=embedding_model,
            chunk_size=500,
            overlap=50,
            dims=DEFAULT_EMBEDDING_DIMS,
        )
        
        # Проверяем индекс
        es_client.verify_index()
        
        print("\nЗагрузка завершена успешно!")
        print(f"Проверка: curl {Config.ELASTIC_URL}/{Config.ELASTIC_INDEX}/_count")

    except Exception as e:
        print(f"\nОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())