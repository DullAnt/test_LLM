"""
Скрипт для загрузки документов в Elasticsearch

Этот скрипт:
1. Подключается к Elasticsearch
2. Создает индекс с правильными настройками
3. Загружает все документы из data/documents/
4. Поддерживает .txt и .md файлы
"""

import sys
from pathlib import Path
from elasticsearch import Elasticsearch
from typing import List


def check_elasticsearch_connection(es_host: str = "localhost", es_port: int = 9200) -> Elasticsearch:
    """
    Проверка подключения к Elasticsearch
    
    Args:
        es_host: Хост Elasticsearch
        es_port: Порт Elasticsearch
        
    Returns:
        Elasticsearch: Клиент Elasticsearch
    """
    print(f"\n[ES] Подключение к Elasticsearch ({es_host}:{es_port})...")
    
    try:
        # Создать клиент
        es = Elasticsearch(
            [f"http://{es_host}:{es_port}"],
            verify_certs=False,
            request_timeout=30
        )
        
        # Проверить подключение
        if not es.ping():
            print("[ERROR] Не удалось подключиться к Elasticsearch!")
            print("\n💡 Как исправить:")
            print("   1. Запустите: docker-compose up -d elasticsearch")
            print("   2. Подождите 30 секунд")
            print("   3. Проверьте: curl http://localhost:9200")
            sys.exit(1)
        
        # Получить версию
        info = es.info()
        version = info['version']['number']
        print(f"[ES] ✅ Подключение успешно! (Elasticsearch {version})")
        
        return es
        
    except Exception as e:
        print(f"[ERROR] Ошибка подключения: {e}")
        print("\n💡 Как исправить:")
        print("   1. Запустите: docker-compose up -d elasticsearch")
        print("   2. Проверьте логи: docker logs test_llm_elasticsearch")
        sys.exit(1)


def create_index(es: Elasticsearch, index_name: str):
    """
    Создание индекса с оптимальными настройками
    
    Args:
        es: Клиент Elasticsearch
        index_name: Название индекса
    """
    # Удалить индекс если существует
    if es.indices.exists(index=index_name):
        print(f"[ES] Удаление существующего индекса '{index_name}'...")
        es.indices.delete(index=index_name)
        print(f"[ES] ✅ Индекс удален")
    
    # Настройки индекса
    index_settings = {
        "settings": {
            "number_of_shards": 1,      # Одна нода = один шард
            "number_of_replicas": 0,    # Без реплик для dev
            "analysis": {
                "analyzer": {
                    "russian": {
                        "type": "standard"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "filename": {
                    "type": "keyword"        # Точное совпадение
                },
                "content": {
                    "type": "text",          # Полнотекстовый поиск
                    "analyzer": "russian"
                },
                "path": {
                    "type": "keyword"
                },
                "chunks": {
                    "type": "text",          # Разбитый на chunks контент
                    "analyzer": "russian"
                }
            }
        }
    }
    
    # Создать индекс
    print(f"[ES] Создание индекса '{index_name}'...")
    es.indices.create(index=index_name, body=index_settings)
    print(f"[ES] ✅ Индекс создан")


def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Разбить текст на chunks с перекрытием
    
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
        # Если параграф + текущий chunk меньше лимита - добавить
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


def find_documents(docs_path: str) -> List[Path]:
    """
    Найти все документы в папке
    
    Args:
        docs_path: Путь к папке с документами
        
    Returns:
        List[Path]: Список путей к файлам
    """
    print(f"\n[ES] Поиск документов в '{docs_path}'...")
    
    docs_dir = Path(docs_path)
    
    # Проверить что папка существует
    if not docs_dir.exists():
        print(f"[ERROR] Папка не найдена: {docs_path}")
        print("\n💡 Как исправить:")
        print(f"   1. Создайте папку: mkdir {docs_path}")
        print(f"   2. Положите туда .txt или .md файлы")
        sys.exit(1)
    
    # Поддерживаемые форматы
    supported_formats = ['.txt', '.md']
    
    # Найти все файлы
    files = []
    for ext in supported_formats:
        files.extend(docs_dir.rglob(f'*{ext}'))
    
    if not files:
        print(f"[WARNING] Не найдено документов в '{docs_path}'")
        print(f"\n💡 Поддерживаемые форматы: {', '.join(supported_formats)}")
        print(f"💡 Положите файлы в папку: {docs_path}")
        sys.exit(1)
    
    print(f"[ES] ✅ Найдено документов: {len(files)}")
    
    # Показать список
    for i, file in enumerate(files, 1):
        size_kb = file.stat().st_size / 1024
        print(f"     {i}. {file.name} ({size_kb:.1f} KB)")
    
    return files


def load_documents(es: Elasticsearch, index_name: str, files: List[Path]):
    """
    Загрузить документы в Elasticsearch
    
    Args:
        es: Клиент Elasticsearch
        index_name: Название индекса
        files: Список файлов для загрузки
    """
    print(f"\n[ES] Загрузка документов в индекс '{index_name}'...")
    
    success_count = 0
    error_count = 0
    
    for i, file_path in enumerate(files, 1):
        try:
            # Прочитать файл
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Проверить что контент не пустой
            if not content.strip():
                print(f"[{i}/{len(files)}] ⚠️  {file_path.name} - пустой файл, пропускаем")
                error_count += 1
                continue
            
            # Разбить на chunks
            chunks = split_into_chunks(content)
            
            # Подготовить документ
            document = {
                'filename': file_path.name,
                'content': content,
                'path': str(file_path),
                'chunks': chunks
            }
            
            # Индексировать
            es.index(index=index_name, body=document)
            
            print(f"[{i}/{len(files)}] ✅ {file_path.name} ({len(content)} символов, {len(chunks)} chunks)")
            success_count += 1
            
        except UnicodeDecodeError:
            print(f"[{i}/{len(files)}] ❌ {file_path.name} - ошибка кодировки")
            error_count += 1
        except Exception as e:
            print(f"[{i}/{len(files)}] ❌ {file_path.name} - {e}")
            error_count += 1
    
    # Обновить индекс (flush changes)
    print(f"\n[ES] Обновление индекса...")
    es.indices.refresh(index=index_name)
    
    # Статистика
    doc_count = es.count(index=index_name)['count']
    
    print(f"\n" + "=" * 80)
    print(f"[ES] ✅ ЗАГРУЗКА ЗАВЕРШЕНА!")
    print("=" * 80)
    print(f"Успешно:      {success_count}/{len(files)} документов")
    print(f"Ошибки:       {error_count}/{len(files)} документов")
    print(f"В индексе:    {doc_count} документов")
    print(f"Индекс:       {index_name}")
    print("=" * 80)
    
    if success_count == 0:
        print("\n[ERROR] Ни один документ не был загружен!")
        sys.exit(1)


def verify_index(es: Elasticsearch, index_name: str):
    """
    Проверка индекса после загрузки
    
    Args:
        es: Клиент Elasticsearch
        index_name: Название индекса
    """
    print(f"\n[ES] Проверка индекса '{index_name}'...")
    
    # Количество документов
    doc_count = es.count(index=index_name)['count']
    print(f"[ES] Документов в индексе: {doc_count}")
    
    # Размер индекса
    stats = es.indices.stats(index=index_name)
    size_bytes = stats['indices'][index_name]['total']['store']['size_in_bytes']
    size_mb = size_bytes / (1024 * 1024)
    print(f"[ES] Размер индекса: {size_mb:.2f} MB")
    
    # Пример поиска
    print(f"\n[ES] Тестовый поиск...")
    try:
        result = es.search(
            index=index_name,
            body={
                "query": {"match_all": {}},
                "size": 1
            }
        )
        
        if result['hits']['total']['value'] > 0:
            first_doc = result['hits']['hits'][0]['_source']
            print(f"[ES] ✅ Первый документ: {first_doc['filename']}")
            print(f"[ES] ✅ Контент: {first_doc['content'][:100]}...")
        else:
            print(f"[ES] ⚠️  Документы не найдены")
    except Exception as e:
        print(f"[ES] ❌ Ошибка поиска: {e}")


def main():
    """Главная функция"""
    
    print("=" * 80)
    print("ЗАГРУЗКА ДОКУМЕНТОВ В ELASTICSEARCH")
    print("=" * 80)
    
    # Параметры (можно изменить)
    ES_HOST = "localhost"
    ES_PORT = 9200
    ES_INDEX = "psb_docs"
    DOCS_PATH = "data/documents"
    
    # 1. Подключение к Elasticsearch
    es = check_elasticsearch_connection(ES_HOST, ES_PORT)
    
    # 2. Создание индекса
    create_index(es, ES_INDEX)
    
    # 3. Поиск документов
    files = find_documents(DOCS_PATH)
    
    # 4. Загрузка документов
    load_documents(es, ES_INDEX, files)
    
    # 5. Проверка
    verify_index(es, ES_INDEX)
    
    # Готово!
    print(f"\n" + "=" * 80)
    print("✅ ГОТОВО! Теперь можете запустить:")
    print("   python main.py --max-questions 5")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Прервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)