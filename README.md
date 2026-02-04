# TEST_LLM

Комплексная система для реализации, тестирования и оценки RAG (Retrieval-Augmented Generation) решений. Проект включает в себя API-сервис на FastAPI и модуль автоматизированного тестирования качества поиска и генерации.

## Стек технологий

*   **Язык:** Python 3.10+
*   **API:** FastAPI, Uvicorn
*   **LLM:** Ollama (поддержка Gemma 2, Llama 3, Qwen и др.)
*   **Vector DB:** Elasticsearch 8.x
*   **Embeddings:** Sentence-Transformers (по умолчанию `intfloat/multilingual-e5-large`)
*   **Метрики:** Cosine Similarity, Custom Retrieval Quality

## Структура проекта

```text
TEST_LLM/
├── api/                        # API Сервис
│   ├── routes/                 # Маршруты (Endpoints)
│   │   ├── rag.py              # Основной эндпоинт RAG (/rag/ask)
│   │   └── system.py           # Системные эндпоинты (/health, /config)
│   ├── app.py                  # Фабрика приложения, Middleware
│   ├── ragser.py               # Логика RAG сервиса и Dependency Injection
│   └── schemas.py              # Pydantic модели
│
├── connections/                # Подключения
│   ├── config.py               # Класс конфигурации (загрузка из файла configuration)
│   └── elastic.py              # Клиент Elasticsearch
│
├── evaluate/                   # Модуль оценки качества
│   ├── evaluator.py            # Логика прогона тестов
│   ├── metrics.py              # Генерация HTML отчетов
│   └── similarity.py           # Расчет косинусного сходства
│
├── llm/                        # Работа с нейросетями
│   ├── ollama_client.py        # Клиент Ollama (с очисткой chain-of-thought)
│   ├── embeddings.py           # Обертка над Sentence-Transformers
│   ├── hyde.py                 # Обертка над Sentence-Transformers
│   ├── ollama_detector.py      # Обертка над Sentence-Transformers
│   ├── prompts.py              # Обертка над Sentence-Transformers
│   ├── ollama_detector.py      # Обертка над Sentence-Transformers
│   └── retriever.py            # Логика поиска и HyDE
│
├── load/                       # Загрузка данных
│   ├── loader.py               # Чтение документов
│   ├── questions.py            # Управление вопросами
│   └── question_generator.py   # Генерация синтетических вопросов
│
├── data/                       # Данные (исключены из git)
│   ├── documents/              # Исходные файлы (.txt, .md)
│   ├── questions/              # JSON с вопросами для тестов
│   └── reports/                # Результаты тестов (HTML)
│
├── configuration               # Файл настроек окружения
├── docker-compose.yaml         # Инфраструктура (ES, Ollama)
├── load_to_elasticsearch.py    # Скрипт индексации
├── main.py                     # CLI для запуска тестов
├── api_main.py                  # Скрипт запуска API сервера
└── requirements.txt            # Зависимости

Установка и настройка
1. Инфраструктура
Установите Python-зависимости:

bash
pip install -r requirements.txt
Запустите Elasticsearch и Ollama через Docker:

bash
docker-compose up -d
2. Конфигурация
Создайте файл configuration в корне проекта. Пример содержимого:

ini
# Ollama
OLLAMA_MODEL=gemma2:2b
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=600

# Elasticsearch
ELASTIC_HOST=localhost
ELASTIC_PORT=9200
ELASTIC_INDEX=psb_docs

# RAG & Embeddings
EMBEDDING_MODEL=intfloat/multilingual-e5-large
# EMBEDDING_DIMS=1024  <-- Можно не указывать, скрипт определит автоматически

# Настройки RAG
TOP_K=4
NEED_HYDE=True
SIMILARITY_THRESHOLD=0.7
3. Загрузка данных
Поместите файлы документов (.txt, .md) в папку data/documents/ и запустите индексацию. Скрипт автоматически создаст индекс, определит размерность векторов и сгенерирует тестовые вопросы.

bash
python load_to_elasticsearch.py
Запуск API
Для запуска веб-сервера используйте скрипт run_api.py:

bash
python run_api.py
После запуска:

Документация (Swagger): http://127.0.0.1:8000/docs

Эндпоинт RAG: POST /rag/ask

Пример запроса:

JSON
{
  "query": "Какие условия по ипотеке?",
  "top_k": 3,
  "hyde": true
}
Тестирование и оценка качества
Для запуска массовой оценки качества работы RAG на сгенерированных вопросах используйте main.py.

bash
# Запуск с параметрами из конфига
python main.py

# Переопределение параметров через CLI
python main.py --no-hyde --top-k 5 --max-questions 20
Метрики в отчетах
После завершения теста в папке data/reports/ создается HTML-отчет.

Reference Retrieval Quality:
Показывает максимальное косинусное сходство между вектором эталонного чанка (из которого был сгенерирован вопрос) и векторами найденных чанков.

Высокое значение (>0.85) означает, что система находит именно ту информацию, которая нужна.

Answer Similarity:
Косинусное сходство между сгенерированным ответом модели и эталонным ответом.

HyDE (Hypothetical Document Embeddings):
Если включено (NEED_HYDE=True), система сначала генерирует гипотетический ответ на вопрос, а затем использует его вектор для поиска. Это улучшает поиск по смыслу, но занимает больше времени.

Решение проблем
Ошибка при создании индекса / неверная размерность:
Если вы сменили модель эмбеддингов, удалите старый индекс вручную:

bash
curl -X DELETE http://localhost:9200/psb_docs
Затем запустите python load_to_elasticsearch.py заново.

Ollama model not found:
Убедитесь, что модель, указанная в configuration (например, gemma2:2b), загружена в Ollama:

bash
ollama list
ollama pull gemma2:2b
