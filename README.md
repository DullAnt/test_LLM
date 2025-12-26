# TEST_LLM - Система тестирования RAG

## Требования

- Python 3.10+
- Elasticsearch 8.x
- Ollama (для LLM)
- Windows 10/11

## Установка

### 1. Клонирование репозитория

```
git clone https://github.com/your-repo/test_llm.git
```

```
cd test_llm
```

### 2. Создание виртуального окружения

```
python -m venv venv
```

```
venv\Scripts\activate
```

### 3. Установка зависимостей

```
pip install -r requirements.txt
```

### 4. Установка и запуск Elasticsearch

Скачайте Elasticsearch 8.x с официального сайта:

https://www.elastic.co/downloads/elasticsearch

Распакуйте архив и запустите:

```
cd elasticsearch-8.x.x\bin
```

```
elasticsearch.bat
```

Проверьте что Elasticsearch запущен:

```
curl http://localhost:9200
```

### 5. Установка и запуск Ollama

Скачайте Ollama с официального сайта:

https://ollama.com/download

Установите и запустите приложение.

Загрузите LLM модель:

```
ollama pull gemma2:2b
```

Проверьте что Ollama запущен:

```
ollama list
```

## Настройка

### Файл .env

Создайте файл `.env` в корне проекта:

```
EMBEDDING_MODEL=intfloat/multilingual-e5-large
OLLAMA_MODEL=gemma2:2b
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=600
ELASTIC_HOST=localhost
ELASTIC_PORT=9200
ELASTIC_INDEX=psb_docs
SIMILARITY_THRESHOLD=0.60
TOP_K=5
```

### Добавление документов

Поместите ваши документы (`.txt`, `.md`) в папку:

```
data/documents/
```

Поддерживаемые форматы: TXT, Markdown.

## Загрузка данных в Elasticsearch

### Создание индекса и загрузка векторов

```
python load_to_elasticsearch.py
```

Скрипт автоматически:
- Определит размерность embedding модели
- Создаст индекс с правильным mapping
- Разобьет документы на chunks
- Вычислит вектора для каждого chunk
- Загрузит в Elasticsearch

Проверка загрузки:

```
curl http://localhost:9200/psb_docs/_count
```

```
curl http://localhost:9200/psb_docs/_search?size=1
```

## Использование

### Базовый запуск

```
python main.py
```

### Запуск с параметрами

Тестирование на 10 вопросах:

```
python main.py --max-questions 10
```

С включенным HyDE:

```
python main.py --hyde
```

Без HyDE:

```
python main.py --no-hyde
```

Указать TOP-K для поиска:

```
python main.py --top-k 3
```

Использовать другую модель Ollama:

```
python main.py --ollama-model qwen2.5:7b
```

Увеличить таймаут для больших моделей:

```
python main.py --timeout 900
```

Использовать готовый набор вопросов:

```
python main.py --testset data/testsets/questions.jsonl
```

### Все параметры

```
python main.py --help
```

## Структура проекта

```
test_LLM/
├── data/
│   ├── documents/              # Исходные документы
│   ├── testsets/              # Наборы вопросов (JSONL)
│   └── reports/               # HTML отчеты
├── evaluate/
│   ├── __init__.py
│   ├── metrics.py             # Генерация метрик и HTML отчетов
│   ├── questions.py           # Извлечение вопросов из документов
│   └── similarity.py          # Вычисление схожести ответов
├── package/
│   ├── __init__.py
│   ├── CLI.py                 # Парсер аргументов командной строки
│   ├── config.py              # Конфигурация и embedding модели
│   ├── elastic.py             # Подключение к Elasticsearch
│   ├── evaluator.py           # Основной evaluator для тестирования
│   ├── loader.py              # Загрузка документов
│   └── ollama_detector.py     # Определение доступности Ollama
├── rag/
│   ├── __init__.py
│   ├── embeddings.py          # Заглушка, при ненадобности можно снести
│   ├── hyde.py                # HyDE генератор гипотез
│   ├── ollama_client.py       # Клиент для Ollama LLM
│   ├── prompts.py             # Системные промпты
│   └── retriever.py           # Векторный поиск в Elasticsearch
├── .env.example               # Пример конфигурации
├── docker-compose.yaml        # Docker конфигурация (опционально)
├── load_to_elasticsearch.py  # Загрузка документов в ES
├── main.py                    # Точка входа
├── README.md                  # Документация
└── requirements.txt           # Python зависимости
```

## Смена embedding модели

### Использование моделей из списка

В `package/config.py` измените:

```python
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2"  # или другую модель
)
```

Доступные модели:
- `paraphrase-multilingual-MiniLM-L12-v2` (384 dims)
- `intfloat/multilingual-e5-base` (768 dims)
- `intfloat/multilingual-e5-large` (1024 dims)

### Использование любой модели с HuggingFace

Укажите название модели:

```python
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/LaBSE"
```

Система автоматически:
- Определит размерность через HuggingFace API
- Создаст правильный индекс
- Загрузит модель

Пересоздайте индекс:

```
curl -X DELETE http://localhost:9200/psb_docs
```

```
python load_to_elasticsearch.py
```

## Формат тестового набора

Файл JSONL (по одному вопросу на строку):

```json
{"question": "Как восстановить ПИН-код?", "expected_answer": "Обратитесь в отделение банка с паспортом"}
{"question": "Где посмотреть баланс?", "expected_answer": "В мобильном приложении PSB-Mobile"}
```

Сохраните в `data/testsets/questions.jsonl`

Запустите тест:

```
python main.py --testset data/testsets/questions.jsonl
```

## Метрики качества

После тестирования генерируется HTML отчет в `data/reports/`

Метрики:
- Процент правильных ответов
- Средняя схожесть ответов
- Качество RAG (средний score найденных chunks)
- Детальная аналитика по каждому вопросу
- Распределение качества chunks
- Статистика использования источников

## Устранение неполадок

### Elasticsearch не запускается

Проверьте порт:

```
netstat -an | findstr 9200
```

Убедитесь что нет других процессов на порту 9200.

### Ollama не отвечает

Проверьте что Ollama запущен:

```
curl http://localhost:11434/api/tags
```

Проверьте доступные модели:

```
ollama list
```

### Ошибка размерности векторов

Удалите индекс и пересоздайте:

```
curl -X DELETE http://localhost:9200/psb_docs
```

```
python load_to_elasticsearch.py
```

### Низкое качество ответов

Увеличьте TOP-K:

```
python main.py --top-k 5
```

Используйте лучшую embedding модель:

Измените в `.env`:
```
EMBEDDING_MODEL=intfloat/multilingual-e5-large
```

Используйте более мощную LLM:

```
ollama pull qwen2.5:7b
```

```
python main.py --ollama-model qwen2.5:7b
```

## Рекомендуемые конфигурации

### Для быстрого тестирования

```
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
OLLAMA_MODEL=gemma2:2b
TOP_K=3
```

```
python main.py --max-questions 5 --no-hyde
```

### Для максимального качества

```
EMBEDDING_MODEL=intfloat/multilingual-e5-large
OLLAMA_MODEL=qwen2.5:7b
TOP_K=5
```

```
python main.py --max-questions 20 --hyde --timeout 900
```

### Для production

```
EMBEDDING_MODEL=intfloat/multilingual-e5-large
OLLAMA_MODEL=gemma2:9b
TOP_K=5
SIMILARITY_THRESHOLD=0.70
```

```
python main.py --testset data/testsets/production_questions.jsonl --hyde
```
