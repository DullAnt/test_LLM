# TEST_LLM - Система тестирования RAG

Продвинутая RAG (Retrieval-Augmented Generation) система для работы с документами банка ПСБ с поддержкой векторного поиска, HyDE и автоматической оценки качества.

## Содержание

- [Что за проект](#что-за-проект)
- [Структура проекта](#структура-проекта)
- [Как запустить](#как-запустить)
- [Метрики качества](#метрики-качества)

---

## Что за проект

**TEST_LLM** - это комплексная система для тестирования и оценки качества RAG (Retrieval-Augmented Generation) систем. Основные возможности:

### Ключевые фичи

-**Векторный поиск** через Elasticsearch с поддержкой cosine similarity
-**LLM генерация** через Ollama (Gemma2, Llama3, Qwen и др.)
-**HyDE** (Hypothetical Document Embeddings) для улучшения поиска
-**Автоматическая оценка качества** с красивыми HTML отчетами
-**Гибкая конфигурация** через .env файлы
-**Docker-ready** с docker-compose для быстрого старта

### Технологический стек

- **Embeddings**: Sentence-Transformers (multilingual-e5-large)
- **Vector DB**: Elasticsearch 8.12
- **LLM**: Ollama (локально или Docker)
- **Framework**: LangChain + Python 3.10+
- **Evaluation**: Cosine similarity для оценки ответов

---

##  Структура проекта
```
TEST_LLM/
├── connections/              # 1 ПОДКЛЮЧЕНИЯ
│   ├── __init__.py
│   ├── config.py            # Конфигурация (Config)
|   ├── configuration             # Конфигурация (.env файл)
│   └── elastic.py           # Elasticsearch клиент
│
├── loader/                   # 2 ЗАГРУЗКА ДАННЫХ
│   ├── __init__.py
│   ├── loader.py            # Загрузка документов
│   └── questions.py         # Загрузка/извлечение вопросов
│
├── llm/                      # 3 LLM СИСТЕМА
│   ├── __init__.py
│   ├── ollama_client.py     # Ollama клиент
│   ├── ollama_detector.py   # Автодетект Ollama
│   ├── embeddings.py        # Embedding модели
│   ├── retriever.py         # Векторный поиск
│   ├── hyde.py              # HyDE генерация
│   ├── prompts.py           # Системные промпты
│   ├── head.py              # Быстрый тестовый интерфейс
│   └── cli.py               # CLI аргументы
│
├── evaluation/               # 🔧 ОЦЕНКА КАЧЕСТВА
│   ├── __init__.py
│   ├── evaluator.py         # Главный evaluator
│   ├── metrics.py           # HTML отчеты
│   └── similarity.py        # Сравнение сгенерированного и ожидаемого ответов
│
├── data/                     # Данные
│   ├── documents/           # Исходные документы (.txt, .md)
│   ├── reports/             # HTML отчеты
│   └── testsets/            # Наборы вопросов (.jsonl)
│
├── main.py                   # Тестирование
├── testrag.py                # Быстрый тест (один вопрос)
├── load_to_elasticsearch.py  # Загрузка документов в ES
├── docker-compose.yaml       # Docker окружение
├── requirements.txt          # Зависимости
└── README.md                 # документация
```

### Основные модули

#### connections/ - Подключения
- **config.py** - Единая конфигурация всей системы
- **elastic.py** - Клиент для Elasticsearch с векторным поиском

#### loader/ - Загрузка данных
- **loader.py** - Загрузка документов из файлов или ES
- **questions.py** - Работа с вопросами (загрузка, извлечение, сохранение)

#### llm/ - LLM система
- **ollama_client.py** - Основной клиент для Ollama
- **embeddings.py** - Sentence-transformers модели
- **retriever.py** - Векторный поиск в Elasticsearch
- **hyde.py** - HyDE для улучшения поиска
- **head.py** - Упрощенный интерфейс для быстрого тестирования

#### evaluation/ - Оценка
- **evaluator.py** - Автоматическое тестирование на наборах вопросов
- **metrics.py** - Генерация HTML отчетов с аналитикой
- **similarity.py** - Вычисление схожести ответов

---

## Как запустить

### Предварительные требования

- Python 3.10+
- Docker & Docker Compose
- 8GB+ RAM (для Elasticsearch + Ollama)

### Шаг 1: Клонирование и установка
```bash
# Клонировать репозиторий
git clone <your-repo>
cd TEST_LLM


# Установить зависимости
pip install -r requirements.txt
```

### Шаг 2: Запуск инфраструктуры
```bash
# Запустить Elasticsearch + Ollama
docker-compose up -d

# Проверить статус
docker ps
```

### Шаг 3: Настройка конфигурации

Отредактируйте файл `configuration`:
```env
# Ollama Configuration
OLLAMA_MODEL=gemma2:2b
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=600

# Elasticsearch Configuration
ELASTIC_HOST=localhost
ELASTIC_PORT=9200
ELASTIC_INDEX=psb_docs

# Embeddings Configuration
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EMBEDDING_DIMS=1024

# Evaluation Configuration
TOP_K=3
NEED_HYDE=True
SIMILARITY_THRESHOLD=0.7
```

### Шаг 4: Загрузка документов в Elasticsearch
```bash
# Положить .txt или .md файлы в data/documents/
# Затем загрузить в Elasticsearch:
python load_to_elasticsearch.py
```

### Шаг 5: Быстрый тест
```bash
# Задать один вопрос, указанный в testrag
python testrag.py "Что такое ИЗП?"

# С параметрами
python testrag.py "Что такое закупка?" --top-k 5 --no-hyde --show-sources

```

### Шаг 6: Полное тестирование
```bash
# Тестирование с Elasticsearch
python main.py  --max-questions 10  #берет по умолчанию из эластика

# Тестирование с локальными файлами
python main.py --local-files --extract-qa --max-questions 10

# Параметры запуска
python main.py \
  --model gemma2:2b \
  --top-k 5 \
  --threshold 0.7 \
  --max-questions 20 \
  --seed 42
```

### Параметры командной строки

#### testrag.py (быстрый тест)
```bash
python testrag.py [вопрос] [опции]

Опции:
  --top-k N              Количество документов (default: 3)
  --no-hyde              Отключить HyDE
  --model MODEL          Ollama модель
  --embeddings MODEL     Embedding модель
  --show-docs            Показать полные документы
  --show-sources         Показать только источники
  --quiet                Минимальный вывод
```


##   Метрики качества

### Автоматическая оценка

Система автоматически оценивает качество ответов по нескольким метрикам:

#### 1. Accuracy (Точность)
```
Accuracy = (Правильных ответов / Всего вопросов) × 100%
```
- Ответ считается **правильным**, если `similarity >= threshold` (по умолчанию 0.7)
- Измеряется в процентах от 0% до 100%

#### 2. Cosine Similarity (Схожесть)
```
Similarity = cosine_similarity(embedding_generated, embedding_expected)
```
- Вычисляется через embedding модель (multilingual-e5-large)
- Диапазон от 0.0 (разные) до 1.0 (идентичные)
- **Порог качества**: 0.7 (70%)

#### 3. RAG Quality Score
```
RAG Score = average(chunk_scores)
```
- Оценка качества найденных документов
- Показывает релевантность векторного поиска
- **Высокое качество**: ≥0.7 (70%+)
- **Среднее качество**: 0.5-0.7 (50-70%)
- **Низкое качество**: <0.5 (<50%)

#### 4. Response Time
- Время генерации одного ответа (в секундах)
- Включает: поиск chunks + генерация LLM

### HTML Отчеты

После каждого тестирования генерируется детальный HTML отчет в `data/reports/`:

**Содержание отчета:**
- Общая статистика (accuracy, similarity, качество RAG)
- Графики распределения качества chunks
- Использование источников (какие документы чаще находились)
- Детальная таблица всех найденных chunks
- Детальное сравнение: вопрос → ожидаемый ответ → ответ системы → найденные chunks

**Пример отчета:**
```
data/reports/report_hyde_20250122_143045.html
```

### Интерпретация результатов

#### Отличные результаты 
- Accuracy: **85-100%**
- Avg Similarity: **0.80-1.00**
- RAG Score: **0.75-1.00**

#### Хорошие результаты 
- Accuracy: **70-85%**
- Avg Similarity: **0.70-0.80**
- RAG Score: **0.65-0.75**

#### Требуют улучшения 
- Accuracy: **<70%**
- Avg Similarity: **<0.70**
- RAG Score: **<0.65**


## 🔧 Дополнительно

### Очистка и перезапуск
```bash
# Очистить Elasticsearch
docker-compose down -v

# Пересоздать индекс
python load_to_elasticsearch.py

# Полная очистка
rm -rf data/reports/*
```

### Troubleshooting

**Ollama не найдена:**
```bash
# Проверить статус
docker ps | grep ollama

# Перезапустить
docker-compose restart ollama

# Загрузить модель
docker exec -it test_llm_ollama ollama pull gemma2:2b
```

**Elasticsearch недоступен:**
```bash
# Проверить
curl http://localhost:9200

# Перезапустить
docker-compose restart elasticsearch
```

**Ошибки импорта:**
```bash
# Переустановить зависимости
pip install --upgrade -r requirements.txt
```

---
