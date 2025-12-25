"""
CLI аргументы для TEST_LLM
"""

import argparse

from package.config import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_TIMEOUT,
    DEFAULT_ELASTIC_HOST,
    DEFAULT_ELASTIC_PORT,
    DEFAULT_ELASTIC_INDEX,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K
)



def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="TEST_LLM - Система тестирования RAG"
    )
    
    # Ollama настройки
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model (default: {DEFAULT_OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=DEFAULT_OLLAMA_HOST,
        help="Ollama host URL"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_OLLAMA_TIMEOUT,
        help="Timeout in seconds (default: {DEFAULT_OLLAMA_TIMEOUT})"
    )
    
    # Источник данных
    parser.add_argument(
        "--local-files",
        action="store_true",
        help="Use local files instead of Elasticsearch (default: Elasticsearch)"
    )
    parser.add_argument(
    "--documents",
        type=str,
        default="data/documents",
        help="Path to documents directory [requires --local-files]"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Path to questions JSONL file (optional)"
    )
    
    # Извлечение Q&A
    parser.add_argument(
        "--extract-qa",
        action="store_true",
        default=True,
        help="Auto-extract Q&A from documents (default: True)"
    )
    parser.add_argument(
        "--no-extract-qa",
        dest="extract_qa",
        action="store_false",
        help="Disable Q&A extraction"
    )
    
    # Параметры оценки
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Similarity threshold (default: {DEFAULT_SIMILARITY_THRESHOLD})"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=10,
        help="Maximum questions to test (default: 10)"
    )
    
    # HyDE
    parser.add_argument(
        "--no-hyde",
        dest="hyde",
        action="store_false",
        help="Disable HyDE"
    )
    
    # Elasticsearch
    parser.add_argument(
    "--es-host",
    type=str,
    default=DEFAULT_ELASTIC_HOST,
    help="Elasticsearch host (default: {DEFAULT_ELASTIC_HOST})"
    )
    parser.add_argument(
    "--es-port",
    type=int,
    default=DEFAULT_ELASTIC_PORT,
    help="Elasticsearch port (default: {DEFAULT_ELASTIC_PORT})"
    )
    parser.add_argument(
    "--es-index",
    type=str,
    default=DEFAULT_ELASTIC_INDEX,
    help="Elasticsearch index (default: {DEFAULT_ELASTIC_INDEX})"
    )

    
    # Дополнительно
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()