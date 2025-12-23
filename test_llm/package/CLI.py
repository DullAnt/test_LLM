"""
CLI аргументы для TEST_LLM
"""

import argparse
import os


def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="TEST_LLM - Система тестирования RAG"
    )
    
    # Ollama настройки
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ["OLLAMA_MODEL"],
        help="Ollama model (default in .env)"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama host URL"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds (default: 600)"
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
        default=3,
        help="Number of chunks to retrieve (default: 3)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Similarity threshold (default: 0.60)"
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
    default="localhost",
    help="Elasticsearch host (default: localhost)"
    )
    parser.add_argument(
    "--es-port",
    type=int,
    default=9200,
    help="Elasticsearch port (default: 9200)"
    )
    parser.add_argument(
    "--es-index",
    type=str,
    default="psb_docs",
    help="Elasticsearch index (default: psb_docs)"
    )

    
    # Дополнительно
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()