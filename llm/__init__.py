"""
LLM система (модели, генерация, поиск)
"""

# Ollama - клиент и детектор
from llm.ollama_client import OllamaClient
from llm.ollama_detector import (
    detect_ollama,
    check_ollama,
    print_ollama_status,
    get_ollama_host_with_fallback
)

# Embeddings
from llm.embeddings import get_embedding_model, EmbeddingModel

# Retrieval
from llm.retriever import DocumentRetriever

# HyDE
from llm.hyde import HyDEGenerator

# Prompts
from llm.prompts import (
    create_rag_prompt,
    SYSTEM_PROMPT,
    TARIFF_PROMPT,
    DEFINITION_PROMPT
)

# CLI
from llm.CLI import parse_arguments

__all__ = [
    # Ollama Client
    "OllamaClient",
    # Ollama Detector
    "detect_ollama",
    "check_ollama",
    "print_ollama_status",
    "get_ollama_host_with_fallback",
    # Embeddings
    "get_embedding_model",
    "EmbeddingModel",
    # Retrieval
    "DocumentRetriever",
    # HyDE
    "HyDEGenerator",
    # Prompts
    "create_rag_prompt",
    "SYSTEM_PROMPT",
    "TARIFF_PROMPT",
    "DEFINITION_PROMPT",
    # CLI
    "parse_arguments",
]