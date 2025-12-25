"""
RAG модули
"""

from .ollama_client import OllamaClient
from .retriever import DocumentRetriever
from .prompts import SYSTEM_PROMPT, TARIFF_PROMPT, DEFINITION_PROMPT

__all__ = [
    'OllamaClient',
    'DocumentRetriever',
    'SYSTEM_PROMPT',
    'TARIFF_PROMPT',
    'DEFINITION_PROMPT',
]