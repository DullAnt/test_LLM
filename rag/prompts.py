"""
Системные промпты для LLM
"""

"""
Промпты для LLM
"""

def create_rag_prompt(question: str, chunks: list, include_scores: bool = True) -> str:
    """
    Создает промпт с контекстом и вопросом
    
    Args:
        question: Вопрос пользователя
        chunks: Список найденных chunks с score
        include_scores: Показывать ли score в промпте
    """
    
    # Формируем контекст из chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get('text', '')
        score = chunk.get('score', 0)
        source = chunk.get('source', 'unknown')
        
        if include_scores:
            # С процентами (новый вариант)
            context_parts.append(
                f"[Chunk {i}, релевантность: {score:.0%}, источник: {source}]\n{text}"
            )
        else:
            # Без процентов (старый вариант)
            context_parts.append(f"[Chunk {i}]\n{text}")
    
    context = "\n\n".join(context_parts)
    
    # Промпт
    prompt = f"""Ты ассистент банка ПСБ (Промсвязьбанк). Отвечай только на основе предоставленного контекста.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ИНСТРУКЦИИ:
1. Используй ТОЛЬКО информацию из контекста
2. Если в контексте нет ответа - скажи "Информация не найдена в документах"
3. Ссылайся на chunks с высокой релевантностью (>80%)
4. Отвечай кратко и по делу
5. Используй русский язык

ОТВЕТ:"""
    
    return prompt


# Системный промпт (если нужен)
SYSTEM_PROMPT = """Ты ассистент банка ПСБ. Отвечаешь только на основе предоставленных документов.
Никогда не придумываешь информацию. Если ответа нет в документах - говоришь об этом честно."""

TARIFF_PROMPT = """Используй информацию о тарифах из контекста."""

DEFINITION_PROMPT = """Дай четкое определение на основе контекста."""