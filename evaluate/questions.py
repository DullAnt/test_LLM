"""
Модуль для работы с вопросами
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


def clean_markdown_text(text: str) -> str:
    """
    Очистка текста от markdown и HTML разметки
    """
    # Убрать HTML теги
    text = re.sub(r'<[^>]+>', '', text)
    
    # Убрать markdown заголовки (##, ###)
    text = re.sub(r'#{1,6}\s+', '', text)
    
    # Убрать ссылки [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Убрать жирный текст (**text**)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    
    # Убрать курсив (*text*)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    
    # Убрать Q: и A: префиксы
    text = re.sub(r'^[QA]:\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[ВО]:\s*', '', text, flags=re.MULTILINE)
    
    # Убрать множественные пробелы
    text = re.sub(r'\s+', ' ', text)
    
    # Убрать пробелы в начале и конце
    text = text.strip()
    
    return text


class QuestionExtractor:
    """Класс для извлечения Q&A из документов"""
    
    def __init__(self):
        self.patterns = [
            # Формат: **Q: вопрос** A: ответ
            r'\*\*Q:\s*([^*]+)\*\*\s*A:\s*([^\n]+(?:\n(?!\*\*Q:)[^\n]+)*)',
            # Формат: Q: вопрос A: ответ
            r'Q:\s*([^\n]+)\s*A:\s*([^\n]+(?:\n(?!Q:)[^\n]+)*)',
            # Формат: В: вопрос О: ответ
            r'В:\s*([^\n]+)\s*О:\s*([^\n]+(?:\n(?!В:)[^\n]+)*)',
        ]
    
    def extract_from_text(self, text: str) -> List[Dict[str, str]]:
        """Извлечение Q&A из текста"""
        questions = []
        
        # Применить все паттерны
        for pattern in self.patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                question_text = clean_markdown_text(match.group(1))
                answer_text = clean_markdown_text(match.group(2))
                
                if question_text and answer_text and len(answer_text) > 10:
                    questions.append({
                        'question': question_text,
                        'answer': answer_text
                    })
        
        # Паттерн для markdown заголовков
        header_pattern = r'###?\s+([^#\n]+\?)\s*\n\s*([^\n#]+(?:\n(?!###?)[^\n]+)*)'
        matches = re.finditer(header_pattern, text, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            question_text = clean_markdown_text(match.group(1))
            answer_text = clean_markdown_text(match.group(2))
            
            if question_text and answer_text and len(answer_text) > 10:
                questions.append({
                    'question': question_text,
                    'answer': answer_text
                })
        
        return questions


def extract_questions(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Извлечение вопросов из списка документов
    
    Args:
        documents: Список словарей с ключом 'content'
        
    Returns:
        Список вопросов
    """
    extractor = QuestionExtractor()
    all_questions = []
    
    for doc in documents:
        content = doc.get('content', '')
        if content:
            extracted = extractor.extract_from_text(content)
            all_questions.extend(extracted)
    
    return all_questions


def load_questions(filepath: str) -> List[Dict[str, str]]:
    """Загрузка вопросов из JSONL файла"""
    questions = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
        return questions
    except Exception as e:
        print(f"Ошибка загрузки вопросов: {e}")
        return []


def save_questions(questions: List[Dict[str, str]], output_dir: str = "data/testsets") -> str:
    """Сохранение вопросов в JSONL файл"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"questions_extracted_{timestamp}.jsonl"
    filepath = Path(output_dir) / filename
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for q in questions:
                f.write(json.dumps(q, ensure_ascii=False) + '\n')
        
        print(f"✅ Сохранено {len(questions)} вопросов в {filepath}")
        return str(filepath)
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")
        return ""


def extract_questions_from_elasticsearch(es_client, index: str = "psb_docs") -> List[Dict[str, str]]:
    """Извлечение вопросов из Elasticsearch"""
    questions = []
    extractor = QuestionExtractor()
    
    try:
        result = es_client.search(
            index=index,
            body={
                "query": {"match_all": {}},
                "size": 100
            }
        )
        
        for hit in result['hits']['hits']:
            content = hit['_source'].get('content', '')
            extracted = extractor.extract_from_text(content)
            questions.extend(extracted)
        
        return questions
    except Exception as e:
        print(f"Ошибка извлечения из Elasticsearch: {e}")
        return []