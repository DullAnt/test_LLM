import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from connections.config import Config
from llm.ollama_client import OllamaClient

def generate_questions_from_files(files: List[Path], max_questions_per_file: int = 5) -> List[Dict]:
    """
    Генерирует вопросы по содержимому файлов, используя LLM (Ollama).
    """
    questions: List[Dict] = []
    
    print(f"[GEN] Подключение к Ollama ({Config.OLLAMA_HOST})...")
    ollama = OllamaClient(
        host=Config.OLLAMA_HOST,
        model=Config.OLLAMA_MODEL,
        timeout=Config.OLLAMA_TIMEOUT
    )
    
    if not ollama.check_connection():
        print(f"[ERROR] Ollama недоступна по адресу {Config.OLLAMA_HOST}!")
        print("[WARN] Включаю режим 'Fallback' (глупые вопросы), так как LLM недоступна.")
        return _fallback_generator(files, max_questions_per_file)

    print(f"[GEN] Ollama OK. Модель: {Config.OLLAMA_MODEL}. Начинаю генерацию...")

    for file_path in tqdm(files, desc="Обработка файлов"):
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[ERR] Не удалось прочитать {file_path.name}: {e}")
            continue

        text = clean_text(text)
        if len(text) < 200:
            continue

        filename = file_path.name
        
        # Разбиваем на чанки ~1000 символов (примерно абзац-два)
        chunks = _split_text_smart(text)
        
        # Берем несколько чанков для генерации вопросов
        selected_chunks = chunks[:max_questions_per_file]

        for chunk_text in selected_chunks:
            # Генерируем вопрос
            qa_pair = _generate_qa_with_llm(ollama, chunk_text)
            
            if qa_pair:
                questions.append({
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "source": filename,
                    "chunk": chunk_text, # ЭТО ВАЖНО для RAG метрик
                })
            else:
                # Если LLM не смогла придумать вопрос, пропускаем или пробуем еще раз
                pass

    print(f"[GEN] Готово. Сгенерировано вопросов: {len(questions)}")
    return questions

def _generate_qa_with_llm(client: OllamaClient, context: str) -> Optional[Dict]:
    prompt = f"""
    Проанализируй текст и придумай 1 (один) конкретный вопрос, ответ на который есть в этом тексте.
    
    Текст:
    {context[:3000]}
    
    Требования:
    1. Вопрос должен быть на русском.
    2. Вопрос должен спрашивать конкретику (цифры, условия, сроки).
    3. Ответ скопируй из текста или переформулируй кратко.
    4. НЕ пиши "В тексте сказано".
    
    Ответ дай ТОЛЬКО в формате JSON:
    {{
        "question": "Твой вопрос",
        "answer": "Твой ответ"
    }}
    """
    
    try:
        response = client.generate(question=prompt, context=[])
        # Попытка вытащить JSON
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return None

def _split_text_smart(text: str, min_len=300) -> List[str]:
    """Умная разбивка на чанки по абзацам"""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        p = p.strip()
        if not p: continue
        
        if len(current_chunk) + len(p) < 1500:
            current_chunk += "\n\n" + p
        else:
            if len(current_chunk) > min_len:
                chunks.append(current_chunk.strip())
            current_chunk = p
            
    if len(current_chunk) > min_len:
        chunks.append(current_chunk.strip())
        
    return chunks

def _fallback_generator(files, max_questions) -> List[Dict]:
    """Старый метод, если все сломалось"""
    questions = []
    for f in files:
        text = clean_text(f.read_text(encoding="utf-8"))
        questions.append({
            "question": f"О чем файл {f.name}?",
            "answer": text[:500],
            "source": f.name,
            "chunk": text[:500]
        })
    return questions

def save_questions_json(questions: List[Dict], output_path: str) -> int:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    return len(questions)

def clean_text(text: str) -> str:
    return text.replace("\r", "").strip()
