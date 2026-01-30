from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import re


def generate_questions_from_files(files: List[Path], max_questions_per_file: int = 10) -> List[Dict]:
    questions: List[Dict] = []

    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            continue

        text = clean_text(text)
        if len(text) < 300:
            continue

        filename = file_path.name

        extracted = extract_qa_blocks(text)
        if extracted:
            for q, a in extracted[:max_questions_per_file]:
                questions.append({"question": q, "answer": a, "source": filename})
            continue

        questions.extend(generate_fallback_questions(text, filename, max_questions_per_file))

    print(f"[QUESTIONS] Сгенерировано: {len(questions)}")
    return questions


def save_questions_json(questions: List[Dict], output_path: str) -> int:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    return len(questions)


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_qa_blocks(text: str) -> List[Tuple[str, str]]:
    qa_pairs: List[Tuple[str, str]] = []

    patterns = [
        r"Q[:\-]\s*(.+?)\nA[:\-]\s*(.+?)(?=\n\n|\Z)",
        r"Вопрос[:\-]\s*(.+?)\nОтвет[:\-]\s*(.+?)(?=\n\n|\Z)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.S | re.I)
        for q, a in matches:
            qn = normalize(q)
            an = normalize(a)
            if qn and an:
                qa_pairs.append((qn, an))

    return qa_pairs


def generate_fallback_questions(text: str, filename: str, max_questions: int) -> List[Dict]:
    questions: List[Dict] = []

    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= 200]
    if not paragraphs:
        return questions

    sample = paragraphs[:max_questions]

    for p in sample:
        questions.append(
            {
                "question": f"О чем говорится в документе «{filename}»?",
                "answer": p[:900],
                "source": filename,
            }
        )

    return questions
