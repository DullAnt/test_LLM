"""
Ollama Client для взаимодействия с локальным LLM
"""
import re
from typing import List, Dict, Any
from ollama import Client
from langchain_ollama import ChatOllama

class OllamaClient:
    """Клиент для работы с Ollama (без requests)"""

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3", timeout: int = 300):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.source = self._detect_source()
        # Официальный клиент
        self.client = Client(host=self.host, timeout=5.0)

        if self.check_connection():
            print(f"[OLLAMA] Подключено ({self.model})")
            print(f"         Источник: {self.source}")
            
            # Проверяем наличие модели
            if not self._check_model_available():
                print(f"[WARNING] Модель '{self.model}' не найдена в списке!")
                self._print_available_models()
            
            self.llm = self._init_langchain_llm()
        else:
            self.llm = None
            print(f"[ERROR] Не удалось подключиться к Ollama: {self.host}")

    def _init_langchain_llm(self) -> ChatOllama:
        """Инициализация ChatOllama с фолбэком."""
        common_params = {
            "model": self.model,
            "validate_model_on_init": True,
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": 200,
            "repeat_penalty": 1.1,
        }
        try:
            return ChatOllama(
                base_url=self.host,
                sync_client_kwargs={"timeout": self.timeout},
                **common_params
            )
        except TypeError:
            return ChatOllama(**common_params)

    # =========================
    # Вспомогательные методы
    # =========================
    def _get_model_name(self, m: Any) -> str:
        """Безопасное извлечение имени модели из ответа Ollama"""
        # 1. Пробуем атрибуты (если это объект)
        name = getattr(m, "model", None) or getattr(m, "name", None)
        
        # 2. Пробуем ключи словаря 
        if not name and isinstance(m, dict):
            name = m.get("model") or m.get("name")
            
        return str(name) if name else "unknown"

    def _get_model_size(self, m: Any) -> float:
        """Безопасное извлечение размера модели"""
        size = getattr(m, "size", None)
        if not size and isinstance(m, dict):
            size = m.get("size")
        try:
            return float(size) / (1024**3) if size else 0.0
        except:
            return 0.0

    # =========================
    # Очистка ответа
    # =========================
    def _extract_final_answer(self, raw_text: str) -> str:
        """Удаляет мысли (think/reasoning) из ответа."""
        # 1. Тег <answer>
        match = re.search(r'<answer>(.*?)</answer>', raw_text, re.DOTALL)
        if match: return match.group(1).strip()
        
        # 2. Ключевые слова
        for keyword in ["Ответ:", "Answer:"]:
            if keyword in raw_text:
                return raw_text.split(keyword, 1)[1].strip()
        
        # 3. Удаление тегов мыслей
        text = re.sub(r'<(thinking|reasoning|thought)>.*?</\1>', '', raw_text, flags=re.DOTALL)
        return text.strip() if len(text) < len(raw_text) else raw_text

    def _clean_response(self, text: str) -> str:
        """Финальная полировка текста."""
        if not text: return ""
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()

    def _detect_source(self) -> str:
        if ":11434" in self.host: return "Локальная Ollama (порт 11434)"
        if ":11435" in self.host: return "Docker Ollama (порт 11435)"
        return f"Кастомный сервер ({self.host})"

    # =========================
    # Диагностика
    # =========================
    def check_connection(self) -> bool:
        try:
            self.client.list()
            return True
        except:
            return False

    def _check_model_available(self) -> bool:
        """Проверка доступности модели (универсальная)"""
        try:
            data = self.client.list()
            # models может быть объектом ListResponse или словарем
            models = getattr(data, "models", []) or (data.get("models", []) if isinstance(data, dict) else [])
            
            for m in models:
                full_name = self._get_model_name(m)
                # Проверяем вхождение (на случай gemma2:2b vs gemma2:2b-instruct)
                if full_name.startswith(self.model):
                    return True
            return False
        except Exception:
            return False

    def _print_available_models(self):
        try:
            data = self.client.list()
            models = getattr(data, "models", []) or (data.get("models", []) if isinstance(data, dict) else [])
            
            if not models:
                print("\n[WARNING] Список моделей пуст.")
                return

            print("\n[INFO] Доступные модели (найденные в системе):")
            for m in models:
                name = self._get_model_name(m)
                size = self._get_model_size(m)
                print(f"       - {name} ({size:.1f}GB)")
        except Exception:
            print("[ERROR] Не удалось получить список моделей.")

    def get_info(self) -> Dict:
        return {
            "host": self.host,
            "model": self.model,
            "connected": self.check_connection()
        }
    
    # =========================
    # Генерация
    # =========================
    def generate(self, question: str, context: List[str]) -> str:
        if self.llm is None:
            return "Ошибка: Ollama недоступна"
            
        system_text = (
            "Ты эксперт-консультант банка. Отвечай строго по правилам:\n"
            "1) Используй ТОЛЬКО информацию из контекста ниже.\n"
            "2) Ответ 1–3 предложения, без markdown.\n"
            "3) Копируй числа и факты точно.\n"
        )
        context_text = "\n\n".join(context) if context else ""
        human_text = f"КОНТЕКСТ:\n{context_text}\n\nВОПРОС:\n{question}\n\nОТВЕТ:"

        try:
            msg = self.llm.invoke([("system", system_text), ("human", human_text)])
            raw = (msg.content or "").strip()
            # Очистка
            return self._clean_response(self._extract_final_answer(raw))
        except Exception as e:
            print(f"[ERROR] Генерация: {e}")
            return "Ошибка"
