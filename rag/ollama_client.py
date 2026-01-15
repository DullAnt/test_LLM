"""
Ollama Client для взаимодействия с локальным LLM
"""

import re
from typing import List, Dict

import ollama
from ollama import Client
from langchain_ollama import ChatOllama


class OllamaClient:
    """Клиент для работы с Ollama (без requests)"""

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3", timeout: int = 300):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

        # Определить источник по порту
        self.source = self._detect_source()

        # Официальный ollama python client для диагностики (без requests)
        # В docs: Client(host=...), list(), chat(), generate() :contentReference[oaicite:4]{index=4}
        self.client = Client(host=self.host, timeout=5.0)

        # Проверка подключения (без requests)
        if self.check_connection():
            print(f"[OLLAMA] Подключено ({self.model})")
            print(f"         Источник: {self.source}")

            # Проверить доступность модели (без requests)
            if not self._check_model_available():
                print(f"[WARNING] Модель {self.model} не найдена!")
                self._print_available_models()

            # ChatOllama для генерации
            # validate_model_on_init есть в langchain-ollama :contentReference[oaicite:5]{index=5}
            self.llm = self._init_langchain_llm()

        else:
            self.llm = None
            print(f"[ERROR] Не удалось подключиться к Ollama: {self.host}")
            self._print_connection_help()

    def _init_langchain_llm(self) -> ChatOllama:
        """Инициализация ChatOllama с фолбэком на случай несовпадения версии."""
        try:
            return ChatOllama(
                model=self.model,
                base_url=self.host,
                validate_model_on_init=True,
                temperature=0.1,
                top_p=0.9,
                top_k=40,
                num_predict=200,
                repeat_penalty=1.1,
                sync_client_kwargs={"timeout": self.timeout},
            )
        except TypeError:
            # Фолбэк: если вдруг base_url/sync_client_kwargs не поддерживаются в твоей версии
            return ChatOllama(
                model=self.model,
                validate_model_on_init=True,
                temperature=0.1,
                top_p=0.9,
                top_k=40,
                num_predict=200,
                repeat_penalty=1.1,
            )

    def _detect_source(self) -> str:
        """Определить источник Ollama по порту"""
        if ":11434" in self.host or self.host.endswith("11434"):
            return "Локальная Ollama (порт 11434)"
        elif ":11435" in self.host or self.host.endswith("11435"):
            return "Docker Ollama (порт 11435)"
        return f"Кастомный сервер ({self.host})"

    # =========================
    # Диагностика (без requests)
    # =========================

    def check_connection(self) -> bool:
        """Проверка подключения к Ollama (через ollama sdk)"""
        try:
            _ = self.client.list()  # docs: ollama.list()/client.list() :contentReference[oaicite:6]{index=6}
            return True
        except Exception:
            return False

    def _check_model_available(self) -> bool:
        """Проверка доступности модели (через ollama sdk)"""
        try:
            data = self.client.list()
            # у разных версий структура чуть отличается — безопасно вытаскиваем
            models = getattr(data, "models", None) or data.get("models", [])
            names = []
            for m in models:
                # m может быть dict или объект
                name = getattr(m, "model", None) or getattr(m, "name", None) or (m.get("model") if isinstance(m, dict) else None) or (m.get("name") if isinstance(m, dict) else None)
                if name:
                    names.append(name)
            return any(n.startswith(self.model) for n in names)
        except Exception:
            return False

    def _print_available_models(self):
        """Вывод списка моделей (через ollama sdk)"""
        try:
            data = self.client.list()
            models = getattr(data, "models", None) or data.get("models", [])

            if not models:
                print("\n[WARNING] Нет установленных моделей!")
                self._print_install_model_help()
                return

            print("\n[INFO] Доступные модели:")
            for m in models:
                name = getattr(m, "model", None) or getattr(m, "name", None) or (m.get("model") if isinstance(m, dict) else None) or (m.get("name") if isinstance(m, dict) else "unknown")
                size = getattr(m, "size", None) or (m.get("size") if isinstance(m, dict) else 0) or 0
                size_gb = float(size) / (1024**3) if size else 0.0
                print(f"       - {name} ({size_gb:.1f}GB)")

            print("\n[TIP] Загрузите нужную модель:")
            if ":11435" in self.host:
                print(f"       docker exec test_llm_ollama ollama pull {self.model}")
            else:
                print(f"       ollama pull {self.model}")

        except Exception:
            pass

    def get_info(self) -> Dict:
        """Инфо об Ollama (без requests)"""
        info = {
            "host": self.host,
            "model": self.model,
            "source": self.source,
            "connected": self.check_connection(),
        }
        try:
            data = self.client.list()
            models = getattr(data, "models", None) or data.get("models", [])
            names = []
            for m in models:
                name = getattr(m, "model", None) or getattr(m, "name", None) or (m.get("model") if isinstance(m, dict) else None) or (m.get("name") if isinstance(m, dict) else None)
                if name:
                    names.append(name)
            info["available_models"] = names
        except Exception:
            info["available_models"] = []
        return info

    # =========================
    # Генерация (ChatOllama)
    # =========================

    def generate(self, question: str, context: List[str]) -> str:
        """Генерация ответа на вопрос с учетом контекста (через ChatOllama)"""
        if self.llm is None:
            return "Ошибка: Ollama недоступна или модель не инициализирована"

        system_text = (
            "Ты эксперт-консультант банка. Отвечай строго по правилам:\n"
            "1) Используй ТОЛЬКО информацию из контекста ниже.\n"
            "2) Ответ 1–3 предложения, без markdown.\n"
            "3) Если в контексте есть точные числа/проценты/формулы — копируй их точно.\n"
            "4) Если нужен расчёт — посчитай и дай итог.\n"
        )

        context_text = "\n\n".join(context) if context else ""
        human_text = (
            "КОНТЕКСТ:\n"
            f"{context_text}\n\n"
            "ВОПРОС:\n"
            f"{question}\n\n"
            "ОТВЕТ:"
        )

        try:
            msg = self.llm.invoke([("system", system_text), ("human", human_text)])
            answer = (msg.content or "").strip()
            return self._clean_response(answer)
        except Exception as e:
            print(f"[ERROR] Ошибка при генерации через ChatOllama: {e}")
            return "Ошибка генерации ответа"

    # =========================
    # Helpers
    # =========================

    def _print_install_model_help(self):
        print("\n[TIP]  Как установить модель:")
        if ":11435" in self.host:
            print("       1. Интерактивно: python setup_ollama.py")
            print(f"       2. Вручную: docker exec test_llm_ollama ollama pull {self.model}")
        else:
            print(f"       ollama pull {self.model}")
            print("       Рекомендуемые: qwen2.5:7b, gemma2:9b, gemma2:2b")

    def _print_connection_help(self):
        print("\n[TIP] Как исправить:")
        if ":11434" in self.host:
            print("       1) Проверьте что Ollama запущена: ollama list")
            print("       2) Переустановите: https://ollama.com/download")
            print("       3) Или используйте Docker: python main.py --ollama-host http://localhost:11435")
        elif ":11435" in self.host:
            print("       1) Проверьте контейнер: docker ps | grep ollama")
            print("       2) Запустите: docker-compose up -d ollama")
            print("       3) Или используйте локальную: python main.py --ollama-host http://localhost:11434")
        else:
            print(f"       Проверьте доступность сервера: {self.host}")

    def _clean_response(self, text: str) -> str:
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[-•]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^(Ответ:|ОТВЕТ:)\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(Краткий ответ:|КРАТКИЙ ОТВЕТ:)\s*", "", text, flags=re.IGNORECASE)
        return text.strip()

    def __repr__(self):
        status = " Connected" if self.check_connection() else " Disconnected"
        return f"OllamaClient(host='{self.host}', model='{self.model}', status={status})"
