"""
Ollama Client для взаимодействия с локальным LLM
"""

import requests
import json
import re
from typing import List, Dict, Optional


class OllamaClient:
    """Клиент для работы с Ollama API"""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3", timeout: int = 300):
        """
        Args:
            host: URL Ollama сервера
            model: Название модели
            timeout: Таймаут запроса в секундах
        """
        self.host = host.rstrip('/')
        self.model = model
        self.timeout = timeout
        self._ollama_version = None
        
        # Определить источник по порту
        self.source = self._detect_source()
        
        # Проверка подключения
        if self.check_connection():
            print(f"[OLLAMA] ✅ Подключено ({self.model})")
            print(f"         Источник: {self.source}")
            
            # Проверить доступность модели
            if not self._check_model_available():
                print(f"[WARNING] ⚠️ Модель {self.model} не найдена!")
                self._print_available_models()
        else:
            print(f"[ERROR] ❌ Не удалось подключиться к Ollama: {self.host}")
            self._print_connection_help()
    
    def _detect_source(self) -> str:
        """Определить источник Ollama по порту"""
        if ":11434" in self.host or self.host.endswith("11434"):
            return "Локальная Ollama (порт 11434)"
        elif ":11435" in self.host or self.host.endswith("11435"):
            return "Docker Ollama (порт 11435)"
        else:
            return f"Кастомный сервер ({self.host})"
    
    def check_connection(self) -> bool:
        """Проверка подключения к Ollama"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                # Получить версию Ollama
                try:
                    version_response = requests.get(f"{self.host}/api/version", timeout=2)
                    if version_response.status_code == 200:
                        self._ollama_version = version_response.json().get('version', 'unknown')
                except:
                    pass
                return True
            return False
        except requests.exceptions.ConnectionError:
            return False
        except requests.exceptions.Timeout:
            print(f"[ERROR] Таймаут подключения к Ollama (>{5}s)")
            return False
        except Exception as e:
            print(f"[ERROR] Ошибка подключения к Ollama: {e}")
            return False
    
    def _check_model_available(self) -> bool:
        """Проверка доступности модели"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                return any(m.get('name', '').startswith(self.model) for m in models)
            return False
        except:
            return False
    
    def _print_available_models(self):
        """Вывод списка доступных моделей"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                
                if models:
                    print("\n[INFO] 📋 Доступные модели:")
                    for m in models:
                        name = m.get('name', 'unknown')
                        size_gb = m.get('size', 0) / (1024**3)
                        print(f"       - {name} ({size_gb:.1f}GB)")
                    
                    print(f"\n[TIP] 💡 Загрузите нужную модель:")
                    if ":11435" in self.host:
                        print(f"       docker exec test_llm_ollama ollama pull {self.model}")
                    else:
                        print(f"       ollama pull {self.model}")
                else:
                    print("\n[WARNING] Нет установленных моделей!")
                    self._print_install_model_help()
        except:
            pass
    
    def _print_install_model_help(self):
        """Подсказка по установке модели"""
        print("\n[TIP] 💡 Как установить модель:")
        if ":11435" in self.host:
            # Docker Ollama
            print("       1. Интерактивно: python setup_ollama.py")
            print(f"       2. Вручную: docker exec test_llm_ollama ollama pull {self.model}")
        else:
            # Локальная Ollama
            print(f"       ollama pull {self.model}")
            print("       Рекомендуемые: qwen2.5:7b, gemma2:9b, gemma2:2b")
    
    def _print_connection_help(self):
        """Подсказка при ошибке подключения"""
        print("\n[TIP] 💡 Как исправить:")
        
        if ":11434" in self.host:
            # Локальная Ollama
            print("       1. Проверьте что Ollama запущена:")
            print("          ollama list")
            print("       2. Переустановите: https://ollama.com/download")
            print("       3. Или используйте Docker:")
            print("          python main.py --ollama-host http://localhost:11435")
        
        elif ":11435" in self.host:
            # Docker Ollama
            print("       1. Проверьте что контейнер запущен:")
            print("          docker ps | grep ollama")
            print("       2. Запустите контейнер:")
            print("          docker-compose up -d ollama")
            print("       3. Или используйте локальную Ollama:")
            print("          python main.py --ollama-host http://localhost:11434")
        else:
            print(f"       Проверьте доступность сервера: {self.host}")
    
    def get_info(self) -> Dict:
        """Получить информацию об Ollama"""
        info = {
            "host": self.host,
            "model": self.model,
            "source": self.source,
            "connected": self.check_connection(),
            "version": self._ollama_version
        }
        
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                info["available_models"] = [m.get('name') for m in data.get('models', [])]
        except:
            info["available_models"] = []
        
        return info
    
    def generate(self, question: str, context: List[str]) -> str:
        """
        Генерация ответа на вопрос с учетом контекста
        
        Args:
            question: Вопрос пользователя
            context: Список релевантных текстов
            
        Returns:
            Сгенерированный ответ
        """
        
        # Формирование улучшенного промпта
        if context:
            context_text = "\n\n".join(context)
            prompt = f"""Ты эксперт-консультант банка. Ответь на вопрос клиента на основе документации.

ПРАВИЛА:
1. Используй ТОЛЬКО информацию из контекста ниже
2. Отвечай кратко и точно (1-3 предложения)
3. Если нужен расчет - посчитай и дай конкретное число
4. Если в контексте есть точная формула или проценты - используй их
5. НЕ говори "информация отсутствует" если она есть в контексте
6. Не используй markdown разметку (жирный, списки и т.д.)
7. Копируй точные числа и формулы из контекста

ПРИМЕРЫ:

Вопрос: Посчитай комиссию если я превышу лимит на 20000
Контекст: "50 000₽ - бесплатно (0%) 20 000₽ - комиссия 0,8% = 160₽"
Ответ: 50 000₽ - бесплатно (0%), 20 000₽ - комиссия 0,8% = 160₽. Итого комиссия: 160₽

Вопрос: Могу ли я снять без комиссии в ВТБ?
Контекст: "Да, 0% комиссии для карт Мир в банкоматах ВТБ"
Ответ: Да, 0% комиссии для карт платежной системы Мир в банкоматах ВТБ.

Вопрос: В чем разница между картами?
Контекст: "Основная карта выпускается на владельца счета. Дополнительная карта выпускается на другое лицо."
Ответ: Основная карта выпускается на владельца счета, а Дополнительная карта выпускается на другое лицо.

КОНТЕКСТ:
{context_text}

ВОПРОС: {question}

ОТВЕТ:"""
        else:
            prompt = f"""Ответь на вопрос на основе твоих знаний о банковских продуктах.

Вопрос: {question}

Ответ:"""
        
        try:
            # Запрос к Ollama API
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Низкая температура для точности
                        "top_p": 0.9,
                        "top_k": 40,
                        "num_predict": 200,  # Ограничение длины ответа
                        "repeat_penalty": 1.1,
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                # Очистка ответа от markdown
                answer = self._clean_response(answer)
                
                return answer
            
            elif response.status_code == 404:
                error_msg = f"Модель {self.model} не найдена"
                print(f"[ERROR] {error_msg}")
                self._print_available_models()
                return f"Ошибка: {error_msg}"
            
            else:
                print(f"[ERROR] Ollama вернул код {response.status_code}")
                try:
                    error_data = response.json()
                    error_detail = error_data.get('error', 'Unknown error')
                    print(f"[ERROR] Детали: {error_detail}")
                except:
                    pass
                return "Ошибка генерации ответа"
                
        except requests.exceptions.Timeout:
            print(f"[ERROR] Таймаут запроса к Ollama (>{self.timeout}s)")
            print(f"[TIP] Увеличьте таймаут: --timeout 900")
            return "Таймаут генерации ответа"
        
        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Потеряно соединение с Ollama")
            self._print_connection_help()
            return "Ошибка соединения с Ollama"
        
        except Exception as e:
            print(f"[ERROR] Ошибка при генерации: {e}")
            return "Ошибка генерации ответа"
    
    def _clean_response(self, text: str) -> str:
        """Очистка ответа от markdown разметки"""
        # Убрать markdown
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **жирный**
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *курсив*
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # заголовки
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)   # списки 1. 2. 3.
        text = re.sub(r'^[-•]\s+', '', text, flags=re.MULTILINE)    # буллеты - и •
        
        # Убрать префиксы если LLM их повторил
        text = re.sub(r'^(Ответ:|ОТВЕТ:)\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(Краткий ответ:|КРАТКИЙ ОТВЕТ:)\s*', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def __repr__(self):
        status = "✅ Connected" if self.check_connection() else "❌ Disconnected"
        return f"OllamaClient(host='{self.host}', model='{self.model}', status={status})"