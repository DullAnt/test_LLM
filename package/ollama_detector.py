"""
Автоопределение Ollama (локальная или Docker)
"""

import requests
from package.config import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_HOST


def detect_ollama():
    """
    Автоопределение доступной Ollama
    
    Returns:
        tuple: (host, source) где source = 'local' | 'docker' | None
    """
    
    # Проверка локальной Ollama (порт 11434)
    local_host = DEFAULT_OLLAMA_HOST
    if check_ollama(local_host):
        return local_host, 'local'
    
    # Проверка Docker Ollama (порт 11435)
    docker_host = "http://localhost:11435"
    if check_ollama(docker_host):
        return docker_host, 'docker'
    
    # Ollama не найдена
    return None, None


def check_ollama(host: str) -> bool:
    """
    Проверка доступности Ollama по указанному адресу
    
    Args:
        host: URL Ollama сервера
        
    Returns:
        bool: True если Ollama доступна
    """
    try:
        response = requests.get(f"{host}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def print_ollama_status():
    """Вывод статуса Ollama"""
    print("\n[OLLAMA] Поиск доступной Ollama...")
    
    host, source = detect_ollama()
    
    if source == 'local':
        print(f"Найдена локальная Ollama: {host}")
        print("   Используется установленная на вашем ПК версия")
        return host, source
    
    elif source == 'docker':
        print(f"Найдена Docker Ollama: {host}")
        print("   Используется контейнерная версия")
        return host, source
    
    else:
        print("Ollama не найдена!")
        print("\n Варианты решения:")
        print("   1. Установить локально: https://ollama.com/download")
        print("   2. Запустить в Docker:")
        print("      docker-compose up -d ollama")
        print("      docker exec test_llm_ollama ollama pull {DEFAULT_OLLAMA_MODEL}")
        return None, None


def get_ollama_host_with_fallback(preferred_host: str = None) -> tuple:
    """
    Получить host Ollama с автоопределением
    
    Args:
        preferred_host: Предпочтительный host (если указан пользователем)
        
    Returns:
        tuple: (host, source) или (None, None) если не найдена
    """
    
    # Если пользователь явно указал host - использовать его
    if preferred_host:
        if check_ollama(preferred_host):
            # Определить источник по порту
            if ":11434" in preferred_host:
                source = 'local'
            elif ":11435" in preferred_host:
                source = 'docker'
            else:
                source = 'custom'
            
            print(f"[OLLAMA] Используется указанный host: {preferred_host} ({source})")
            return preferred_host, source
        else:
            print(f"[WARNING] Указанный host недоступен: {preferred_host}")
            print("[OLLAMA] Попытка автоопределения...")
    
    # Автоопределение
    return detect_ollama()