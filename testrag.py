"""
Быстрое тестирование RAG
Запуск: python testrag.py [вопрос] [опции]
"""

import sys
import argparse
from llm.head import HeadRAG
from llm import head


# ДЕФОЛТНЫЙ ВОПРОС (можно менять здесь)

DEFAULT_QUERY = "Что такое СБП и как им пользоваться?"


def parse_args():
    parser = argparse.ArgumentParser(description="Быстрое тестирование RAG системы")
    
    # Необязательный аргумент с дефолтом
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    
    # RAG параметры
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-hyde", dest="hyde", action="store_false", default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--embeddings", type=str, default=None)
    
    # Опции вывода
    parser.add_argument("--show-docs", action="store_true")
    parser.add_argument("--show-sources", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not args.quiet:
        print("\n" + "=" * 80)
        print("RAG ТЕСТИРОВАНИЕ")
        print("=" * 80)
        print(f"\nВопрос: {args.query}")
        if args.query == DEFAULT_QUERY:
            print("(используется дефолтный вопрос)")
    
    # Выполняем запрос
    result = HeadRAG.run(
    query=args.query,
    top_k_param=args.top_k,
    need_hyde_param=args.hyde,
    model_param=args.model,
    embeddings_param=args.embeddings
    )

    
    # Вывод результатов
    if args.quiet:
        # Только ответ
        print(result.answer)
    else:
        # Полный вывод
        print("\n" + "=" * 80)
        print("ОТВЕТ:")
        print("=" * 80)
        print(result.answer)
        print("=" * 80)
        
        if args.show_docs:
            # Показать полные документы
            print(f"\nНАЙДЕНО ДОКУМЕНТОВ: {len(result.docs)}")
            print("=" * 80)
            for i, doc in enumerate(result.docs, 1):
                print(f"\n[{i}] Источник: {doc.source}")
                print("-" * 80)
                print(doc.text[:500] + "..." if len(doc.text) > 500 else doc.text)
                print("-" * 80)
        
        elif args.show_sources:
            # Показать только источники
            print(f"\nИСТОЧНИКИ ({len(result.docs)}):")
            for i, doc in enumerate(result.docs, 1):
                print(f"  [{i}] {doc.source}")
        
        print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())