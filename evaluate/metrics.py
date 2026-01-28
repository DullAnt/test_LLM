# evaluate/metrics.py
"""
Модуль для генерации метрик и HTML отчетов
"""

from datetime import datetime
from typing import List, Dict
from collections import Counter


def generate_html_report(
    results: List[Dict],
    output_path: str,
    threshold: float = 0.7,
    model_name: str = "llama3",
    top_k: int = 5
):
    """Генерация HTML отчета с RAG аналитикой"""

    total = len(results)
    correct = sum(1 for r in results if r.get('is_correct', False))
    incorrect = total - correct
    accuracy = (correct / total * 100) if total > 0 else 0
    avg_similarity = sum(r.get('similarity', 0) for r in results) / total if total > 0 else 0

    # =============================================================================
    # ✅ НОВОЕ: RAG-аналитика считается по ЛУЧШЕМУ чанку на вопрос (best_chunk)
    # =============================================================================
    best_scores = []
    sources_used = []
    best_chunks = []  # для детальной таблицы (по 1 на вопрос)

    for r in results:
        best_chunk = r.get("best_chunk")
        best_score = float(r.get("best_chunk_score", 0.0) or 0.0)

        if best_chunk:
            best_scores.append(best_score)
            sources_used.append(best_chunk.get("source", "unknown"))
            best_chunks.append({
                "question": r.get("question", ""),
                "chunk": best_chunk,
                "score": best_score
            })

    # Качество RAG = средний best_score (а не среднее по всем chunks)
    avg_chunk_score = (sum(best_scores) / len(best_scores)) if best_scores else 0.0

    # Статистика по источникам (по лучшим чанкам)
    source_stats = Counter(sources_used)

    # Распределение scores (по лучшим чанкам)
    high_quality = sum(1 for s in best_scores if s >= 0.7)
    medium_quality = sum(1 for s in best_scores if 0.5 <= s < 0.7)
    low_quality = sum(1 for s in best_scores if s < 0.5)

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TEST_LLM Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1900px;
            margin: 0 auto;
            background: white;
            border-radius: 24px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 50px 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 3em;
            margin-bottom: 15px;
            font-weight: 800;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: 800;
            margin: 10px 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }}
        .section {{
            padding: 40px;
            border-bottom: 1px solid #e5e7eb;
        }}
        .section h2 {{
            font-size: 2em;
            margin-bottom: 30px;
            color: #333;
            font-weight: 700;
            border-left: 5px solid #667eea;
            padding-left: 15px;
        }}
        .rag-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }}
        .rag-card {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 12px;
            border: 2px solid #e5e7eb;
        }}
        .rag-card h3 {{
            color: #667eea;
            font-size: 1.2em;
            margin-bottom: 15px;
            font-weight: 700;
        }}
        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-top: 20px;
        }}
        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        th {{
            padding: 18px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 1px;
        }}
        td {{
            padding: 15px;
            border-bottom: 1px solid #e5e7eb;
            vertical-align: top;
        }}
        tr:hover {{
            background: #f9fafb;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        .bar {{
            height: 28px;
            background: #e5e7eb;
            border-radius: 14px;
            overflow: hidden;
            position: relative;
            min-width: 100px;
        }}
        .bar-fill {{
            height: 100%;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75em;
            font-weight: 700;
            color: white;
        }}
        .bar-high {{ background: linear-gradient(90deg, #10b981 0%, #059669 100%); }}
        .bar-medium {{ background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%); }}
        .bar-low {{ background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%); }}
        .chunk-source {{
            color: #7c3aed;
            font-weight: 600;
        }}
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e5e7eb;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-segment {{
            height: 100%;
            float: left;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75em;
            font-weight: 700;
            color: white;
            padding: 0 5px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>TEST_LLM Report</h1>
            <p style="font-size: 1.2em; margin-top: 10px;">{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</p>
            <p style="margin-top: 15px; opacity: 0.9;">Модель: {model_name} | TOP_K: {top_k} | Порог: {threshold:.0%}</p>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Всего вопросов</div>
                <div class="stat-value">{total}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Правильных</div>
                <div class="stat-value">{correct}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Неправильных</div>
                <div class="stat-value">{incorrect}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Точность</div>
                <div class="stat-value">{accuracy:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Средняя схожесть</div>
                <div class="stat-value">{avg_similarity:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Качество RAG</div>
                <div class="stat-value">{avg_chunk_score:.1%}</div>
            </div>
        </div>

        <div class="section" style="background: #f8f9fa;">
            <h2>Аналитика работы RAG системы</h2>

            <div class="rag-grid">
                <div class="rag-card">
                    <h3>Статистика поиска</h3>
                    <table style="box-shadow: none;">
                        <tr>
                            <td style="border: none;"><strong>Лучших chunks (по 1 на вопрос):</strong></td>
                            <td style="border: none;">{len(best_scores)}</td>
                        </tr>
                        <tr>
                            <td style="border: none;"><strong>Средний best score:</strong></td>
                            <td style="border: none;">{avg_chunk_score:.1%}</td>
                        </tr>
                        <tr>
                            <td style="border: none;"><strong>Уникальных источников:</strong></td>
                            <td style="border: none;">{len(source_stats)}</td>
                        </tr>
                    </table>
                </div>

                <div class="rag-card">
                    <h3>Распределение качества chunks</h3>
                    <div class="progress-bar">
                        <div class="progress-segment bar-high" style="width: {high_quality / len(best_scores) * 100 if best_scores else 0}%; background: linear-gradient(90deg, #10b981 0%, #059669 100%);">
                            {high_quality} высокое (≥70%)
                        </div>
                        <div class="progress-segment bar-medium" style="width: {medium_quality / len(best_scores) * 100 if best_scores else 0}%; background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);">
                            {medium_quality} среднее (50-70%)
                        </div>
                        <div class="progress-segment bar-low" style="width: {low_quality / len(best_scores) * 100 if best_scores else 0}%; background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);">
                            {low_quality} низкое (&lt;50%)
                        </div>
                    </div>
                    <p style="margin-top: 15px; font-size: 0.85em; color: #666;">
                        Высокое качество: {high_quality / len(best_scores) * 100 if best_scores else 0:.1f}%<br>
                        Среднее качество: {medium_quality / len(best_scores) * 100 if best_scores else 0:.1f}%<br>
                        Низкое качество: {low_quality / len(best_scores) * 100 if best_scores else 0:.1f}%
                    </p>
                </div>

                <div class="rag-card">
                    <h3>Использование источников</h3>
                    <div style="max-height: 300px; overflow-y: auto;">
"""

    # Добавить статистику по источникам
    for source, count in source_stats.most_common():
        percentage = (count / len(best_scores) * 100) if best_scores else 0
        html += f"""
                        <div style="margin: 10px 0;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span style="font-weight: 600; font-size: 0.9em;">{source}</span>
                                <span style="color: #667eea; font-weight: 700;">{count} ({percentage:.1f}%)</span>
                            </div>
                            <div class="bar">
                                <div class="bar-fill bar-high" style="width: {percentage}%;"></div>
                            </div>
                        </div>
"""

    html += """
                    </div>
                </div>
            </div>

            <h3 style="margin-top: 40px; margin-bottom: 20px; color: #667eea; font-size: 1.5em;">
                Детальная таблица лучших chunks (по 1 на вопрос)
            </h3>
            <table>
                <thead>
                    <tr>
                        <th style="width: 60px;">#</th>
                        <th style="width: 22%;">Вопрос</th>
                        <th style="width: 80px;">Ранг</th>
                        <th style="width: 18%;">Источник</th>
                        <th style="width: 100px;">Score</th>
                        <th>Найденный chunk</th>
                    </tr>
                </thead>
                <tbody>
"""

    for i, row in enumerate(best_chunks, 1):
        question = (row.get("question", "")[:50] + "...") if len(row.get("question", "")) > 50 else row.get("question", "")
        chunk = row.get("chunk", {}) or {}
        chunk_text_full = chunk.get("text", "") or ""
        chunk_text = chunk_text_full[:200] + ("..." if len(chunk_text_full) > 200 else "")
        chunk_text = chunk_text.replace('<', '&lt;').replace('>', '&gt;')
        chunk_source = chunk.get("source", "unknown")
        chunk_score = float(row.get("score", 0.0) or 0.0)
        chunk_rank = chunk.get("rank", 0)

        if chunk_score >= 0.7:
            bar_class = 'bar-high'
        elif chunk_score >= 0.5:
            bar_class = 'bar-medium'
        else:
            bar_class = 'bar-low'

        html += f"""
                    <tr>
                        <td><strong>{i}</strong></td>
                        <td style="font-size: 0.85em; color: #666;">Q{i}: {question}</td>
                        <td style="text-align: center;"><strong>#{chunk_rank}</strong></td>
                        <td><span class="chunk-source">{chunk_source}</span></td>
                        <td>
                            <div class="bar">
                                <div class="bar-fill {bar_class}" style="width: {chunk_score * 100}%; {''}">
                                    {chunk_score:.1%}
                                </div>
                            </div>
                        </td>
                        <td style="font-size: 0.85em;">{chunk_text}</td>
                    </tr>
"""

    # дальше — исходный блок "Детальные результаты" (оставляем без изменений)
    html += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Детальные результаты</h2>
            <table>
                <thead>
                    <tr>
                        <th style="width: 40px;">#</th>
                        <th style="width: 25%;">Вопрос</th>
                        <th style="width: 100px;">Результат</th>
                        <th style="width: 120px;">Схожесть</th>
                        <th>Сравнение</th>
                    </tr>
                </thead>
                <tbody>
"""

    for i, result in enumerate(results, 1):
        question = result.get('question', '')
        is_correct = result.get('is_correct', False)
        similarity = result.get('similarity', 0)
        generated = result.get('generated_answer', '')
        expected = result.get('expected_answer', '')

        question = question.replace('<', '&lt;').replace('>', '&gt;')
        generated_escaped = generated.replace('<', '&lt;').replace('>', '&gt;')
        expected_escaped = expected.replace('<', '&lt;').replace('>', '&gt;')

        status_class = 'status-correct' if is_correct else 'status-incorrect'
        status_text = 'Правильно' if is_correct else 'Неправильно'

        if similarity >= 0.7:
            bar_class = 'bar-high'
        elif similarity >= 0.5:
            bar_class = 'bar-medium'
        else:
            bar_class = 'bar-low'

        similarity_percent = similarity * 100

        html += f"""
                    <tr>
                        <td><strong>{i}</strong></td>
                        <td>{question}</td>
                        <td>{status_text}</td>
                        <td>
                            <div class="bar">
                                <div class="bar-fill {bar_class}" style="width: {similarity_percent}%;">
                                    {similarity:.1%}
                                </div>
                            </div>
                        </td>
                        <td>
                            <div style="padding: 10px; background: #f0fdf4; margin-bottom: 8px; border-left: 4px solid #10b981;">
                                <strong>Ожидаемый:</strong><br>{expected_escaped[:200]}{'...' if len(expected_escaped) > 200 else ''}
                            </div>
                            <div style="padding: 10px; background: #fef3c7; border-left: 4px solid #f59e0b;">
                                <strong>Сгенерированный:</strong><br>{generated_escaped[:200]}{'...' if len(generated_escaped) > 200 else ''}
                            </div>
                        </td>
                    </tr>
"""

    html += """
                </tbody>
            </table>
        </div>

        <div class="footer" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center;">
            <p style="font-size: 1.1em; font-weight: 600;">TEST_LLM - Система тестирования LLM</p>
            <p style="margin-top: 10px; opacity: 0.9;">Векторный поиск + Косинусное сходство</p>
        </div>
    </div>
</body>
</html>
"""

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"[SUCCESS] HTML отчет сохранен: {output_path}")
    except Exception as e:
        print(f"[FAIL] Ошибка при сохранении: {e}")
