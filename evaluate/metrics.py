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
    
    # RAG АНАЛИТИКА
    all_chunks = []
    sources_used = []
    
    for r in results:
        chunks = r.get('retrieved_chunks', [])
        all_chunks.extend(chunks)
        for chunk in chunks:
            sources_used.append(chunk.get('source', 'unknown'))
    
    avg_chunk_score = sum(c.get('score', 0) for c in all_chunks) / len(all_chunks) if all_chunks else 0
    
    # Статистика по источникам
    source_stats = Counter(sources_used)
    
    # Распределение scores
    high_quality = sum(1 for c in all_chunks if c.get('score', 0) >= 0.7)
    medium_quality = sum(1 for c in all_chunks if 0.5 <= c.get('score', 0) < 0.7)
    low_quality = sum(1 for c in all_chunks if c.get('score', 0) < 0.5)
    
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
        .status {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 24px;
            font-weight: 700;
            font-size: 0.85em;
            white-space: nowrap;
        }}
        .status-correct {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
        }}
        .status-incorrect {{
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
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
        .text-wrap {{
            word-wrap: break-word;
            white-space: normal;
            line-height: 1.8;
            font-size: 1em;
        }}
        .answer-box {{
            padding: 20px;
            border-radius: 12px;
            margin: 10px 0;
            line-height: 1.8;
            font-size: 1em;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }}
        .expected-answer {{
            background: #f0fdf4;
            border-left: 4px solid #10b981;
        }}
        .generated-answer {{
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
        }}
        .answer-label {{
            font-weight: 700;
            font-size: 0.75em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
            color: #666;
        }}
        .chunk-box {{
            background: #f3f4f6;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 0 0 15px 0;
            border-radius: 12px;
            font-size: 0.95em;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        .chunk-meta {{
            font-size: 0.85em;
            color: #666;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .chunk-score {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.9em;
            color: white;
        }}
        .chunk-source {{
            color: #7c3aed;
            font-weight: 600;
        }}
        .chunk-text {{
            line-height: 1.8;
            color: #333;
            font-size: 1em;
            margin-top: 10px;
            white-space: pre-wrap;
        }}
        .source-badge {{
            display: inline-block;
            padding: 4px 12px;
            background: #667eea;
            color: white;
            border-radius: 16px;
            font-size: 0.85em;
            margin: 5px;
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
        .expandable {{
            cursor: pointer;
            user-select: none;
        }}
        .expandable:hover {{
            background: #f9fafb;
        }}
        .details {{
            display: none;
            padding: 20px;
            background: #fafafa;
            border-top: 1px solid #e5e7eb;
        }}
        .details.open {{
            display: table-row;
        }}
        .footer {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
            font-size: 0.95em;
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
                            <td style="border: none;"><strong>Всего chunks найдено:</strong></td>
                            <td style="border: none;">{len(all_chunks)}</td>
                        </tr>
                        <tr>
                            <td style="border: none;"><strong>Chunks на вопрос:</strong></td>
                            <td style="border: none;">{len(all_chunks) / total if total > 0 else 0:.1f}</td>
                        </tr>
                        <tr>
                            <td style="border: none;"><strong>Средний score:</strong></td>
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
                        <div class="progress-segment bar-high" style="width: {high_quality / len(all_chunks) * 100 if all_chunks else 0}%;">
                            {high_quality} высокое (≥70%)
                        </div>
                        <div class="progress-segment bar-medium" style="width: {medium_quality / len(all_chunks) * 100 if all_chunks else 0}%;">
                            {medium_quality} среднее (50-70%)
                        </div>
                        <div class="progress-segment bar-low" style="width: {low_quality / len(all_chunks) * 100 if all_chunks else 0}%;">
                            {low_quality} низкое (<50%)
                        </div>
                    </div>
                    <p style="margin-top: 15px; font-size: 0.85em; color: #666;">
                        Высокое качество: {high_quality / len(all_chunks) * 100 if all_chunks else 0:.1f}%<br>
                        Среднее качество: {medium_quality / len(all_chunks) * 100 if all_chunks else 0:.1f}%<br>
                        Низкое качество: {low_quality / len(all_chunks) * 100 if all_chunks else 0:.1f}%
                    </p>
                </div>
                
                <div class="rag-card">
                    <h3>Использование источников</h3>
                    <div style="max-height: 300px; overflow-y: auto;">
"""
    
    # Добавить статистику по источникам
    for source, count in source_stats.most_common():
        percentage = (count / len(all_chunks) * 100) if all_chunks else 0
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
                Детальная таблица всех найденных chunks
            </h3>
            <table>
                <thead>
                    <tr>
                        <th style="width: 60px;">#</th>
                        <th style="width: 15%;">Вопрос</th>
                        <th style="width: 80px;">Ранг</th>
                        <th style="width: 15%;">Источник</th>
                        <th style="width: 100px;">Score</th>
                        <th>Найденный chunk</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Добавить детали всех chunks
    chunk_counter = 1
    for i, result in enumerate(results, 1):
        question = result.get('question', '')[:50] + '...'
        chunks = result.get('retrieved_chunks', [])
        
        for chunk in chunks:
            chunk_text = chunk.get('text', '')[:200] + ('...' if len(chunk.get('text', '')) > 200 else '')
            chunk_text = chunk_text.replace('<', '&lt;').replace('>', '&gt;')
            chunk_source = chunk.get('source', 'unknown')
            chunk_score = chunk.get('score', 0)
            chunk_rank = chunk.get('rank', 0)
            
            if chunk_score >= 0.7:
                bar_class = 'bar-high'
            elif chunk_score >= 0.5:
                bar_class = 'bar-medium'
            else:
                bar_class = 'bar-low'
            
            html += f"""
                    <tr>
                        <td><strong>{chunk_counter}</strong></td>
                        <td style="font-size: 0.85em; color: #666;">Q{i}: {question}</td>
                        <td style="text-align: center;"><strong>#{chunk_rank}</strong></td>
                        <td><span class="chunk-source">{chunk_source}</span></td>
                        <td>
                            <div class="bar">
                                <div class="bar-fill {bar_class}" style="width: {chunk_score * 100}%;">
                                    {chunk_score:.1%}
                                </div>
                            </div>
                        </td>
                        <td style="font-size: 0.85em;">{chunk_text}</td>
                    </tr>
"""
            chunk_counter += 1
    
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
        chunks = result.get('retrieved_chunks', [])
        
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
        
        # Генерация блока с chunks
        chunks_html = ""
        for chunk in chunks:
            chunk_text = chunk.get('text', '').replace('<', '&lt;').replace('>', '&gt;')
            chunk_source = chunk.get('source', 'unknown')
            chunk_score = chunk.get('score', 0)
            chunk_rank = chunk.get('rank', 0)
            
            # Цвет badge в зависимости от score
            if chunk_score >= 0.8:
                score_color = '#10b981'
                score_label = 'Отлично'
            elif chunk_score >= 0.6:
                score_color = '#f59e0b'
                score_label = 'Хорошо'
            else:
                score_color = '#ef4444'
                score_label = 'Средне'
            
            chunks_html += f"""
                <div class="chunk-box">
                    <div class="chunk-meta">
                        <div>
                            <strong style="color: #667eea;">Ранг #{chunk_rank}</strong>
                            <span style="color: #666; margin: 0 10px;">|</span>
                            <span class="chunk-source">{chunk_source}</span>
                        </div>
                        <span class="chunk-score" style="background: {score_color};">
                            {chunk_score:.1%} - {score_label}
                        </span>
                    </div>
                    <div class="chunk-text">{chunk_text}</div>
                </div>
            """
        
        html += f"""
                    <tr class="expandable" onclick="this.nextElementSibling.classList.toggle('open')">
                        <td><strong>{i}</strong></td>
                        <td><div class="text-wrap">{question}</div></td>
                        <td><span class="status {status_class}">{status_text}</span></td>
                        <td>
                            <div class="bar">
                                <div class="bar-fill {bar_class}" style="width: {similarity_percent}%">
                                    {similarity:.1%}
                                </div>
                            </div>
                        </td>
                        <td>
                            <div class="answer-box expected-answer">
                                <div class="answer-label">Ожидаемый ответ</div>
                                <div class="text-wrap">{expected_escaped[:200]}{'...' if len(expected_escaped) > 200 else ''}</div>
                            </div>
                            <div class="answer-box generated-answer">
                                <div class="answer-label">Ответ системы (клик для деталей)</div>
                                <div class="text-wrap">{generated_escaped[:200]}{'...' if len(generated_escaped) > 200 else ''}</div>
                            </div>
                        </td>
                    </tr>
                    <tr class="details">
                        <td colspan="5" style="padding: 20px; background: #fafafa;">
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; align-items: start;">
                                
                                <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                                    <h4 style="margin: 0 0 20px 0; color: #667eea; font-size: 1.4em; border-bottom: 2px solid #667eea; padding-bottom: 10px;">
                                        RAG: Найденные чанки (TOP {len(chunks)})
                                    </h4>
                                    <div style="display: grid; grid-template-columns: 1fr; gap: 15px; max-height: 600px; overflow-y: auto; padding-right: 10px;">
                                        {chunks_html if chunks_html else '<p style="color: #666; text-align: center; padding: 40px;">Чанки не найдены</p>'}
                                    </div>
                                </div>
                                
                                <div>
                            <!-- Сначала Ожидаемый ответ (зеленый) -->
                                    <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px;">
                                        <h4 style="margin: 0 0 15px 0; color: #10b981; font-size: 1.4em; border-bottom: 2px solid #10b981; padding-bottom: 10px;">
                                            Ожидаемый ответ
                                        </h4>
                                        <div style="background: #f0fdf4; padding: 20px; border-radius: 8px; border-left: 4px solid #10b981; line-height: 1.8; white-space: pre-wrap; font-size: 1em;">
                                    {expected_escaped}
                                        </div>
                                    </div>

                                    <!-- Потом Полный ответ системы (желтый) -->
                                    <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                                        <h4 style="margin: 0 0 15px 0; color: #f59e0b; font-size: 1.4em; border-bottom: 2px solid #f59e0b; padding-bottom: 10px;">
                                            Полный ответ системы
                                        </h4>
                                        <div style="background: #fef3c7; padding: 20px; border-radius: 8px; border-left: 4px solid #f59e0b; line-height: 1.8; white-space: pre-wrap; font-size: 1em;">
                                    {generated_escaped}
                                        </div>
                                    </div>
                                </div>
                                
                            </div>
                        </td>
                    </tr>
"""
    
    html += """
                </tbody>
            </table>
            <p style="margin-top: 20px; color: #666; font-size: 0.9em;">
                Нажмите на строку чтобы развернуть детали RAG поиска
            </p>
        </div>
        
        <div class="footer">
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