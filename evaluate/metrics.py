"""
Module for generating HTML reports (Full Analytics + Details)
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
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct", False))
    accuracy = (correct / total * 100) if total else 0.0
    
    avg_similarity = (
        sum(r.get("similarity", 0.0) for r in results) / total if total else 0.0
    )
    
    # --- MODIFIED LOGIC: Calculate MAX retrieval quality for the top card ---
    max_rag_quality = (
        max(r.get("retrieval_quality", 0.0) for r in results) if results else 0.0
    )

    # --- Analytics Calculation ---
    all_chunks = []
    for r in results:
        all_chunks.extend(r.get("retrieved_chunks", []))
    
    total_chunks = len(all_chunks)
    unique_sources = Counter(c.get("source", "unknown") for c in all_chunks)
    avg_chunk_score = sum(c.get("score", 0) for c in all_chunks) / total_chunks if total_chunks else 0.0
    
    # Chunk Quality Distribution
    high_qual = sum(1 for c in all_chunks if c.get("score", 0) >= 0.7)
    med_qual = sum(1 for c in all_chunks if 0.5 <= c.get("score", 0) < 0.7)
    low_qual = sum(1 for c in all_chunks if c.get("score", 0) < 0.5)
    
    # HTML Start
    html_head = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>TEST_LLM Report</title>
        <style>
            body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #f4f6f9; padding: 20px; color: #333; margin: 0; }}
            .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); overflow: hidden; }}
            .header {{ background: linear-gradient(135deg, #6b73ff 0%, #000dff 100%); padding: 40px; color: white; text-align: center; }}
            .header h1 {{ margin: 0; font-size: 32px; font-weight: 700; }}
            .header p {{ margin: 10px 0 0; opacity: 0.8; font-size: 14px; }}
            
            .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; padding: 30px 40px; border-bottom: 1px solid #eee; }}
            .stat-card {{ background: #fff; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #edf2f7; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }}
            .stat-val {{ font-size: 28px; font-weight: 800; color: #5a67d8; margin-bottom: 5px; }}
            .stat-label {{ color: #718096; font-size: 11px; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; }}
            
            .analytics-section {{ padding: 30px 40px; background: #f8f9fa; border-bottom: 1px solid #eee; }}
            .analytics-title {{ font-size: 20px; font-weight: 700; color: #2d3748; margin-bottom: 20px; border-left: 4px solid #5a67d8; padding-left: 15px; }}
            .analytics-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }}
            .analytics-card {{ background: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; }}
            .ac-title {{ font-size: 14px; font-weight: 600; color: #5a67d8; margin-bottom: 15px; text-transform: uppercase; }}
            
            .stat-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #f7fafc; font-size: 13px; }}
            .stat-row:last-child {{ border: none; }}
            
            .progress-bar {{ height: 24px; background: #edf2f7; border-radius: 12px; overflow: hidden; display: flex; margin-bottom: 10px; }}
            .pb-segment {{ height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-size: 11px; font-weight: 700; }}
            
            .chunk-table-wrapper {{ padding: 30px 40px; background: #fff; }}
            .chunk-table {{ width: 100%; border-collapse: separate; border-spacing: 0; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }}
            .chunk-table th {{ background: #5a67d8; color: white; padding: 12px 15px; text-align: left; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
            .chunk-table td {{ padding: 12px 15px; border-bottom: 1px solid #edf2f7; font-size: 13px; color: #4a5568; vertical-align: top; }}
            .chunk-table tr:last-child td {{ border-bottom: none; }}
            .chunk-table tr:nth-child(even) {{ background: #fcfcfc; }}
            
            .score-badge {{ padding: 4px 10px; border-radius: 12px; color: white; font-weight: 700; font-size: 11px; display: inline-block; }}
            .bg-high {{ background: #10b981; }}
            .bg-med {{ background: #f59e0b; }}
            .bg-low {{ background: #ef4444; }}
            
            .results-section {{ padding: 30px 40px; }}
            .results-table-main {{ width: 100%; border-collapse: separate; border-spacing: 0; }}
            .results-table-main th {{ background: #fff; padding: 15px; text-align: left; font-size: 12px; color: #a0aec0; text-transform: uppercase; border-bottom: 2px solid #edf2f7; }}
            .results-table-main td {{ padding: 15px; border-bottom: 1px solid #edf2f7; }}
            
            .row-main {{ cursor: pointer; transition: background 0.1s; }}
            .row-main:hover {{ background: #f7fafc; }}
            
            .row-details {{ display: none; background: #f8f9fa; }}
            .details-box {{ padding: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }}
            
            .col-left {{ border-right: 1px solid #e2e8f0; padding-right: 30px; }}
            .chunk-card {{ background: white; border: 1px solid #e2e8f0; padding: 12px; border-radius: 6px; margin-bottom: 15px; border-left: 3px solid #5a67d8; }}
            
            /* --- СТИЛЬ ДЛЯ БЛОКА ТЕКСТА (Окошко со скроллом) --- */
            .chunk-text-box {{
                /* Фиксированная максимальная высота. Текст внутри будет скроллиться */
                max-height: 180px; 
                overflow-y: auto; 
                
                /* Визуальное оформление "блока кода" */
                background-color: #f1f5f9; 
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 10px;
                margin-top: 8px;
                
                /* Шрифт и форматирование */
                font-family: Consolas, Monaco, 'Andale Mono', monospace; 
                font-size: 12px;
                line-height: 1.5;
                color: #334155;
                white-space: pre-wrap; /* Сохраняем переносы строк и абзацы */
                word-wrap: break-word; /* Переносим длинные слова */
            }}
            
            /* Кастомный скроллбар для красоты */
            .chunk-text-box::-webkit-scrollbar {{ width: 8px; }}
            .chunk-text-box::-webkit-scrollbar-track {{ background: #e2e8f0; border-radius: 4px; }}
            .chunk-text-box::-webkit-scrollbar-thumb {{ background: #94a3b8; border-radius: 4px; }}
            .chunk-text-box::-webkit-scrollbar-thumb:hover {{ background: #64748b; }}
            
            /* Общий контейнер, если чанков очень много */
            .chunks-scroll-container {{
                max-height: 800px;
                overflow-y: auto;
                padding-right: 10px;
            }}
            .chunks-scroll-container::-webkit-scrollbar {{ width: 8px; }}
            .chunks-scroll-container::-webkit-scrollbar-track {{ background: #edf2f7; }}
            .chunks-scroll-container::-webkit-scrollbar-thumb {{ background: #cbd5e0; border-radius: 4px; }}
            
            .answer-box {{ background: white; padding: 15px; border-radius: 6px; border: 1px solid #e2e8f0; margin-bottom: 15px; }}
            .ab-title {{ font-size: 11px; text-transform: uppercase; color: #718096; font-weight: 700; margin-bottom: 8px; }}
        </style>
        <script>
            function toggleRow(id) {{
                var row = document.getElementById('details-' + id);
                row.style.display = row.style.display === 'table-row' ? 'none' : 'table-row';
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>RAG Evaluation Report</h1>
                <p>Generated on: {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-val">{total}</div>
                    <div class="stat-label">Всего вопросов</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">{correct}</div>
                    <div class="stat-label">Правильных</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">{avg_similarity:.1%}</div>
                    <div class="stat-label">Средняя схожесть</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">{max_rag_quality:.1%}</div>
                    <div class="stat-label">Качество RAG</div>
                </div>
            </div>
        <!-- ANALYTICS SECTION -->
        <div class="analytics-section">
            <div class="analytics-title">Аналитика работы RAG системы</div>
            <div class="analytics-grid">
                <!-- Search Stats -->
                <div class="analytics-card">
                    <div class="ac-title">Статистика поиска</div>
                    <div class="stat-row"><span>Всего chunks найдено:</span> <b>{total_chunks}</b></div>
                    <div class="stat-row"><span>Chunks на вопрос:</span> <b>{total_chunks / total if total else 0:.1f}</b></div>
                    <div class="stat-row"><span>Средний score:</span> <b>{avg_chunk_score:.1%}</b></div>
                    <div class="stat-row"><span>Уникальных источников:</span> <b>{len(unique_sources)}</b></div>
                </div>
                
                <!-- Quality Distribution -->
                <div class="analytics-card">
                    <div class="ac-title">Распределение качества chunks</div>
                    <div class="progress-bar">
                        <div class="pb-segment bg-high" style="width: {(high_qual/total_chunks*100) if total_chunks else 0}%;"></div>
                        <div class="pb-segment bg-med" style="width: {(med_qual/total_chunks*100) if total_chunks else 0}%;"></div>
                        <div class="pb-segment bg-low" style="width: {(low_qual/total_chunks*100) if total_chunks else 0}%;"></div>
                    </div>
                    <div class="stat-row"><span>Высокое (≥70%):</span> <b>{high_qual}</b></div>
                    <div class="stat-row"><span>Среднее (50-70%):</span> <b>{med_qual}</b></div>
                    <div class="stat-row"><span>Низкое (&lt;50%):</span> <b>{low_qual}</b></div>
                </div>
                
                <!-- Sources -->
                <div class="analytics-card">
                    <div class="ac-title">Использование источников</div>
                    <div style="max-height: 120px; overflow-y: auto;">
"""
    # Добавляем источники
    for src, count in unique_sources.most_common(5):
        perc = (count / total_chunks * 100) if total_chunks else 0
        html_head += f"""
                        <div style="margin-bottom:8px;">
                            <div style="display:flex; justify-content:space-between; font-size:12px;">
                                <span>{src[:25]}...</span> <b>{count} ({perc:.0f}%)</b>
                            </div>
                            <div style="height:4px; background:#edf2f7; border-radius:2px; overflow:hidden;">
                                <div style="height:100%; background:#5a67d8; width:{perc}%;"></div>
                            </div>
                        </div>
        """
    
    html_head += """
                    </div>
                </div>
            </div>
        </div>
        <!-- DETAILED CHUNKS TABLE -->
        <div class="chunk-table-wrapper">
            <h3 style="color:#5a67d8; margin-top:0;">Детальная таблица всех найденных chunks</h3>
            <table class="chunk-table">
                <thead>
                    <tr>
                        <th style="width:40px;">#</th>
                        <th>Вопрос</th>
                        <th style="width:60px;">Ранг</th>
                        <th>Источник</th>
                        <th style="width:80px;">Score</th>
                        <th>Найденный Chunk</th>
                    </tr>
                </thead>
                <tbody>
"""
    # Генерируем строки таблицы чанков
    chunk_row_idx = 1
    for r in results:
        q_text = r.get("question", "")[:50] + "..."
        chunks = r.get("retrieved_chunks", [])
        
        for idx, c in enumerate(chunks, 1): 
            score = c.get("score", 0)
            badge_cls = "bg-high" if score >= 0.7 else ("bg-med" if score >= 0.5 else "bg-low")
            
            html_head += f"""
                    <tr>
                        <td>{chunk_row_idx}</td>
                        <td style="font-size:12px; color:#718096;">Q{results.index(r)+1}: {q_text}</td>
                        <td style="text-align:center;"><b>#{idx}</b></td>
                        <td style="color:#5a67d8;">{c.get('source', 'unknown')}</td>
                        <td><span class="score-badge {badge_cls}">{score:.1%}</span></td>
                        <td style="font-size:12px;">{c.get('text', '')[:150]}...</td>
                    </tr>
            """
            chunk_row_idx += 1
    html_head += """
                </tbody>
            </table>
        </div>
        <!-- MAIN RESULTS ACCORDION -->
        <div class="results-section">
            <h3 style="color:#2d3748;">Детальные результаты</h3>
            <table class="results-table-main">
                <thead>
                    <tr>
                        <th style="width:50px;">#</th>
                        <th>Вопрос</th>
                        <th style="width:100px;">Результат</th>
                        <th style="width:100px;">Схожесть</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    html_rows = ""
    for i, r in enumerate(results, 1):
        status_cls = "bg-high" if r.get("is_correct") else "bg-low"
        status_text = "Правильно" if r.get("is_correct") else "Ошибка"
        sim = r.get("similarity", 0.0)
        
        # Чанки для левой колонки
        chunks_html = ""
        for idx, c in enumerate(r.get("retrieved_chunks", []), 1):
            chunks_html += f"""
            <div class="chunk-card">
                <div style="display:flex; justify-content:space-between; font-size:11px; margin-bottom:5px; color:#718096;">
                    <b>#{idx} {c.get('source')}</b>
                    <span>{c.get('score', 0):.4f}</span>
                </div>
                <!-- ЗДЕСЬ ПРИМЕНЕН КЛАСС chunk-text-box (Серый блок со скроллом) -->
                <div class="chunk-text-box">{c.get('text', '')}</div>
            </div>
            """
        
        html_rows += f"""
            <tr class="row-main" onclick="toggleRow({i})">
                <td>{i}</td>
                <td style="font-weight:600;">{r.get("question")}</td>
                <td><span class="score-badge {status_cls}">{status_text}</span></td>
                <td><b>{sim:.1%}</b></td>
            </tr>
            <tr id="details-{i}" class="row-details">
                <td colspan="4" style="padding:0;">
                    <div class="details-box">
                        <div class="col-left">
                            <h4 style="color:#5a67d8; margin-top:0;">RAG Context (Top-{top_k})</h4>
                            <div class="chunks-scroll-container">
                                {chunks_html}
                            </div>
                        </div>
                        <div class="col-right">
                            <div class="answer-box" style="border-left: 4px solid #10b981; background:#f0fff4;">
                                <div class="ab-title">Ожидаемый ответ</div>
                                <div>{r.get("expected_answer")}</div>
                            </div>
                            <div class="answer-box" style="border-left: 4px solid #ed8936; background:#fffaf0;">
                                <div class="ab-title">Ответ системы</div>
                                <div>{r.get("generated_answer")}</div>
                            </div>
                            <div style="margin-top:10px; font-size:12px; color:#a0aec0;">
                                Reference Retrieval Quality: <b>{r.get('retrieval_quality', 0):.1%}</b>
                            </div>
                        </div>
                    </div>
                </td>
            </tr>
        """
    html_end = """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
    full_html = html_head + html_rows + html_end
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    
    print(f"[SUCCESS] Report saved: {output_path}")
