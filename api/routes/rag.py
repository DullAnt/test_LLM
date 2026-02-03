from fastapi import APIRouter, Depends
from api.schemas import RagAskRequest, RagAskResponse, RagDocument
from api.ragser import RAGService, get_rag_service  # Импортируем сервис и зависимость

router = APIRouter(prefix="/rag", tags=["rag"])

@router.post("/ask", response_model=RagAskResponse)
def rag_ask(
    payload: RagAskRequest,
    rag_service: RAGService = Depends(get_rag_service) 
) -> RagAskResponse:
    
    # 1. Выполняем поиск и генерацию
    result = rag_service.ask(query=payload.query, top_k=payload.top_k, hyde=payload.hyde)
    
    # --- НОВОЕ: Логируем ответ в терминал ---
    print("\n" + "="*60)
    print(f"[QUERY] {payload.query}")
    print("-" * 20)
    # Выводим первые 300 символов ответа, чтобы не засорять консоль
    ans_preview = result["answer"][:300] + "..." if len(result["answer"]) > 300 else result["answer"]
    print(f"[ANSWER] {ans_preview}")
    print("="*60 + "\n")
    # ----------------------------------------

    chunks = result.get("chunks", [])
    docs = []
    for c in chunks:
        docs.append(
            RagDocument(
                text=c.get("text", ""),
                source=c.get("source", "unknown"),
                score=c.get("score"),
                rank=c.get("rank"),
                chunk_id=c.get("chunk_id"),
                total_chunks=c.get("total_chunks"),
            )
        )
    
    return RagAskResponse(answer=result["answer"], docs=docs, debug=True)

