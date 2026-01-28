from fastapi import APIRouter, Request, HTTPException

from api.schemas import RagAskRequest, RagAskResponse, RagDocument

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ask", response_model=RagAskResponse)
def rag_ask(payload: RagAskRequest, request: Request) -> RagAskResponse:
    svc = getattr(request.app.state, "rag", None)
    if svc is None:
        raise HTTPException(status_code=500, detail="RAG service is not initialized")

    result = svc.ask(query=payload.query, top_k=payload.top_k, hyde=payload.hyde)
    chunks = result["chunks"]

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
