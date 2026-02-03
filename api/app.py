from contextlib import asynccontextmanager
import json
from fastapi import FastAPI, Request, status
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from api.ragser import build_rag_service

# Импортируем роуты
from api.routes.rag import router as rag_router
from api.routes.system import router as system_router

# =======================================================
# 1. MIDDLEWARE ДЛЯ ЛОГИРОВАНИЯ (Показывает тело запроса)
# =======================================================
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Логируем только POST запросы (где есть тело)
        if request.method == "POST":
            try:
                # ВАЖНО: Читаем тело и возвращаем его обратно, иначе FastAPI зависнет
                body_bytes = await request.body()
                request._body = body_bytes  # Возвращаем тело в запрос
                
                body_str = body_bytes.decode("utf-8")
                print(f"\n[REQUEST] {request.method} {request.url}")
                if body_str:
                    print(f"[BODY] {body_str}")
                else:
                    print(f"[BODY] <EMPTY>")
            except Exception as e:
                print(f"[LOG ERROR] Не удалось прочитать тело: {e}")
        else:
            # Для GET запросов просто пишем URL
            print(f"\n[REQUEST] {request.method} {request.url}")

        response = await call_next(request)
        
        # Логируем статус ответа
        if response.status_code >= 400:
            print(f"[RESPONSE] ERROR Status: {response.status_code}")
        else:
            print(f"[RESPONSE] OK Status: {response.status_code}")
            
        return response

# =======================================================
# 2. LIFESPAN (Запуск и остановка)
# =======================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[API] Initializing RAG Service...")
    try:
        app.state.rag = build_rag_service()
        print("[API] RAG Service ready.")
    except Exception as e:
        print(f"[API] CRITICAL ERROR: {e}")
    
    yield
    
    print("[API] Shutting down...")

# =======================================================
# 3. CREATE APP
# =======================================================
def create_app() -> FastAPI:
    app = FastAPI(
        title="TEST_LLM API",
        version="0.1.0",
        lifespan=lifespan
    )

    # Добавляем наш логгер
    app.add_middleware(LoggingMiddleware)

    # --- Обработчик ошибок валидации (чтобы видеть детали 422 в консоли) ---
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        error_details = exc.errors()
        print(f"\n[VALIDATION ERROR 422] Детали ошибки:\n{json.dumps(error_details, indent=2, ensure_ascii=False)}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": error_details},
        )

    # Редирект на документацию
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    # Подключаем роуты
    app.include_router(system_router)
    app.include_router(rag_router)

    return app

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    uvicorn.run("api.app:create_app", host="127.0.0.1", port=8000, reload=True, factory=True)
