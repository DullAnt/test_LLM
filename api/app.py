from fastapi import FastAPI

from api.deps import build_rag_service
from api.routes import rag_router, system_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="TEST_LLM API",
        version="0.1.0",
    )

    @app.on_event("startup")
    def startup():
        # Инициализируем один раз и кэшируем
        app.state.rag = build_rag_service()

    @app.on_event("shutdown")
    def shutdown():
        # при необходимости можно закрыть подключения
        pass

    app.include_router(system_router)
    app.include_router(rag_router)
    return app
