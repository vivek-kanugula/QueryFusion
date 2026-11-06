from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from hybrid_rag.api.documents import router as documents_router
from hybrid_rag.api.health import router as health_router
from hybrid_rag.api.jobs import router as jobs_router
from hybrid_rag.api.query import router as query_router
from hybrid_rag.api.traces import router as traces_router
from hybrid_rag.settings import settings

DIST_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    def _init_bm25():
        try:
            from hybrid_rag.db.engine import SessionLocal
            from hybrid_rag.retrieval.bm25_store import get_bm25_store

            store = get_bm25_store()
            with SessionLocal() as db:
                for strat in ["fixed", "structure", "semantic"]:
                    store.rebuild_from_db(db, strat)
        except Exception:
            pass

    await asyncio.to_thread(_init_bm25)

    if settings.is_production:
        from hybrid_rag.retrieval.reranker import warmup_reranker

        await asyncio.to_thread(warmup_reranker)
    yield


def create_app() -> FastAPI:
    settings.apply_cache_env()

    app = FastAPI(
        title="Hybrid RAG Intelligence & Benchmarking Platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(jobs_router)
    app.include_router(query_router)
    app.include_router(traces_router)

    dist_dir = DIST_DIR
    if dist_dir.is_dir():
        assets_dir = dist_dir / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="static-assets")

        index_html = dist_dir / "index.html"
        resolved_dist = dist_dir.resolve()

        @app.get("/{full_path:path}")
        async def spa_fallback(request: Request, full_path: str):
            try:
                clean_path = full_path.lstrip("/\\")
                resolved_file = (dist_dir / clean_path).resolve()
                if (
                    clean_path
                    and resolved_file.is_file()
                    and resolved_file.is_relative_to(resolved_dist)
                ):
                    return FileResponse(resolved_file)
            except (ValueError, OSError, RuntimeError):
                pass
            return FileResponse(index_html)

    return app


app = create_app()
