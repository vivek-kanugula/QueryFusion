import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from hybrid_rag.api.documents import router as documents_router
from hybrid_rag.api.evaluation import router as evaluation_router
from hybrid_rag.api.health import router as health_router
from hybrid_rag.api.jobs import router as jobs_router
from hybrid_rag.api.query import router as query_router
from hybrid_rag.api.traces import router as traces_router
from hybrid_rag.settings import settings


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

DIST_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    def _init_bm25():
        try:
            from concurrent.futures import ThreadPoolExecutor
            from hybrid_rag.db.engine import SessionLocal
            from hybrid_rag.retrieval.bm25_store import get_bm25_store

            store = get_bm25_store()
            strategies = ["fixed", "structure", "semantic"]

            def _rebuild_strat(strat: str):
                try:
                    with SessionLocal() as db:
                        store.rebuild_from_db(db, strat)
                except Exception:
                    logger.warning("Failed to initialize BM25 for strategy %s", strat, exc_info=True)

            with ThreadPoolExecutor(max_workers=3) as executor:
                list(executor.map(_rebuild_strat, strategies))
        except Exception:
            logger.error(
                "BM25 index initialization failed; sparse retrieval will be unavailable",
                exc_info=True,
            )

    await asyncio.to_thread(_init_bm25)

    from hybrid_rag.ingest.embedder import embed_query
    from hybrid_rag.retrieval.reranker import warmup_reranker

    try:
        await asyncio.to_thread(lambda: embed_query("warmup"))
    except Exception:
        logger.error("Embedder warmup failed; dense retrieval may cold-start", exc_info=True)

    try:
        await asyncio.to_thread(warmup_reranker)
    except Exception:
        logger.error("Reranker warmup failed; first rerank request will be slow", exc_info=True)

    from hybrid_rag.generation.cache import LLMCache

    try:
        LLMCache().evict_stale()
    except Exception:
        logger.warning("LLM cache eviction on startup failed", exc_info=True)

    yield


def create_app() -> FastAPI:
    settings.apply_cache_env()

    app = FastAPI(
        title="Hybrid RAG Intelligence & Benchmarking Platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=r"https://.*\.hf\.space" if settings.is_production else None,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(evaluation_router)
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
            clean_path = full_path.lstrip("/\\")
            if clean_path.startswith("api/") or clean_path == "api":
                raise HTTPException(status_code=404, detail="Not Found")
            try:
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
