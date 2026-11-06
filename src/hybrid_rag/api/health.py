import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from hybrid_rag.db.engine import check_db_connection

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/health/db")
def health_db():
    connected = check_db_connection()
    status = "ok" if connected else "unavailable"
    code = 200 if connected else 503
    return JSONResponse(content={"status": status, "database": connected}, status_code=code)


@router.get("/ready")
def readiness():
    checks: dict[str, bool] = {}

    checks["database"] = check_db_connection()

    try:
        from hybrid_rag.ingest.embedder import get_embedder

        get_embedder()
        checks["embedding_model"] = True
    except Exception:
        logger.warning("Embedding model readiness check failed", exc_info=True)
        checks["embedding_model"] = False

    all_ready = all(checks.values())
    return JSONResponse(
        content={"ready": all_ready, "checks": checks},
        status_code=200 if all_ready else 503,
    )
