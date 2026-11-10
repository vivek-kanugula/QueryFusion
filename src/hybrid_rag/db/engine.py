from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from hybrid_rag.settings import settings

connect_args = {}
if "psycopg" in settings.database_url or "postgres" in settings.database_url:
    # Explicitly disable client-side prepared statements for Supabase/PgBouncer pooler compatibility
    connect_args["prepare_threshold"] = 0

engine = create_engine(
    settings.database_url,
    pool_pre_ping=settings.db_pool_pre_ping,
    pool_recycle=settings.db_pool_recycle,
    pool_size=settings.db_pool_size,
    connect_args=connect_args,
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_db() -> Session:
    """Dependency for FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> bool:
    """Return True if database is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
