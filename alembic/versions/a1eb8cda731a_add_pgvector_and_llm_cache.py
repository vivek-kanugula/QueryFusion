"""add_pgvector_and_llm_cache

Revision ID: a1eb8cda731a
Revises: 7b42d54a7b44
Create Date: 2026-11-04 19:10:46.244170

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1eb8cda731a'
down_revision: Union[str, None] = '7b42d54a7b44'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    from pgvector.sqlalchemy import Vector

    op.add_column(
        "document_chunks",
        sa.Column("embedding", Vector(384), nullable=True),
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops);"
    )
    op.create_table(
        "llm_cache",
        sa.Column("key", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("response", sa.Text(), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("created_at", sa.Float(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("llm_cache")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding;")
    op.drop_column("document_chunks", "embedding")
