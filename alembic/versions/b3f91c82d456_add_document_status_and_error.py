"""add document status and error message

Revision ID: b3f91c82d456
Revises: a1eb8cda731a
Create Date: 2026-11-10 18:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3f91c82d456'
down_revision: Union[str, None] = 'a1eb8cda731a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'documents',
        sa.Column('status', sa.String(length=20), server_default='ready', nullable=False),
    )
    op.add_column(
        'documents',
        sa.Column('error_message', sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('documents', 'error_message')
    op.drop_column('documents', 'status')

