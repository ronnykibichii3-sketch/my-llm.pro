"""Add memories table.

Revision ID: 002
Revises: 001
Create Date: 2025-01-01 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "memories",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding_id", sa.String(length=64), nullable=False),
        sa.Column("importance_score", sa.Float(), server_default="0.5", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name=op.f("fk_memories_user_id_users"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], name=op.f("fk_memories_conversation_id_conversations"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], name=op.f("fk_memories_message_id_messages"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_memories")),
    )
    op.create_index(op.f("ix_memories_user_id"), "memories", ["user_id"], unique=False)
    op.create_index(op.f("ix_memories_conversation_id"), "memories", ["conversation_id"], unique=False)
    op.create_index(op.f("ix_memories_message_id"), "memories", ["message_id"], unique=False)
    op.create_index(op.f("ix_memories_embedding_id"), "memories", ["embedding_id"], unique=True)


def downgrade() -> None:
    op.drop_index(op.f("ix_memories_embedding_id"), table_name="memories")
    op.drop_index(op.f("ix_memories_message_id"), table_name="memories")
    op.drop_index(op.f("ix_memories_conversation_id"), table_name="memories")
    op.drop_index(op.f("ix_memories_user_id"), table_name="memories")
    op.drop_table("memories")