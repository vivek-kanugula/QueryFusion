import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from hybrid_rag.models.base import Base, new_uuid


class GoldenQuestion(Base):
    __tablename__ = "golden_questions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_uuid)
    question: Mapped[str] = mapped_column(Text)
    expected_answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(30))
    ground_truth_spans: Mapped[str] = mapped_column(Text)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_uuid)
    experiment_name: Mapped[str] = mapped_column(String(200))
    retrieval_strategy: Mapped[str] = mapped_column(String(30))
    chunking_strategy: Mapped[str] = mapped_column(String(20))
    config_hash: Mapped[str] = mapped_column(String(64))
    corpus_snapshot_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latency_breakdown_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )

    metrics: Mapped[list["EvaluationMetric"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )


class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_uuid)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("evaluation_runs.id", ondelete="CASCADE"))
    precision_at_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_at_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    mrr: Mapped[float | None] = mapped_column(Float, nullable=True)
    ndcg_at_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    faithfulness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    relevancy_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    citation_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    citation_coverage: Mapped[float | None] = mapped_column(Float, nullable=True)

    run: Mapped["EvaluationRun"] = relationship(back_populates="metrics")
