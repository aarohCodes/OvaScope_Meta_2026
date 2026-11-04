"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-04
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "studies",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("retention_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("input_sha256", sa.String(64), nullable=False),
        sa.Column("image_format", sa.String(8), nullable=False),
        sa.Column("width", sa.Integer, nullable=False),
        sa.Column("height", sa.Integer, nullable=False),
        sa.Column("size_bytes", sa.Integer, nullable=False),
        sa.Column("image_key", sa.String(128), nullable=False),
        sa.Column("external_transmission_consent", sa.Boolean, nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_studies_retention_expires_at", "studies", ["retention_expires_at"])
    op.create_index("ix_studies_input_sha256", "studies", ["input_sha256"])
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("study_id", sa.String(32), sa.ForeignKey("studies.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("stage", sa.String(32)),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer, nullable=False, server_default="2"),
        sa.Column("idempotency_key", sa.String(128)),
        sa.Column("engine_fingerprint", sa.String(128)),
        sa.Column("cancel_requested", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("worker_id", sa.String(64)),
        sa.Column("error_code", sa.String(64)),
        sa.Column("error_message", sa.Text),
        sa.Column("request_id", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_jobs_study_id", "jobs", ["study_id"])
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_idempotency_key", "jobs", ["idempotency_key"])
    op.create_table(
        "results",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("study_id", sa.String(32), sa.ForeignKey("studies.id", ondelete="CASCADE"), nullable=False),
        sa.Column("job_id", sa.String(32), sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("payload", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_results_study_id", "results", ["study_id"])
    op.create_index("ix_results_job_id", "results", ["job_id"])
    op.create_table(
        "artifacts",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("study_id", sa.String(32), sa.ForeignKey("studies.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("storage_key", sa.String(128), nullable=False),
        sa.Column("media_type", sa.String(64), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_artifacts_study_id", "artifacts", ["study_id"])
    op.create_table(
        "annotations",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("study_id", sa.String(32), sa.ForeignKey("studies.id", ondelete="CASCADE"), nullable=False),
        sa.Column("polygon", sa.JSON, nullable=False),
        sa.Column("note", sa.String(500)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_annotations_study_id", "annotations", ["study_id"])
    op.create_table(
        "worker_heartbeats",
        sa.Column("worker_id", sa.String(64), primary_key=True),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=False),
        sa.Column("mode", sa.String(32), nullable=False),
        sa.Column("inline", sa.Boolean, nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    for t in ("worker_heartbeats", "annotations", "artifacts", "results", "jobs", "studies"):
        op.drop_table(t)
