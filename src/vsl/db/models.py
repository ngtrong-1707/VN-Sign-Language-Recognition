"""
Database models for VSL Recognition Platform using SQLAlchemy 2.0.
Supports PostgreSQL and SQLite asynchronously.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import (
    String, Text, Boolean, Integer, Float, DateTime, ForeignKey
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    icon: Mapped[str] = mapped_column(String(20), default="📁")

    vocabularies: Mapped[List["Vocabulary"]] = relationship("Vocabulary", back_populates="category")


class Vocabulary(Base):
    __tablename__ = "vocabularies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("categories.id", ondelete="SET NULL"), nullable=True)
    action_key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(150), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    icon: Mapped[str] = mapped_column(String(20), default="👋")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    category: Mapped[Optional[Category]] = relationship("Category", back_populates="vocabularies")
    samples: Mapped[List["SignSample"]] = relationship("SignSample", back_populates="vocabulary")


class SignSample(Base):
    __tablename__ = "sign_samples"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    vocabulary_id: Mapped[int] = mapped_column(Integer, ForeignKey("vocabularies.id", ondelete="CASCADE"), nullable=False)
    contributor_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, default="Anonymous")
    sample_path: Mapped[str] = mapped_column(String(255), nullable=False)
    frames_count: Mapped[int] = mapped_column(Integer, default=30)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    vocabulary: Mapped[Vocabulary] = relationship("Vocabulary", back_populates="samples")


class InferenceSession(Base):
    __tablename__ = "inference_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=0.0)
    client_ip: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    device_info: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)

    feedback_records: Mapped[List["FeedbackRecord"]] = relationship("FeedbackRecord", back_populates="session")


class FeedbackRecord(Base):
    __tablename__ = "feedback_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("inference_sessions.id", ondelete="CASCADE"), nullable=True)
    predicted_action: Mapped[str] = mapped_column(String(100), nullable=False)
    actual_action: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    session: Mapped[Optional[InferenceSession]] = relationship("InferenceSession", back_populates="feedback_records")
