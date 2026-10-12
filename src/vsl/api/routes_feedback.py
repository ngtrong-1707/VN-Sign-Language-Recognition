"""
REST API routes for User Feedback and Active Learning correction submission.
"""

from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from vsl.db.base import get_db
from vsl.db.repositories import FeedbackRepository

router = APIRouter(prefix="/api/feedback", tags=["Feedback"])


class FeedbackCreateRequest(BaseModel):
    predicted_action: str = Field(..., description="Cử chỉ do AI dự đoán")
    actual_action: str = Field(..., description="Cử chỉ thực tế chính xác")
    confidence: float = Field(0.0, description="Độ tin cậy của AI lúc nhận diện")
    session_id: Optional[str] = Field(None, description="Mã phiên nhận diện")
    notes: Optional[str] = Field(None, description="Ghi chú đóng góp bổ sung")


@router.post("")
async def submit_feedback(payload: FeedbackCreateRequest, db: AsyncSession = Depends(get_db)):
    """Receives user correction for active learning data collection."""
    repo = FeedbackRepository(db)
    record = await repo.create(
        predicted_action=payload.predicted_action,
        actual_action=payload.actual_action,
        confidence=payload.confidence,
        session_id=payload.session_id,
        notes=payload.notes
    )
    return {
        "status": "success",
        "message": "Cảm ơn bạn đã đóng góp phản hồi để cải thiện mô hình!",
        "feedback_id": record.id
    }
