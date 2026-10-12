"""
CRUD repository methods for database operations.
"""

from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from vsl.db.models import Category, Vocabulary, InferenceSession, FeedbackRecord, SignSample


class VocabularyRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all(self, active_only: bool = True) -> List[Vocabulary]:
        query = select(Vocabulary).options(selectinload(Vocabulary.category))
        if active_only:
            query = query.where(Vocabulary.is_active.is_(True))
        result = await self.session.execute(query.order_by(Vocabulary.id))
        return list(result.scalars().all())

    async def get_by_key(self, action_key: str) -> Optional[Vocabulary]:
        result = await self.session.execute(
            select(Vocabulary).where(Vocabulary.action_key == action_key).options(selectinload(Vocabulary.category))
        )
        return result.scalars().first()


class FeedbackRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        predicted_action: str,
        actual_action: str,
        confidence: float,
        session_id: Optional[str] = None,
        notes: Optional[str] = None
    ) -> FeedbackRecord:
        feedback = FeedbackRecord(
            predicted_action=predicted_action,
            actual_action=actual_action,
            confidence=confidence,
            session_id=session_id,
            notes=notes
        )
        self.session.add(feedback)
        await self.session.commit()
        await self.session.refresh(feedback)
        return feedback
