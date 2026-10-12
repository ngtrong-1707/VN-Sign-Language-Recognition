"""
REST API routes for VSL Dictionary and Vocabulary management.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from vsl.db.base import get_db
from vsl.db.repositories import VocabularyRepository

router = APIRouter(prefix="/api/dictionary", tags=["Dictionary"])


@router.get("")
async def get_all_vocabularies(db: AsyncSession = Depends(get_db)):
    """Returns catalog of sign language vocabulary with category metadata."""
    repo = VocabularyRepository(db)
    items = await repo.get_all()
    return {
        "status": "success",
        "total": len(items),
        "data": [
            {
                "id": v.id,
                "name": v.action_key,
                "display": v.display_name,
                "category": v.category.name if v.category else "Chung",
                "desc": v.description,
                "icon": v.icon
            }
            for v in items
        ]
    }


@router.get("/{action_key}")
async def get_vocabulary_by_key(action_key: str, db: AsyncSession = Depends(get_db)):
    """Returns single vocabulary entry details."""
    repo = VocabularyRepository(db)
    item = await repo.get_by_key(action_key)
    if not item:
        raise HTTPException(status_code=404, detail=f"Cử chỉ '{action_key}' không tồn tại trong từ điển.")
    return {
        "status": "success",
        "data": {
            "id": item.id,
            "name": item.action_key,
            "display": item.display_name,
            "category": item.category.name if item.category else "Chung",
            "desc": item.description,
            "icon": item.icon
        }
    }
