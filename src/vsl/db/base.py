"""
Async Database Session & Initialization Layer.
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select
from vsl.config import settings, ACTIONS_METADATA
from vsl.db.models import Base, Category, Vocabulary

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True
)

async_session_factory = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency injection for FastAPI routes."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Creates database tables and seeds initial 20 VSL signs."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed initial vocabulary data
    async with async_session_factory() as session:
        async with session.begin():
            # Check if vocabularies exist
            result = await session.execute(select(Vocabulary).limit(1))
            first_vocab = result.scalars().first()

            if not first_vocab:
                print("[DB] Khởi tạo hạt giống (Seed Data) cho 20 ký hiệu VSL...")
                # Map categories
                cat_map = {}
                for item in ACTIONS_METADATA:
                    cat_name = item.get("category", "Chung")
                    if cat_name not in cat_map:
                        cat = Category(name=cat_name, icon=item.get("icon", "📁"))
                        session.add(cat)
                        cat_map[cat_name] = cat

                await session.flush()

                # Add vocabularies
                for item in ACTIONS_METADATA:
                    cat = cat_map.get(item.get("category", "Chung"))
                    vocab = Vocabulary(
                        category_id=cat.id if cat else None,
                        action_key=item["name"],
                        display_name=item["display"],
                        description=item.get("desc", ""),
                        icon=item.get("icon", "👋")
                    )
                    session.add(vocab)

                print("[DB] Hoàn tất nạp dữ liệu từ điển ban đầu!")
