"""Temporary live smoke test: QdrantVectorStore + MemoryService against a real Qdrant server."""

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database.base import Base
from app.database.models.chat import Conversation, Message
from app.database.models.user import User
from app.memory.services.memory_service import MemoryService
from app.memory.services.vector_store import QdrantVectorStore
from tests.conftest import FakeEmbeddingService


async def main() -> None:
    store = QdrantVectorStore(url="http://localhost:6333", collection_name="smoke_memories")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        user = User(email="smoke@example.com", username="smoke", hashed_password="x")
        session.add(user)
        await session.commit()
        await session.refresh(user)
        conversation = Conversation(user_id=user.id, title="smoke")
        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)
        first = Message(conversation_id=conversation.id, role="user", content="my cat is orange")
        second = Message(conversation_id=conversation.id, role="user", content="stocks rose today")
        session.add_all([first, second])
        await session.commit()
        await session.refresh(first)
        await session.refresh(second)

        service = MemoryService(session, FakeEmbeddingService(), store)

        result = await service.reindex_memory(user.id)
        assert result.created == 2 and result.reindexed == 0, result
        print("reindex create pass:", result)

        refresh = await service.reindex_memory(user.id)
        assert refresh.created == 0 and refresh.reindexed == 2, refresh
        print("reindex refresh pass:", refresh)

        found = await service.search_memory(user.id, "orange cat", 5)
        assert found, "search returned no results"
        assert "cat" in found[0].content, [hit.content for hit in found]
        assert found[0].score >= found[1].score
        print(f"search pass: top='{found[0].content}' score={found[0].score:.3f}")

        foreign = await store.search([0.1] * 32, 5, 9999)
        assert foreign == [], foreign
        print("user filter pass: foreign user sees nothing")

        memories = await service.list_memories(user.id)
        await service.delete_memory(user.id, memories[0].id)
        assert len(await service.list_memories(user.id)) == 1
        remaining = await store.search([0.1] * 32, 10, user.id)
        assert len(remaining) == 1, remaining
        print("delete pass: vector point and metadata removed")

    await engine.dispose()
    print("SMOKE OK")


if __name__ == "__main__":
    asyncio.run(main())
