"""Temporary live outage check: chat ingestion must survive a dead vector store."""

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database.base import Base
from app.database.models.chat import Conversation, Message
from app.database.models.user import User
from app.memory.services.memory_service import MemoryService
from app.memory.services.vector_store import QdrantVectorStore
from tests.conftest import FakeEmbeddingService


async def main() -> None:
    store = QdrantVectorStore(url="http://localhost:6399", collection_name="smoke_down")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        user = User(email="down@example.com", username="down", hashed_password="x")
        session.add(user)
        await session.commit()
        await session.refresh(user)
        conversation = Conversation(user_id=user.id, title="down")
        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)
        message = Message(conversation_id=conversation.id, role="user", content="hello there")
        session.add(message)
        await session.commit()
        await session.refresh(message)

        service = MemoryService(session, FakeEmbeddingService(), store)
        swallowed = await service.try_create_memory(user.id, conversation.id, message.id, "hello there")
        assert swallowed is None, swallowed
        assert await service.list_memories(user.id) == []
        print("OUTAGE OK: chat-path ingestion degraded gracefully, no metadata written")

        try:
            await service.create_memory(user.id, conversation.id, message.id, "hello there")
        except Exception as exc:  # noqa: BLE001 - intentionally asserting the raised type
            print(f"explicit create_memory surfaced: {type(exc).__module__}.{type(exc).__name__}")
        else:
            raise AssertionError("expected create_memory to raise when Qdrant is down")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
