from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.models.memory import Memory


class MemoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        user_id: int,
        conversation_id: int,
        message_id: int,
        content: str,
        embedding_id: str,
        importance_score: float,
    ) -> Memory:
        memory = Memory(
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            content=content,
            embedding_id=embedding_id,
            importance_score=importance_score,
        )
        self._session.add(memory)
        await self._session.commit()
        await self._session.refresh(memory)
        return memory

    async def list_by_user(self, user_id: int, limit: int | None = None) -> list[Memory]:
        statement = (
            select(Memory)
            .where(Memory.user_id == user_id)
            .order_by(Memory.id.desc())
        )
        if limit is not None:
            statement = statement.limit(limit)
        return list((await self._session.scalars(statement)).all())

    async def get_for_user(self, memory_id: int, user_id: int) -> Memory | None:
        statement = select(Memory).where(Memory.id == memory_id, Memory.user_id == user_id)
        return await self._session.scalar(statement)

    async def get_by_embedding_ids(self, embedding_ids: list[str]) -> list[Memory]:
        statement = select(Memory).where(Memory.embedding_id.in_(embedding_ids))
        return list((await self._session.scalars(statement)).all())

    async def delete(self, memory: Memory) -> None:
        await self._session.delete(memory)
        await self._session.commit()