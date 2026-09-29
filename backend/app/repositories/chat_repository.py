from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.models.chat import Conversation


class ChatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_conversations(self, user_id: int) -> list[Conversation]:
        statement = select(Conversation).where(Conversation.user_id == user_id).order_by(Conversation.updated_at.desc())
        return list((await self._session.scalars(statement)).all())

    async def get_conversation(self, conversation_id: int, user_id: int) -> Conversation | None:
        return await self._session.scalar(select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id))

    async def create_conversation(self, user_id: int, title: str) -> Conversation:
        conversation = Conversation(user_id=user_id, title=title)
        self._session.add(conversation)
        await self._session.commit()
        await self._session.refresh(conversation)
        return conversation
