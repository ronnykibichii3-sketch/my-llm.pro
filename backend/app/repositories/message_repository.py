from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.models.chat import Conversation, Message


class MessageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_message(self, conversation_id: int, role: str, content: str) -> Message:
        message = Message(conversation_id=conversation_id, role=role, content=content)
        self._session.add(message)
        await self._session.commit()
        await self._session.refresh(message)
        return message

    async def get_messages(self, conversation_id: int) -> list[Message]:
        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id.asc())
        )
        return list((await self._session.scalars(statement)).all())

    async def get_messages_for_user(self, user_id: int) -> list[Message]:
        """List every message the user owns, oldest first (used by memory reindexing)."""
        statement = (
            select(Message)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(Conversation.user_id == user_id)
            .order_by(Message.id.asc())
        )
        return list((await self._session.scalars(statement)).all())

    async def delete_message(self, conversation_id: int, message_id: int) -> bool:
        statement = select(Message).where(Message.id == message_id, Message.conversation_id == conversation_id)
        message = await self._session.scalar(statement)
        if message is None:
            return False
        await self._session.delete(message)
        await self._session.commit()
        return True