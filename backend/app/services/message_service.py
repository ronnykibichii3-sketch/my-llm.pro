from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.models.chat import Message
from app.memory.services.memory_service import MemoryService
from app.repositories.chat_repository import ChatRepository
from app.repositories.message_repository import MessageRepository


class MessageService:
    def __init__(self, session: AsyncSession, memory: MemoryService) -> None:
        self._chats = ChatRepository(session)
        self._messages = MessageRepository(session)
        self._memory = memory

    async def list_messages(self, user_id: int, conversation_id: int) -> list[Message]:
        await self._require_conversation(conversation_id, user_id)
        return await self._messages.get_messages(conversation_id)

    async def create_user_message(self, user_id: int, conversation_id: int, content: str) -> Message:
        await self._require_conversation(conversation_id, user_id)
        message = await self._messages.create_message(conversation_id, Message.ROLE_USER, content)
        await self._memory.try_create_memory(user_id, conversation_id, message.id, message.content)
        return message

    async def delete_message(self, user_id: int, conversation_id: int, message_id: int) -> None:
        await self._require_conversation(conversation_id, user_id)
        deleted = await self._messages.delete_message(conversation_id, message_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")

    async def _require_conversation(self, conversation_id: int, user_id: int) -> None:
        conversation = await self._chats.get_conversation(conversation_id, user_id)
        if conversation is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")