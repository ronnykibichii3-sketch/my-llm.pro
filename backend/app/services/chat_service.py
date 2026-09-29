from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.models.chat import Conversation, Message
from app.memory.services.memory_service import MemoryService
from app.repositories.chat_repository import ChatRepository
from app.repositories.message_repository import MessageRepository

PLACEHOLDER_RESPONSE = "temporary placeholder"


class ChatService:
    def __init__(self, session: AsyncSession, memory: MemoryService) -> None:
        self._chats = ChatRepository(session)
        self._messages = MessageRepository(session)
        self._memory = memory

    async def list_conversations(self, user_id: int) -> list[Conversation]:
        return await self._chats.list_conversations(user_id)

    async def create_conversation(self, user_id: int, title: str) -> Conversation:
        return await self._chats.create_conversation(user_id, title)

    async def get_conversation(self, conversation_id: int, user_id: int) -> Conversation:
        conversation = await self._chats.get_conversation(conversation_id, user_id)
        if conversation is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        return conversation

    async def chat(self, user_id: int, conversation_id: int, content: str) -> Message:
        """Save the user message, generate a reply, save it, return the assistant message."""
        await self.get_conversation(conversation_id, user_id)
        user_message = await self._messages.create_message(conversation_id, Message.ROLE_USER, content)
        assistant_message = await self._messages.create_message(conversation_id, Message.ROLE_ASSISTANT, PLACEHOLDER_RESPONSE)
        for message in (user_message, assistant_message):
            await self._memory.try_create_memory(user_id, conversation_id, message.id, message.content)
        return assistant_message
