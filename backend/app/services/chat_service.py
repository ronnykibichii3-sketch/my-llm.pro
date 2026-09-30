"""Conversation turns: persist the exchange and produce the assistant reply.

The reply pipeline is ``user message -> ContextBuilderService -> InferenceService``.
Profile and knowledge lookups sit behind the context builder, and the model behind
the inference service, on purpose: this service neither queries a context source
nor speaks to a provider, so a new context source or a new backend is adopted in
one place each (see app/context/README.md and app/inference/README.md).
"""

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.context.schemas.context import ContextObject
from app.context.services.context_builder_service import ContextBuilderService
from app.database.models.chat import Conversation, Message
from app.inference.services.inference_service import InferenceService
from app.memory.services.memory_service import MemoryService
from app.repositories.chat_repository import ChatRepository
from app.repositories.message_repository import MessageRepository


class ChatService:
    def __init__(
        self,
        session: AsyncSession,
        memory: MemoryService,
        context: ContextBuilderService,
        inference: InferenceService,
    ) -> None:
        self._chats = ChatRepository(session)
        self._messages = MessageRepository(session)
        self._memory = memory
        self._context = context
        self._inference = inference

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
        context = await self._context.build_context(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=content,
        )
        reply = await self._generate_reply(content, context)
        assistant_message = await self._messages.create_message(conversation_id, Message.ROLE_ASSISTANT, reply)
        for message in (user_message, assistant_message):
            await self._memory.try_create_memory(user_id, conversation_id, message.id, message.content)
        return assistant_message

    async def _generate_reply(self, content: str, context: ContextObject) -> str:
        """Produce the answer to ``content`` given the assembled ``context``.

        The inference service composes the prompt, calls the configured provider and
        reports the turn; a provider that cannot answer comes back as the stored
        apology rather than an exception, so the conversation is never left with a
        user message and no reply.
        """
        response = await self._inference.answer(context, content)
        return response.content

