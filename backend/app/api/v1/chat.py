"""Chat and conversation API routes."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.context.services.context_builder_service import ContextBuilderService
from app.database.session import get_db_session
from app.dependencies.auth import CurrentUser
from app.dependencies.context import get_context_builder
from app.dependencies.memory import get_memory_service
from app.memory.services.memory_service import MemoryService
from app.schemas.chat import ConversationCreate, ConversationRead, MessageCreate, MessageRead
from app.services.chat_service import ChatService
from app.services.message_service import MessageService

router = APIRouter(tags=["chat"])


@router.get("/conversations", response_model=list[ConversationRead])
async def list_conversations(
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db_session),
    memory_service: MemoryService = Depends(get_memory_service),
    context_builder: ContextBuilderService = Depends(get_context_builder),
) -> list[ConversationRead]:
    """List all conversations for the authenticated user."""
    service = ChatService(session, memory_service, context_builder)
    conversations = await service.list_conversations(current_user.id)
    return [ConversationRead.model_validate(c) for c in conversations]


@router.post("/conversations", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreate,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db_session),
    memory_service: MemoryService = Depends(get_memory_service),
    context_builder: ContextBuilderService = Depends(get_context_builder),
) -> ConversationRead:
    """Create a new conversation for the authenticated user."""
    service = ChatService(session, memory_service, context_builder)
    conversation = await service.create_conversation(current_user.id, payload.title)
    return ConversationRead.model_validate(conversation)


@router.get("/conversations/{conversation_id}", response_model=ConversationRead)
async def get_conversation(
    conversation_id: int,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db_session),
    memory_service: MemoryService = Depends(get_memory_service),
    context_builder: ContextBuilderService = Depends(get_context_builder),
) -> ConversationRead:
    """Get a specific conversation by ID for the authenticated user."""
    service = ChatService(session, memory_service, context_builder)
    conversation = await service.get_conversation(conversation_id, current_user.id)
    return ConversationRead.model_validate(conversation)


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageRead])
async def list_messages(
    conversation_id: int,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db_session),
    memory_service: MemoryService = Depends(get_memory_service),
) -> list[MessageRead]:
    """List all persisted messages for a conversation, oldest first."""
    service = MessageService(session, memory_service)
    messages = await service.list_messages(current_user.id, conversation_id)
    return [MessageRead.model_validate(m) for m in messages]


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=MessageRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_message(
    conversation_id: int,
    payload: MessageCreate,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db_session),
    memory_service: MemoryService = Depends(get_memory_service),
) -> MessageRead:
    """Persist a user message in a conversation."""
    service = MessageService(session, memory_service)
    message = await service.create_user_message(current_user.id, conversation_id, payload.content)
    return MessageRead.model_validate(message)


@router.post("/conversations/{conversation_id}/chat", response_model=MessageRead)
async def chat(
    conversation_id: int,
    payload: MessageCreate,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db_session),
    memory_service: MemoryService = Depends(get_memory_service),
    context_builder: ContextBuilderService = Depends(get_context_builder),
) -> MessageRead:
    """Save the user message, generate and save the reply, return the assistant message.

    The turn's prompt context (profile plus retrieved knowledge) is assembled by
    the context builder inside ``ChatService``; this route only wires it in.
    """
    service = ChatService(session, memory_service, context_builder)
    message = await service.chat(current_user.id, conversation_id, payload.content)
    return MessageRead.model_validate(message)
