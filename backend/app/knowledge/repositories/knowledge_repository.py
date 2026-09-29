"""Persistence for knowledge documents and their embedded chunks."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models.knowledge import KnowledgeChunk, KnowledgeDocument
from app.knowledge.schemas.knowledge import ChunkPlan, DocumentDraft


class KnowledgeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, draft: DocumentDraft) -> KnowledgeDocument:
        """Insert a document already being processed, so a crash stays visible."""
        document = KnowledgeDocument(
            user_id=draft.user_id,
            title=draft.title,
            source=draft.source,
            content_type=draft.content_type,
            content=draft.content,
            status=KnowledgeDocument.STATUS_PROCESSING,
        )
        self._session.add(document)
        await self._session.commit()
        await self._session.refresh(document)
        return document

    async def list_for_user(self, user_id: int, limit: int) -> list[KnowledgeDocument]:
        statement = (
            select(KnowledgeDocument)
            .where(KnowledgeDocument.user_id == user_id)
            .order_by(KnowledgeDocument.id.desc())
            .limit(limit)
        )
        return list((await self._session.scalars(statement)).all())

    async def get_for_user(self, document_id: int, user_id: int) -> KnowledgeDocument | None:
        statement = select(KnowledgeDocument).where(
            KnowledgeDocument.id == document_id,
            KnowledgeDocument.user_id == user_id,
        )
        return await self._session.scalar(statement)

    async def get_by_ids(self, document_ids: list[int], user_id: int) -> list[KnowledgeDocument]:
        statement = select(KnowledgeDocument).where(
            KnowledgeDocument.id.in_(document_ids),
            KnowledgeDocument.user_id == user_id,
        )
        return list((await self._session.scalars(statement)).all())

    async def add_chunks(self, document: KnowledgeDocument, plans: list[ChunkPlan]) -> None:
        self._session.add_all(
            [
                KnowledgeChunk(
                    document_id=document.id,
                    user_id=document.user_id,
                    position=plan.position,
                    content=plan.content,
                    embedding_id=plan.embedding_id,
                    char_count=len(plan.content),
                )
                for plan in plans
            ]
        )
        await self._session.commit()

    async def list_chunks(self, document_id: int, user_id: int) -> list[KnowledgeChunk]:
        statement = (
            select(KnowledgeChunk)
            .where(KnowledgeChunk.document_id == document_id, KnowledgeChunk.user_id == user_id)
            .order_by(KnowledgeChunk.position.asc())
        )
        return list((await self._session.scalars(statement)).all())

    async def get_chunks_by_embedding_ids(self, embedding_ids: list[str], user_id: int) -> list[KnowledgeChunk]:
        statement = select(KnowledgeChunk).where(
            KnowledgeChunk.embedding_id.in_(embedding_ids),
            KnowledgeChunk.user_id == user_id,
        )
        return list((await self._session.scalars(statement)).all())

    async def mark_ready(
        self, document: KnowledgeDocument, char_count: int, chunk_count: int
    ) -> KnowledgeDocument:
        document.status = KnowledgeDocument.STATUS_READY
        document.char_count = char_count
        document.chunk_count = chunk_count
        document.error = None
        await self._session.commit()
        await self._session.refresh(document)
        return document

    async def mark_failed(self, document: KnowledgeDocument, error: str) -> KnowledgeDocument:
        document.status = KnowledgeDocument.STATUS_FAILED
        document.error = error
        document.char_count = 0
        document.chunk_count = 0
        await self._session.commit()
        await self._session.refresh(document)
        return document

    async def delete(self, document: KnowledgeDocument, chunks: list[KnowledgeChunk]) -> None:
        """Delete chunks explicitly; SQLite does not enforce ON DELETE CASCADE."""
        for chunk in chunks:
            await self._session.delete(chunk)
        await self._session.delete(document)
        await self._session.commit()
