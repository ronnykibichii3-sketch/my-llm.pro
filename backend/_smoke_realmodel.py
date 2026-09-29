"""Temporary live check of the production embedding path: real model + real Qdrant."""

import asyncio

from app.memory.services.embedding_service import SentenceTransformerEmbeddingService
from app.memory.services.vector_store import QdrantVectorStore

COLLECTION = "smoke_real_model"
DOCUMENTS = [
    "the user's cat is named marmalade and it is orange",
    "the user bought shares of a technology company today",
]
QUERY = "what colour is my cat"


async def main() -> None:
    embedding = SentenceTransformerEmbeddingService()
    store = QdrantVectorStore(url="http://localhost:6333", collection_name=COLLECTION)

    vectors = await embedding.embed_batch(DOCUMENTS)
    print(f"model={embedding._model_name!r} dimension={embedding.dimension} vectors={len(vectors)}")
    assert all(len(vector) == embedding.dimension for vector in vectors)

    await store.ensure_collection(embedding.dimension)
    for index, (document, vector) in enumerate(zip(DOCUMENTS, vectors, strict=True)):
        await store.upsert(f"real-{index}", vector, {"user_id": 1, "content": document})

    query_vector = await embedding.embed(QUERY)
    hits = await store.search(query_vector, 5, 1)
    for hit in hits:
        print(f"  score={hit.score:.3f} {hit.payload['content']}")
    assert hits and hits[0].payload["content"] == DOCUMENTS[0], "irrelevant top hit"
    assert hits[0].score > hits[1].score
    assert await store.search(query_vector, 5, 2) == [], "user filter leaked points"

    other = await embedding.embed("totally unrelated query about gardening")
    assert len(await store.search(other, 5, 1)) == 2
    await store.delete("real-0")
    assert len(await store.search(query_vector, 5, 1)) == 1

    store._get_client().delete_collection(COLLECTION)
    print("REAL MODEL SMOKE OK")


if __name__ == "__main__":
    asyncio.run(main())
