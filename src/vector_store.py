from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from src.config import CHROMA_DIR, DATA_DIR
from src.parse_pdf import ChunkRecord
from src.embeddings import embed_texts


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def get_collection(name: str = "paper_chunks") -> Collection:
    _ensure_dirs()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(collection: Collection, chunks: list[ChunkRecord]) -> None:
    if not chunks:
        return
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    ids = [c.chunk_id for c in chunks]
    metadatas: list[dict[str, Any]] = []
    for c in chunks:
        metadatas.append(
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "title": c.title[:2000],
                "section": c.section[:500],
                "page": int(c.page),
            }
        )
    collection.add(ids=ids, embeddings=embeddings.tolist(), documents=texts, metadatas=metadatas)


def query_raw(
    collection: Collection,
    query_text: str,
    k: int,
    doc_id: str | None = None,
) -> dict[str, Any]:
    q_emb = embed_texts([query_text])[0].tolist()
    kwargs: dict[str, Any] = {
        "query_embeddings": [q_emb],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if doc_id:
        kwargs["where"] = {"doc_id": doc_id}
    return collection.query(**kwargs)


def delete_document(collection: Collection, doc_id: str) -> None:
    collection.delete(where={"doc_id": doc_id})
