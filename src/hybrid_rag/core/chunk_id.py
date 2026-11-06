"""Deterministic chunk ID generation via UUIDv5.

Every chunk gets a stable ID derived from its document, strategy, and index.
Re-ingesting the same document with the same strategy yields identical IDs,
making upserts idempotent and cross-store consistency verifiable.
"""

import uuid

CHUNK_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_URL


def make_chunk_id(document_id: uuid.UUID, strategy: str, chunk_index: int) -> uuid.UUID:
    name = f"{document_id}:{strategy}:{chunk_index}"
    return uuid.uuid5(CHUNK_NAMESPACE, name)
