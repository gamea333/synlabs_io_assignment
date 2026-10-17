"""Token-window chunking with overlap (cl100k tokens, so sizes match LLM token budgets)."""
import hashlib
from dataclasses import dataclass

import tiktoken

from app.loaders import Document

_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    chunk_id: str
    source: str
    doc_type: str
    title: str
    chunk_idx: int
    page: int | None
    text: str
    n_tokens: int


def chunk_id(source: str, idx: int, text: str) -> str:
    """Deterministic id: same file + same position + same text => same id on every run."""
    return hashlib.sha256(f"{source}\x1f{idx}\x1f{text}".encode()).hexdigest()[:32]


def chunk_document(doc: Document, size: int, overlap: int) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError(f"need 0 <= overlap < size, got size={size} overlap={overlap}")
    tokens = _ENC.encode(doc.text)
    if not tokens:
        return []
    _, char_offsets = _ENC.decode_with_offsets(tokens)
    step = size - overlap
    chunks = []
    for idx, start in enumerate(range(0, len(tokens), step)):
        window = tokens[start:start + size]
        text = _ENC.decode(window).strip()
        if text:
            chunks.append(Chunk(
                chunk_id=chunk_id(doc.source, idx, text),
                source=doc.source,
                doc_type=doc.doc_type,
                title=doc.title,
                chunk_idx=idx,
                page=doc.page_at(char_offsets[start]),
                text=text,
                n_tokens=len(window),
            ))
        if start + size >= len(tokens):
            break
    return chunks
