"""Idempotent ingestion.

Three layers guarantee no duplicate vectors:
  1. doc-level skip  - unchanged file bytes + same chunk config + same model => not re-embedded
  2. replace-on-change - a changed file has ALL its old chunks deleted before new ones go in
  3. upsert by deterministic chunk_id (merge_insert) - even a forced re-run cannot duplicate
"""
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.chunking import chunk_document
from app.config import get_settings
from app.embeddings import get_embedder
from app.loaders import discover, load_file
from app.store import VectorStore

log = logging.getLogger("ingest")


@dataclass
class IngestReport:
    files_seen: int = 0
    files_skipped: int = 0
    files_ingested: int = 0
    files_pruned: int = 0
    chunks_written: int = 0
    vectors_before: int = 0
    vectors_after: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ingest_dir(root: Path, chunk_size: int | None = None, chunk_overlap: int | None = None,
               force: bool = False, prune: bool = False) -> IngestReport:
    s = get_settings()
    size = chunk_size or s.chunk_size
    overlap = s.chunk_overlap if chunk_overlap is None else chunk_overlap
    embedder = get_embedder()
    store = VectorStore(embedder.dim)
    rep = IngestReport(vectors_before=store.count())
    t0 = time.perf_counter()

    paths = discover(root)
    seen_sources = set()
    for path in paths:
        rep.files_seen += 1
        try:
            doc_hash = file_hash(path)
            source = path.relative_to(root).as_posix()
            seen_sources.add(source)
            state = store.doc_state(source)
            unchanged = state and (state["doc_hash"], state["chunk_size"], state["chunk_overlap"],
                                   state["embed_model"]) == (doc_hash, size, overlap, embedder.model_name)
            if unchanged and not force:
                rep.files_skipped += 1
                log.info("skip     %s (unchanged)", source)
                continue

            doc = load_file(path, root)
            chunks = chunk_document(doc, size, overlap)
            vectors = embedder.embed_docs([c.text for c in chunks])
            rows = [{
                **c.__dict__, "page": c.page or 0, "doc_hash": doc_hash,
                "chunk_size": size, "chunk_overlap": overlap,
                "embed_model": embedder.model_name, "vector": v.tolist(),
            } for c, v in zip(chunks, vectors)]

            if state and not unchanged:
                store.delete_source(source)   # file or config changed: drop stale chunks
            store.upsert(rows)
            rep.files_ingested += 1
            rep.chunks_written += len(rows)
            log.info("ingest   %s -> %d chunks", source, len(rows))
        except Exception as e:  # one bad file shouldn't kill the run
            rep.errors.append(f"{path}: {e}")
            log.exception("failed   %s", path)

    if prune:
        for stale in store.sources() - seen_sources:
            store.delete_source(stale)
            rep.files_pruned += 1
            log.info("pruned   %s (no longer on disk)", stale)

    rep.vectors_after = store.count()
    rep.seconds = round(time.perf_counter() - t0, 2)
    return rep
