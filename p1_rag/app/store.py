"""LanceDB vector store: embedded, file-based, no always-on server."""
from dataclasses import dataclass

import lancedb
import pyarrow as pa

from app.config import get_settings


def _q(value: str) -> str:
    """Quote a string literal for a LanceDB SQL filter."""
    return "'" + value.replace("'", "''") + "'"


def schema(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("source", pa.string()),
        pa.field("doc_type", pa.string()),
        pa.field("title", pa.string()),
        pa.field("chunk_idx", pa.int32()),
        pa.field("page", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("n_tokens", pa.int32()),
        pa.field("doc_hash", pa.string()),
        pa.field("chunk_size", pa.int32()),
        pa.field("chunk_overlap", pa.int32()),
        pa.field("embed_model", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])


@dataclass
class Hit:
    chunk_id: str
    source: str
    doc_type: str
    title: str
    page: int | None
    chunk_idx: int
    text: str
    score: float  # cosine similarity in [-1, 1]


class VectorStore:
    def __init__(self, dim: int, uri: str | None = None, table: str | None = None):
        s = get_settings()
        self.db = lancedb.connect(uri or s.lancedb_uri)
        self.table = self.db.create_table(table or s.lancedb_table, schema=schema(dim), exist_ok=True)

    # ---- ingest side -------------------------------------------------------
    def doc_state(self, source: str) -> dict | None:
        rows = (self.table.search()
                .where(f"source = {_q(source)}")
                .select(["doc_hash", "chunk_size", "chunk_overlap", "embed_model"])
                .limit(1).to_list())
        return rows[0] if rows else None

    def delete_source(self, source: str) -> None:
        self.table.delete(f"source = {_q(source)}")

    def upsert(self, rows: list[dict]) -> None:
        if not rows:
            return
        (self.table.merge_insert("chunk_id")
         .when_matched_update_all()
         .when_not_matched_insert_all()
         .execute(rows))

    def sources(self) -> set[str]:
        return set(self.table.to_arrow().column("source").unique().to_pylist()) if self.count() else set()

    # ---- query side --------------------------------------------------------
    def count(self, where: str | None = None) -> int:
        return self.table.count_rows(where)

    def count_distinct_ids(self) -> int:
        return len(set(self.table.to_arrow().column("chunk_id").to_pylist())) if self.count() else 0

    def search(self, vector, k: int, doc_type: str | None = None, source: str | None = None) -> list[Hit]:
        filters = []
        if doc_type:
            filters.append(f"doc_type = {_q(doc_type)}")
        if source:
            filters.append(f"source = {_q(source)}")
        q = self.table.search(vector).distance_type("cosine").limit(k)
        if filters:
            q = q.where(" AND ".join(filters), prefilter=True)
        return [
            Hit(r["chunk_id"], r["source"], r["doc_type"], r["title"], r["page"],
                r["chunk_idx"], r["text"], 1.0 - float(r["_distance"]))
            for r in q.to_list()
        ]
