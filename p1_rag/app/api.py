"""HTTP API.  Run: uvicorn app.api:app --port 8000"""
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.embeddings import get_embedder
from app.ingest import ingest_dir
from app.rag import answer
from app.store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
for noisy in ("httpx", "httpx2", "httpcore", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

class UTF8JSONResponse(JSONResponse):
    # explicit charset: Windows PowerShell 5.1 decodes bare application/json as Latin-1
    media_type = "application/json; charset=utf-8"


app = FastAPI(title="Cost-efficient RAG (LanceDB)", version="1.0", default_response_class=UTF8JSONResponse)


class QueryIn(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default_factory=lambda: get_settings().top_k, ge=1, le=50)
    doc_type: Literal["pdf", "html", "md"] | None = None     # metadata filter
    source: str | None = None                                # metadata filter, e.g. "md/pgvector_readme.md"
    min_score: float | None = Field(default=None, ge=-1, le=1)


class IngestIn(BaseModel):
    path: str = "data/corpus"
    chunk_size: int | None = Field(default=None, ge=32, le=4096)
    chunk_overlap: int | None = Field(default=None, ge=0)
    force: bool = False
    prune: bool = False


@app.on_event("startup")
def warm_up() -> None:
    get_embedder()   # load the ONNX model once, not on the first request


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict:
    emb = get_embedder()
    store = VectorStore(emb.dim)
    return {"vectors_total": store.count(), "documents": len(store.sources()),
            "by_doc_type": {t: store.count(f"doc_type = '{t}'") for t in ("pdf", "html", "md")},
            "embed_model": emb.model_name, "embed_dim": emb.dim}


@app.post("/ingest")
def ingest(body: IngestIn) -> dict:
    root = Path(body.path)
    if not root.is_dir():
        raise HTTPException(400, f"not a directory: {body.path}")
    return ingest_dir(root, body.chunk_size, body.chunk_overlap, body.force, body.prune).__dict__


@app.post("/query")
def query(body: QueryIn) -> dict:
    return answer(body.question, k=body.k, doc_type=body.doc_type,
                  source=body.source, min_score=body.min_score).to_dict()
