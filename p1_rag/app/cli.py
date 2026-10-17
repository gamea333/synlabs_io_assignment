"""CLI: python -m app.cli {ingest,stats} ..."""
import argparse
import json
import logging
from pathlib import Path

from app.config import get_settings


def cmd_ingest(args) -> None:
    from app.ingest import ingest_dir
    rep = ingest_dir(Path(args.path), args.chunk_size, args.chunk_overlap, force=args.force, prune=args.prune)
    print(json.dumps(rep.__dict__, indent=2))


def cmd_query(args) -> None:
    from app.rag import answer
    a = answer(args.question, k=args.k, doc_type=args.doc_type, source=args.source)
    out = a.to_dict()
    out.pop("retrieved")
    for c in out["citations"]:
        c.pop("snippet")
    print(json.dumps(out, indent=2))


def cmd_stats(_args) -> None:
    from app.embeddings import get_embedder
    from app.store import VectorStore
    emb = get_embedder()
    store = VectorStore(emb.dim)
    total = store.count()
    print(json.dumps({
        "vectors_total": total,
        "distinct_chunk_ids": store.count_distinct_ids(),
        "duplicates": total - store.count_distinct_ids(),
        "by_doc_type": {t: store.count(f"doc_type = '{t}'") for t in ("pdf", "html", "md")},
        "documents": len(store.sources()),
        "embed_model": emb.model_name,
        "embed_dim": emb.dim,
        "table": f"{get_settings().lancedb_uri}/{get_settings().lancedb_table}",
    }, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser(prog="app.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="ingest a folder of PDF/HTML/MD files")
    pi.add_argument("path")
    pi.add_argument("--chunk-size", type=int, default=None, help="tokens (default from CHUNK_SIZE=512)")
    pi.add_argument("--chunk-overlap", type=int, default=None, help="tokens (default from CHUNK_OVERLAP=64)")
    pi.add_argument("--force", action="store_true", help="re-embed even unchanged files (still no duplicates)")
    pi.add_argument("--prune", action="store_true", help="delete vectors for files no longer on disk")
    pi.set_defaults(func=cmd_ingest)

    pq = sub.add_parser("query", help="ask a question, get a cited answer")
    pq.add_argument("question")
    pq.add_argument("-k", type=int, default=None, help="top-k (default TOP_K=5)")
    pq.add_argument("--doc-type", choices=["pdf", "html", "md"], default=None, help="metadata filter")
    pq.add_argument("--source", default=None, help='metadata filter, e.g. "md/pgvector_readme.md"')
    pq.set_defaults(func=cmd_query)

    ps = sub.add_parser("stats", help="vector counts + duplicate check")
    ps.set_defaults(func=cmd_stats)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
