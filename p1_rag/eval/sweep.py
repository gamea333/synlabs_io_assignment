"""Chunking + k sweep (retrieval layer only, no LLM calls).

  python -m eval.sweep            -> results/chunk_sweep.json
Each config is ingested into its own LanceDB table (chunks_<size>_<overlap>), then
scored on the same gold set. Labels are span-based, so they are valid at every chunk size.
"""
import json
import logging
import time
from pathlib import Path

from app.embeddings import get_embedder
from app.ingest import ingest_dir
from app.store import VectorStore
from eval.run_eval import eval_retrieval, eval_retrieval_latency, load_gold

CHUNK_CONFIGS = [(128, 16), (256, 32), (512, 0), (512, 64), (512, 128), (1024, 128)]
K_VALUES = [1, 3, 5, 10]
DEFAULT = (512, 64)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("ingest").setLevel(logging.WARNING)
    gold = load_gold("eval/gold.json")
    dim = get_embedder().dim
    out = {"run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "chunking": [], "k_sweep": []}

    for size, overlap in CHUNK_CONFIGS:
        table = f"chunks_{size}_{overlap}"
        rep = ingest_dir(Path("data/corpus"), size, overlap, table=table)
        store = VectorStore(dim, table=table)
        r = eval_retrieval(gold, store, k=5)
        lat = eval_retrieval_latency(gold, store, k=5, reps=5)
        row = {"chunk_size": size, "overlap": overlap, "vectors": store.count(),
               "context_tokens_at_k5": 5 * size, "ingest_seconds": rep.seconds,
               **r["summary"], "search_p50_ms": lat["search_only_ms"]["p50"],
               "search_p95_ms": lat["search_only_ms"]["p95"],
               "missed": [p["id"] for p in r["per_question"] if not p["hit_rate"]]}
        out["chunking"].append(row)
        logging.info("chunk %s/%s: %s", size, overlap, {k: row[k] for k in ("vectors", "hit_rate", "mrr", "ndcg")})

    store = VectorStore(dim, table=f"chunks_{DEFAULT[0]}_{DEFAULT[1]}")
    for k in K_VALUES:
        r = eval_retrieval(gold, store, k=k)
        out["k_sweep"].append({"k": k, **r["summary"]})
        logging.info("k=%d: %s", k, r["summary"])

    Path("results").mkdir(exist_ok=True)
    Path("results/chunk_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    logging.info("wrote results/chunk_sweep.json")


if __name__ == "__main__":
    main()
