"""Bonus: LanceDB vs FAISS at scale (latency, ANN recall vs exact, build time, disk/RAM).

  python -m eval.bench_stores --sizes 100000 1000000

Vectors: synthetic but *structured* — each is a real corpus chunk embedding plus Gaussian noise,
re-normalised, so neighbourhoods are clustered like real data (pure random vectors make ANN look
unrealistically hard). Queries: the 27 gold questions' real query embeddings.
Ground truth for ANN recall: exact inner-product top-10 (FAISS IndexFlatIP).
"""
import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

import faiss
import lancedb
import numpy as np
import pyarrow as pa

from app.embeddings import get_embedder
from app.store import VectorStore
from eval.metrics import percentile
from eval.run_eval import load_gold

K = 10
BENCH_DIR = Path("data/bench")


def make_vectors(base: np.ndarray, n: int, noise: float = 0.35, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(base), size=n)
    v = base[idx] + rng.normal(0, noise / np.sqrt(base.shape[1]), size=(n, base.shape[1])).astype(np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def timeit(fn, queries, reps=5):
    fn(queries[0])  # warm
    lat = []
    for _ in range(reps):
        for q in queries:
            t = time.perf_counter()
            fn(q)
            lat.append((time.perf_counter() - t) * 1000)
    return {"p50": round(percentile(lat, 50), 2), "p95": round(percentile(lat, 95), 2),
            "mean": round(statistics.mean(lat), 2)}


def recall(results: list[list[int]], truth: list[list[int]]) -> float:
    return round(statistics.mean(len(set(r) & set(t)) / K for r, t in zip(results, truth)), 4)


def dir_size_mb(p: Path) -> float:
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6, 1)


def bench(n: int, base: np.ndarray, queries: np.ndarray) -> dict:
    vecs = make_vectors(base, n)
    dim = vecs.shape[1]
    out = {"vectors": n, "dim": dim, "raw_vector_mb": round(vecs.nbytes / 1e6, 1)}

    # ---- FAISS exact = ground truth
    t = time.perf_counter()
    flat = faiss.IndexFlatIP(dim)
    flat.add(vecs)
    out["faiss_flat"] = {"build_s": round(time.perf_counter() - t, 2)}
    truth = [flat.search(q[None], K)[1][0].tolist() for q in queries]
    out["faiss_flat"]["latency_ms"] = timeit(lambda q: flat.search(q[None], K), queries)
    out["faiss_flat"]["recall@10"] = 1.0

    # ---- FAISS HNSW (in-memory ANN)
    t = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 64
    hnsw.add(vecs)
    hnsw.hnsw.efSearch = 64
    out["faiss_hnsw"] = {"build_s": round(time.perf_counter() - t, 2), "params": "M=32, efC=64, efS=64",
                         "latency_ms": timeit(lambda q: hnsw.search(q[None], K), queries),
                         "recall@10": recall([hnsw.search(q[None], K)[1][0].tolist() for q in queries], truth)}

    # ---- LanceDB (on-disk)
    path = BENCH_DIR / f"lance_{n}"
    shutil.rmtree(path, ignore_errors=True)
    db = lancedb.connect(str(path))
    t = time.perf_counter()
    tbl = db.create_table("v", pa.table({
        "id": pa.array(np.arange(n, dtype=np.int64)),
        "vector": pa.FixedSizeListArray.from_arrays(pa.array(vecs.ravel()), dim)}))
    load_s = time.perf_counter() - t
    lance_ids = lambda q, **kw: [r["id"] for r in _lance_q(tbl, q, **kw)]
    out["lancedb_exact"] = {"load_s": round(load_s, 2),
                            "latency_ms": timeit(lambda q: _lance_q(tbl, q), queries),
                            "recall@10": recall([lance_ids(q) for q in queries], truth),
                            "disk_mb": dir_size_mb(path)}

    parts = max(16, int(np.sqrt(n)))
    t = time.perf_counter()
    tbl.create_index(metric="cosine", index_type="IVF_PQ", num_partitions=parts, num_sub_vectors=dim // 8)
    idx_s = time.perf_counter() - t
    for nprobes, refine in ((20, 10), (50, 20)):
        kw = {"nprobes": nprobes, "refine": refine}
        out[f"lancedb_ivfpq_np{nprobes}_rf{refine}"] = {
            "index_build_s": round(idx_s, 2), "params": f"partitions={parts}, sub_vectors={dim // 8}, {kw}",
            "latency_ms": timeit(lambda q: _lance_q(tbl, q, **kw), queries),
            "recall@10": recall([lance_ids(q, **kw) for q in queries], truth),
            "disk_mb": dir_size_mb(path)}
    return out


def _lance_q(tbl, q, nprobes=None, refine=None):
    s = tbl.search(q).distance_type("cosine").limit(K).select(["id"])
    if nprobes:
        s = s.nprobes(nprobes)
    if refine:
        s = s.refine_factor(refine)
    return s.to_list()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000])
    args = ap.parse_args()
    emb = get_embedder()
    store = VectorStore(emb.dim)
    base = np.stack(store.table.to_arrow().column("vector").to_numpy(zero_copy_only=False)).astype(np.float32)
    queries = np.stack([emb.embed_query(q["question"]) for q in load_gold("eval/gold.json")])
    results = {"run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "k": K, "n_queries": len(queries),
               "hardware": "local Windows laptop CPU (see README)", "runs": []}
    for n in args.sizes:
        r = bench(n, base, queries)
        results["runs"].append(r)
        print(json.dumps(r, indent=1))
    shutil.rmtree(BENCH_DIR, ignore_errors=True)
    Path("results/store_benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
