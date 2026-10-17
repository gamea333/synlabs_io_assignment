"""Evaluation harness: retrieval metrics, answer quality, latency.

  python -m eval.run_eval                     # all three layers -> results/eval_results.json
  python -m eval.run_eval --retrieval-only    # IR metrics + retrieval latency only (no LLM calls)
  python -m eval.run_eval --table chunks_256 --retrieval-only --out results/sweep_256.json
"""
import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from app.config import get_settings
from app.embeddings import get_embedder
from app.rag import _format_context, answer, retrieve
from app.store import Hit, VectorStore
from eval import judge
from eval.metrics import (context_precision, contains_gold, evidence_recall, exact_match, hit_rate,
                          is_relevant, mrr, ndcg_at_k, percentile, precision_at_k, recall_at_k, token_f1)

log = logging.getLogger("eval")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


def load_gold(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["questions"]


# ----------------------------------------------------------------------------- retrieval
def eval_retrieval(gold: list[dict], store: VectorStore, k: int) -> dict:
    all_rows = store.table.to_arrow().select(["source", "text"]).to_pylist()
    per_q = []
    for q in [q for q in gold if q["evidence"]]:
        hits, _ = retrieve(q["question"], k, store=store)
        rels = [int(is_relevant(h.source, h.text, q["evidence"])) for h in hits]
        n_rel = sum(is_relevant(r["source"], r["text"], q["evidence"]) for r in all_rows)
        per_q.append({
            "id": q["id"], "n_relevant_in_index": n_rel, "rels": rels,
            "retrieved": [f"{h.source}#{h.chunk_idx}" for h in hits],
            "hit_rate": hit_rate(rels), "recall": recall_at_k(rels, n_rel),
            "evidence_recall": evidence_recall([(h.source, h.text) for h in hits], q["evidence"]),
            "mrr": mrr(rels), "ndcg": ndcg_at_k(rels, n_rel),
            "precision_at_k": precision_at_k(rels), "context_precision": context_precision(rels),
        })
    keys = ["hit_rate", "recall", "evidence_recall", "mrr", "ndcg", "precision_at_k", "context_precision"]
    return {"k": k, "n_questions": len(per_q),
            "summary": {key: mean([p[key] for p in per_q]) for key in keys},
            "per_question": per_q}


def eval_retrieval_latency(gold: list[dict], store: VectorStore, k: int, reps: int) -> dict:
    embedder = get_embedder()
    retrieve(gold[0]["question"], k, store=store)  # warm-up
    embed_search, search_only = [], []
    for _ in range(reps):
        for q in gold:
            t0 = time.perf_counter()
            vec = embedder.embed_query(q["question"])
            t1 = time.perf_counter()
            store.search(vec, k)
            t2 = time.perf_counter()
            embed_search.append((t2 - t0) * 1000)
            search_only.append((t2 - t1) * 1000)
    pct = lambda xs: {"p50": round(percentile(xs, 50), 2), "p95": round(percentile(xs, 95), 2),
                      "mean": round(statistics.mean(xs), 2)}
    return {"n_samples": len(search_only), "k": k, "vectors": store.count(),
            "search_only_ms": pct(search_only), "embed_plus_search_ms": pct(embed_search)}


# ----------------------------------------------------------------------------- answers
def eval_answers(gold: list[dict], k: int) -> dict:
    s = get_settings()
    per_q, audits = [], []
    judge_tokens = {"prompt": 0, "completion": 0, "calls": 0}

    def track(a):
        audits.append(a)
        judge_tokens["prompt"] += a["tokens"]["prompt"]
        judge_tokens["completion"] += a["tokens"]["completion"]
        judge_tokens["calls"] += 1

    for i, q in enumerate(gold, start=1):
        ans = answer(q["question"], k=k)
        rec = {"id": q["id"], "question": q["question"], "gold": q["answer"], "answer": ans.answer,
               "answerable_gold": bool(q["evidence"]), "answered": ans.answerable,
               "refusal_reason": ans.refusal_reason, "citations": [c.chunk_id for c in ans.citations],
               "latency_ms": ans.latency_ms, "tokens": ans.tokens}

        if not q["evidence"]:                                   # should refuse
            rec["correct_refusal"] = not ans.answerable
        elif not ans.answerable:                                # false refusal
            rec.update(faithfulness=None, relevance=0.0, judge_correct=False,
                       em=0.0, f1=0.0, contains=0.0)
        else:
            used = [Hit(**{k_: r[k_] for k_ in Hit.__dataclass_fields__})
                    for r in ans.retrieved if r["score"] >= s.min_score]
            f, a1 = judge.faithfulness(ans.answer, _format_context(used))
            r, correct, a2 = judge.relevance_and_correctness(q["question"], q["answer"], ans.answer)
            for a, kind in ((a1, "faithfulness"), (a2, "relevance")):
                a["id"], a["kind"] = q["id"], kind
                track(a)
            rec.update(faithfulness=f, relevance=r, judge_correct=correct,
                       unsupported_claims=[c["claim"] for c in a1["parsed"].get("claims", [])
                                           if c.get("verdict", "").lower() != "supported"],
                       em=exact_match(ans.answer, q["answer"]), f1=token_f1(ans.answer, q["answer"]),
                       contains=contains_gold(ans.answer, q["answer"]))
        per_q.append(rec)
        log.info("[%d/%d] %s answered=%s faith=%s rel=%s", i, len(gold), q["id"], ans.answerable,
                 rec.get("faithfulness"), rec.get("relevance"))

    ans_q = [p for p in per_q if p["answerable_gold"]]
    una_q = [p for p in per_q if not p["answerable_gold"]]
    answered = [p for p in ans_q if p["answered"]]
    totals = [p["latency_ms"]["total"] for p in per_q if p["answered"]]
    walls = [p["latency_ms"]["wall"] for p in per_q]
    gen_tokens = [p["tokens"]["total"] for p in per_q]
    summary = {
        "n_answerable": len(ans_q), "n_unanswerable": len(una_q),
        "faithfulness": mean([p["faithfulness"] for p in answered]),
        "answer_relevance": mean([p["relevance"] for p in ans_q]),
        "judge_correctness": mean([float(p["judge_correct"]) for p in ans_q]),
        "exact_match": mean([p["em"] for p in ans_q]),
        "token_f1": mean([p["f1"] for p in ans_q]),
        "contains_gold": mean([p["contains"] for p in ans_q]),
        "answer_rate_on_answerable": mean([float(p["answered"]) for p in ans_q]),
        "correct_refusal_rate_on_unanswerable": mean([float(p["correct_refusal"]) for p in una_q]),
        "e2e_latency_ms_excl_throttle": {"p50": round(percentile(totals, 50), 1),
                                         "p95": round(percentile(totals, 95), 1)},
        "e2e_wall_ms_incl_throttle": {"p50": round(percentile(walls, 50), 1),
                                      "p95": round(percentile(walls, 95), 1)},
        "generator_tokens": {"total": sum(gen_tokens), "mean_per_query": mean(gen_tokens)},
        "judge_tokens": judge_tokens,
    }
    return {"generator_model": s.llm_model, "judge_model": s.judge_model,
            "summary": summary, "per_question": per_q, "judge_audit": audits}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser()
    p.add_argument("--gold", default="eval/gold.json")
    p.add_argument("-k", type=int, default=None)
    p.add_argument("--table", default=None, help="LanceDB table to evaluate (default LANCEDB_TABLE)")
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--latency-reps", type=int, default=20)
    p.add_argument("--out", default="results/eval_results.json")
    args = p.parse_args()

    s = get_settings()
    k = args.k or s.top_k
    gold = load_gold(args.gold)
    store = VectorStore(get_embedder().dim, table=args.table)
    first = store.table.search().limit(1).to_list()[0]

    result = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"k": k, "table": args.table or s.lancedb_table, "vectors": store.count(),
                   "chunk_size": first["chunk_size"], "chunk_overlap": first["chunk_overlap"],
                   "embed_model": s.embed_model, "embed_dim": get_embedder().dim,
                   "min_score": s.min_score, "store": "LanceDB (embedded, exact cosine kNN, no ANN index)"},
        "retrieval": eval_retrieval(gold, store, k),
        "retrieval_latency": eval_retrieval_latency(gold, store, k, args.latency_reps),
    }
    log.info("retrieval: %s", result["retrieval"]["summary"])
    log.info("latency:   %s", result["retrieval_latency"])
    if not args.retrieval_only:
        result["answers"] = eval_answers(gold, k)
        log.info("answers:   %s", json.dumps(result["answers"]["summary"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
