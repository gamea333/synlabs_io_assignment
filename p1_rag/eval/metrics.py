"""Retrieval (IR) and answer (EM/F1) metrics. Pure functions, no I/O."""
import math
import re
import string
import unicodedata
from collections import Counter


def norm_text(s: str) -> str:
    """Normalisation used for evidence-span matching."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip().casefold()


def is_relevant(chunk_source: str, chunk_text: str, evidence: list[dict]) -> bool:
    """evidence = list of facts: {"source": ..., "spans": [alternative verbatim quotes]}"""
    t = norm_text(chunk_text)
    return any(chunk_source == e["source"] and any(norm_text(s) in t for s in e["spans"]) for e in evidence)


# ---- retrieval metrics: `rels` is the binary relevance of the ranked top-k list ----
#      `n_relevant` is how many relevant chunks exist in the whole index for this question.

def hit_rate(rels: list[int]) -> float:
    return float(any(rels))


def recall_at_k(rels: list[int], n_relevant: int) -> float:
    return sum(rels) / n_relevant if n_relevant else 0.0


def evidence_recall(retrieved: list[tuple[str, str]], evidence: list[dict]) -> float:
    """Fraction of required facts (not chunks) covered by the top-k. Matters for multi-source questions."""
    covered = sum(any(is_relevant(src, txt, [e]) for src, txt in retrieved) for e in evidence)
    return covered / len(evidence) if evidence else 0.0


def mrr(rels: list[int]) -> float:
    for i, r in enumerate(rels, start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels: list[int], n_relevant: int) -> float:
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rels, start=1))
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(n_relevant, len(rels)) + 1))
    return dcg / ideal if ideal else 0.0


def precision_at_k(rels: list[int]) -> float:
    return sum(rels) / len(rels) if rels else 0.0


def context_precision(rels: list[int]) -> float:
    """Rank-weighted context precision (RAGAS definition): mean of precision@i over ranks i that are relevant."""
    hits, total = 0, 0.0
    for i, r in enumerate(rels, start=1):
        if r:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


# ---- answer metrics (SQuAD-style normalisation) ----

def _norm_answer(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"\[\d+\]", " ", s)                                  # drop citation markers
    s = "".join(ch for ch in s if ch not in set(string.punctuation) - {"%"})
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(_norm_answer(pred) == _norm_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = _norm_answer(pred).split(), _norm_answer(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def contains_gold(pred: str, gold: str) -> float:
    """Lenient EM for generative answers: normalised gold string appears in the prediction."""
    return float(_norm_answer(gold) in _norm_answer(pred))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    idx = (len(v) - 1) * p / 100
    lo, hi = math.floor(idx), math.ceil(idx)
    return v[lo] + (v[hi] - v[lo]) * (idx - lo)
