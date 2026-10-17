"""Retrieve top-k -> grounded, cited answer, with a two-gate 'no relevant context' path.

Gate 1 (retrieval): chunks below MIN_SCORE cosine are dropped. If none survive, we
         refuse WITHOUT calling the LLM (no tokens spent, nothing to hallucinate from).
Gate 2 (generation): the prompt requires the model to answer only from the numbered
         context and to emit the exact refusal string when the context is insufficient.
"""
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.embeddings import get_embedder
from app.llm import chat
from app.store import Hit, VectorStore

log = logging.getLogger("rag")

NO_ANSWER = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered context passages provided.
Rules:
- Every factual sentence MUST end with the bracketed number of the passage(s) it came from, e.g. [1] or [2][3].
- Do not use outside knowledge. Do not guess.
- If the passages do not contain the answer, reply with exactly: {NO_ANSWER}
- Be concise: at most 4 sentences.

Example
Context passages:
[1] (a.md) The Eiffel Tower is 330 metres tall.
[2] (b.md) It was completed in 1889 for the World's Fair.
Question: How tall is the Eiffel Tower and when was it finished?
Answer: The Eiffel Tower is 330 metres tall [1]. It was completed in 1889 [2]."""

# Repeated at the end of the user turn: models follow trailing instructions more reliably.
USER_SUFFIX = (f"\n\nAnswer using only the passages above. End every sentence with its citation like [1]. "
               f"If they do not contain the answer, reply exactly: {NO_ANSWER}")


@dataclass
class Citation:
    ref: int
    chunk_id: str
    source: str
    page: int | None
    score: float
    snippet: str


@dataclass
class Answer:
    question: str
    answer: str
    answerable: bool
    citations: list[Citation]
    retrieved: list[dict]                      # every top-k hit (for eval), incl. below-threshold
    latency_ms: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)
    k: int = 0
    filters: dict = field(default_factory=dict)
    refusal_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        loc = f"{h.source}" + (f", p.{h.page}" if h.page else "")
        blocks.append(f"[{i}] ({loc})\n{h.text}")
    return "\n\n".join(blocks)


def _log_query(ans: Answer) -> None:
    path = Path(get_settings().log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": ans.question, "k": ans.k, "filters": ans.filters,
        "latency_ms": ans.latency_ms,
        "chunks_retrieved": len(ans.retrieved), "chunks_used": len(ans.citations),
        "tokens": ans.tokens, "answerable": ans.answerable, "refusal_reason": ans.refusal_reason,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    log.info("query latency=%sms (embed=%s search=%s llm=%s throttle=%s) chunks_retrieved=%d "
             "chunks_used=%d tokens=%s/%s answerable=%s",
             ans.latency_ms["total"], ans.latency_ms["embed"], ans.latency_ms["search"],
             ans.latency_ms["llm"], ans.latency_ms["throttle"], len(ans.retrieved), len(ans.citations),
             ans.tokens["prompt"], ans.tokens["completion"], ans.answerable)


@lru_cache
def get_store() -> VectorStore:
    return VectorStore(get_embedder().dim)   # open the LanceDB table once per process


def retrieve(question: str, k: int, doc_type: str | None = None, source: str | None = None,
             store: VectorStore | None = None) -> tuple[list[Hit], dict]:
    embedder = get_embedder()
    store = store or get_store()
    t0 = time.perf_counter()
    qvec = embedder.embed_query(question)
    t1 = time.perf_counter()
    hits = store.search(qvec, k, doc_type=doc_type, source=source)
    t2 = time.perf_counter()
    return hits, {"embed": round((t1 - t0) * 1000, 1), "search": round((t2 - t1) * 1000, 1)}


def answer(question: str, k: int | None = None, doc_type: str | None = None,
           source: str | None = None, min_score: float | None = None) -> Answer:
    s = get_settings()
    k = k or s.top_k
    min_score = s.min_score if min_score is None else min_score
    t_start = time.perf_counter()

    hits, lat = retrieve(question, k, doc_type, source)
    relevant = [h for h in hits if h.score >= min_score]
    ans = Answer(question=question, answer=NO_ANSWER, answerable=False, citations=[],
                 retrieved=[{**asdict(h), "rank": i} for i, h in enumerate(hits, start=1)],
                 k=k, filters={"doc_type": doc_type, "source": source, "min_score": min_score})
    tokens = {"prompt": 0, "completion": 0, "total": 0}
    llm_ms = throttle_ms = 0.0

    if not relevant:
        ans.refusal_reason = "no_chunk_above_threshold"           # gate 1: LLM never called
    else:
        user = f"Context passages:\n\n{_format_context(relevant)}\n\nQuestion: {question}{USER_SUFFIX}"
        res = chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}])
        llm_ms, throttle_ms = res.gen_ms, res.throttle_ms
        tokens = {"prompt": res.prompt_tokens, "completion": res.completion_tokens, "total": res.total_tokens}

        if NO_ANSWER.lower().rstrip(".") in res.text.lower():
            ans.refusal_reason = "llm_insufficient_context"       # gate 2: model declined
        else:
            refs = sorted({int(n) for n in re.findall(r"\[(\d+)\]", res.text)})
            valid = [r for r in refs if 1 <= r <= len(relevant)]
            ans.answer = res.text
            ans.answerable = True
            ans.citations = [Citation(r, relevant[r - 1].chunk_id, relevant[r - 1].source,
                                      relevant[r - 1].page, round(relevant[r - 1].score, 4),
                                      relevant[r - 1].text[:200]) for r in valid]
            if not valid:
                ans.refusal_reason = "answer_without_citations"   # flagged, kept for eval

    wall_ms = (time.perf_counter() - t_start) * 1000
    ans.latency_ms = {**lat, "llm": round(llm_ms, 1), "throttle": round(throttle_ms, 1),
                      # total excludes provider rate-limit waits; wall includes them
                      "total": round(wall_ms - throttle_ms, 1), "wall": round(wall_ms, 1)}
    ans.tokens = tokens
    _log_query(ans)
    return ans
