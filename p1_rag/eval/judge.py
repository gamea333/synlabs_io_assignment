"""LLM-as-judge for answer quality. Judge model (JUDGE_MODEL) is a different family from the generator."""
import json
import re

from app.config import get_settings
from app.llm import chat

FAITHFULNESS_PROMPT = """You are a strict fact-checker. Break the ANSWER into atomic factual claims (ignore citation markers like [1]).
For each claim decide if it is directly SUPPORTED by the CONTEXT passages, or UNSUPPORTED (not stated, contradicted, or requires outside knowledge).
Return JSON only: {{"claims": [{{"claim": "...", "verdict": "supported" | "unsupported"}}]}}

CONTEXT:
{context}

ANSWER:
{answer}"""

RELEVANCE_PROMPT = """You grade a question-answering system.
QUESTION: {question}
REFERENCE ANSWER: {gold}
SYSTEM ANSWER: {answer}

Score two things:
1. relevance (1-5): does the system answer directly address what was asked? 5 = fully on point, 3 = partially / padded with off-topic content, 1 = does not address the question or is a refusal. Judge relevance only, not correctness.
2. correct (true/false): is the system answer consistent with the reference answer (same key fact; wording may differ)?
Return JSON only: {{"relevance": <1-5>, "correct": <true|false>, "reason": "<one sentence>"}}"""


def parse_json(text: str) -> dict:
    """Tolerate code fences / chatter around the JSON object."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


def _call(prompt: str) -> tuple[dict, dict]:
    res = chat([{"role": "user", "content": prompt}], model=get_settings().judge_model,
               temperature=0.0, max_tokens=1500, json_mode=True)
    audit = {"prompt": prompt, "raw": res.text, "model": res.model,
             "tokens": {"prompt": res.prompt_tokens, "completion": res.completion_tokens}}
    return parse_json(res.text), audit


def faithfulness(answer: str, context: str) -> tuple[float | None, dict]:
    verdict, audit = _call(FAITHFULNESS_PROMPT.format(context=context, answer=answer))
    claims = verdict.get("claims", [])
    supported = sum(c.get("verdict", "").lower() == "supported" for c in claims)
    audit["parsed"] = verdict
    return (supported / len(claims) if claims else None), audit


def relevance_and_correctness(question: str, gold: str, answer: str) -> tuple[float, bool, dict]:
    verdict, audit = _call(RELEVANCE_PROMPT.format(question=question, gold=gold, answer=answer))
    audit["parsed"] = verdict
    score = min(max(int(verdict.get("relevance", 1)), 1), 5)
    return (score - 1) / 4, bool(verdict.get("correct", False)), audit
