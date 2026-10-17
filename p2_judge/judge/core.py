"""Judging engine: pointwise rubric scoring and position-debiased pairwise comparison."""
import logging

from judge.config import get_settings
from judge.llm import chat
from judge.parsing import ParseError, parse_json, regex_fallback, validate_pairwise, validate_pointwise
from judge.prompts import PAIRWISE, POINTWISE
from judge.rubric import passes, weighted_score

log = logging.getLogger("judge")
RETRY_MSG = "Your previous output was not valid JSON. Return ONLY the JSON object, nothing else."


def _call_and_parse(messages: list[dict], tag: str, sample: int, model: str | None,
                    validate, criteria: list[str]) -> tuple[dict | None, dict]:
    """Run the parse ladder. Returns (validated verdict or None, meta)."""
    model = model or get_settings().judge_model
    # Groq caps Qwen at 1,000 output tokens/min, so a Qwen judge must request less than that per call.
    max_tokens = 900 if "qwen" in model else 2000
    try:
        calls = [chat("judge", messages, model=model, json_mode=True, sample=sample, tag=tag, max_tokens=max_tokens)]
    except Exception as e:  # one failed call must not sink a whole suite run: record it and move on
        log.error("judge call failed for %s: %s", tag, str(e)[:200])
        return None, {"parse_status": "call_failed", "issues": [str(e)[:300]], "raw": "", "request_hash": None,
                      "tokens": {"prompt": 0, "completion": 0, "calls": 1, "cache_hits": 0}}
    meta = {"parse_status": None, "issues": [], "raw": calls[0].text, "request_hash": calls[0].request_hash}
    try:
        obj, status = parse_json(calls[0].text)
    except ParseError:
        retry = messages + [{"role": "assistant", "content": calls[0].text or "(empty)"},
                            {"role": "user", "content": RETRY_MSG}]
        try:
            calls.append(chat("judge", retry, model=model, json_mode=True, sample=sample, tag=tag + ":retry",
                              max_tokens=max_tokens))
        except Exception as e:
            log.error("judge retry failed for %s: %s", tag, str(e)[:200])
            calls.append(type(calls[0])("", model, "judge", 0, 0, 0.0, 0.0, False, ""))
        meta["raw_retry"] = calls[1].text
        try:
            obj, _ = parse_json(calls[1].text)
            status = "retried"
        except ParseError:
            obj = regex_fallback(calls[0].text + "\n" + calls[1].text, criteria)
            status = "regex_fallback" if obj else "failed"
    meta["parse_status"] = status
    meta["tokens"] = {"prompt": sum(c.prompt_tokens for c in calls),
                      "completion": sum(c.completion_tokens for c in calls), "calls": len(calls),
                      "cache_hits": sum(c.cache_hit for c in calls)}
    if obj is None:
        return None, meta
    verdict, issues = validate(obj)
    meta["issues"] = issues
    return verdict, meta


def judge_pointwise(case: dict, output: str, version: str = "v1", sample: int = 0,
                    model: str | None = None) -> dict:
    build, criteria = POINTWISE[version]
    verdict, meta = _call_and_parse(build(case, output), f"pointwise:{version}:{case['id']}", sample, model,
                                    lambda o: validate_pointwise(o, criteria), criteria)
    scores = {c: v["score"] for c, v in (verdict or {}).get("criteria", {}).items()}
    complete = verdict is not None and len(scores) == len(criteria)
    return {"case_id": case["id"], "version": version, "scores": scores,
            "weighted": round(weighted_score(scores, criteria), 3) if scores else None,
            "pass": passes(scores) if complete else None,
            "verdict": verdict, **meta}


def _pair_once(case: dict, first: str, second: str, version: str, sample: int, model: str | None,
               order: str) -> dict:
    verdict, meta = _call_and_parse(PAIRWISE[version](case, first, second),
                                    f"pairwise:{version}:{order}:{case['id']}", sample, model,
                                    validate_pairwise, [])
    return {"winner_shown": (verdict or {}).get("winner"), "verdict": verdict, **meta}


def judge_pairwise(case: dict, out_a: str, out_b: str, version: str = "v1", both_orders: bool = True,
                   sample: int = 0, model: str | None = None) -> dict:
    """Compare config A's output vs config B's output.

    Run 1 shows (A, B). Run 2 shows (B, A) and its verdict is mapped back to config labels.
    Final verdict = the shared answer when both orders agree, otherwise 'tie' (position-inconsistent).
    """
    r1 = _pair_once(case, out_a, out_b, version, sample, model, "AB")
    v1 = r1["winner_shown"]                                 # labels already match configs
    res = {"case_id": case["id"], "version": version, "order_AB": r1, "verdict_AB": v1}
    if not both_orders:
        res.update(final=v1, consistent=None, flipped=None)
        return res
    r2 = _pair_once(case, out_b, out_a, version, sample, model, "BA")
    v2 = {"A": "B", "B": "A", "tie": "tie"}.get(r2["winner_shown"])   # map back to config labels
    consistent = v1 is not None and v1 == v2
    res.update(order_BA=r2, verdict_BA=v2, consistent=consistent,
               # flip = the judge picked the SAME SCREEN POSITION both times (i.e. it followed position)
               flipped=v1 is not None and v2 is not None and v1 != v2,
               first_position_won=[v1 == "A", r2["winner_shown"] == "A"],
               final=v1 if consistent else "tie")
    return res
