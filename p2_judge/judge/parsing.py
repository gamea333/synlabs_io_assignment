"""Robust structured-verdict parsing.

Ladder (each step recorded in `parse_status`):
  1. strict json.loads                                  -> "ok"
  2. clean-up: strip <think>, ``` fences, prose; take the outermost {...}  -> "extracted"
  3. repair: trailing commas, single quotes, Python True/False/None, unquoted keys, // comments -> "repaired"
  4. caller retries once, sending the bad output back with "return valid JSON only" -> "retried"
  5. last resort: regex-scrape `"criterion": <int>` pairs and a verdict letter  -> "regex_fallback"
  6. give up -> "failed" (verdict excluded from aggregates and counted in the report)
Then the object is validated against the expected keys/ranges; out-of-range scores are clamped and flagged.
"""
import json
import re


class ParseError(ValueError):
    pass


def _strip(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"```(?:json)?", "", text)
    return text.strip()


def _outer_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ParseError("no '{' found")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            esc = (c == "\\" and not esc)
            if c == '"' and not esc:
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:] + "}" * depth          # truncated output: close open braces


def _repair(s: str) -> str:
    s = re.sub(r"//[^\n]*", "", s)                                   # // comments
    s = re.sub(r",\s*([}\]])", r"\1", s)                             # trailing commas
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', s)  # unquoted keys
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')                                      # all-single-quoted JSON
    return s


def parse_json(text: str) -> tuple[dict, str]:
    try:
        return json.loads(text), "ok"
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        obj = _outer_object(_strip(text or ""))
    except ParseError as e:
        raise ParseError(str(e)) from None
    try:
        return json.loads(obj), "extracted"
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_repair(obj)), "repaired"
    except json.JSONDecodeError as e:
        raise ParseError(f"unrepairable JSON: {e}") from None


def regex_fallback(text: str, criteria: list[str]) -> dict | None:
    scores = {}
    for c in criteria:
        m = re.search(rf'"?{c}"?\s*[:=]\s*\{{?[^0-9]{{0,40}}?([1-5])\b', text or "", flags=re.I | re.S)
        if m:
            scores[c] = {"score": int(m.group(1)), "rationale": "(recovered by regex fallback)", "evidence": ""}
    verdict = re.search(r'"?(?:winner|verdict)"?\s*[:=]\s*"?(A|B|tie)\b', text or "", flags=re.I)
    if not scores and not verdict:
        return None
    out = {"criteria": scores}
    if verdict:
        out["winner"] = verdict.group(1).upper() if verdict.group(1).lower() != "tie" else "tie"
    return out


def validate_pointwise(obj: dict, criteria: list[str]) -> tuple[dict, list[str]]:
    """Normalise to {criteria: {name: {score:int 1-5, rationale, evidence}}, overall_rationale}."""
    issues = []
    raw = obj.get("criteria", obj.get("scores", {}))
    if isinstance(raw, list):  # [{"name":..., "score":...}] variant
        raw = {d.get("name") or d.get("criterion"): d for d in raw if isinstance(d, dict)}
    crit = {}
    for c in criteria:
        v = raw.get(c)
        if v is None:
            issues.append(f"missing:{c}")
            continue
        if not isinstance(v, dict):
            v = {"score": v}
        try:
            score = int(round(float(v.get("score"))))
        except (TypeError, ValueError):
            issues.append(f"non_numeric:{c}")
            continue
        if not 1 <= score <= 5:
            issues.append(f"clamped:{c}={score}")
            score = min(max(score, 1), 5)
        crit[c] = {"score": score, "rationale": str(v.get("rationale", "")),
                   "evidence": str(v.get("evidence", ""))}
    return {"criteria": crit, "overall_rationale": str(obj.get("overall_rationale", obj.get("summary", "")))}, issues


def validate_pairwise(obj: dict) -> tuple[dict, list[str]]:
    issues = []
    w = str(obj.get("winner", obj.get("verdict", ""))).strip().upper()
    w = {"TIE": "tie", "A": "A", "B": "B", "RESPONSE A": "A", "RESPONSE B": "B"}.get(w)
    if w is None:
        issues.append("bad_winner")
    conf = obj.get("confidence")
    try:
        conf = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf = None
    return {"winner": w, "confidence": conf, "rationale": str(obj.get("rationale", "")),
            "per_criterion": obj.get("per_criterion", {})}, issues
