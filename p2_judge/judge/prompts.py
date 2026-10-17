"""Judge prompt templates.

v0 = NAIVE baseline ("before"): criterion names only, no anchors, no evidence requirement, no bias guidance,
     pairwise judged in a single order. Used only to MEASURE biases.
v1 = HARDENED judge ("after"): anchored rubric + few-shot calibration examples, per-criterion evidence quotes,
     explicit anti-length / anti-confidence / anti-position rules, conciseness criterion, pairwise run in BOTH
     orders in code (see pairwise.py). This is the production judge.
"""
from judge.rubric import CRITERIA, render

V0_CRITERIA = ["correctness", "faithfulness", "completeness", "instruction_following", "safety_tone"]
V1_CRITERIA = list(CRITERIA)


def automated_checks(output: str) -> str:
    """Countable facts computed in code. LLM judges miscount sentences/words (seen in smoke testing: a
    two-sentence summary was scored 1 for 'is a single sentence'), so v1 receives these as ground truth."""
    import json
    import re
    text = output.strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text) if s.strip()]
    lines = [ln for ln in text.splitlines() if ln.strip()]
    bullets = [ln for ln in lines if re.match(r"\s*([-*•]|\d+[.)])\s+", ln)]
    try:
        json.loads(text)
        is_json = True
    except ValueError:
        is_json = False
    return (f"AUTOMATED CHECKS (computed by code - treat as reliable, do not recount): words={len(text.split())}, "
            f"sentences={len(sentences)}, non_empty_lines={len(lines)}, bullet_lines={len(bullets)}, "
            f"valid_json={str(is_json).lower()}")


def _case_block(case: dict) -> str:
    parts = [f"SYSTEM PROMPT GIVEN TO THE ASSISTANT:\n{case.get('system_prompt') or '(none)'}",
             f"USER INPUT:\n{case['input']}"]
    if case.get("context"):
        parts.append(f"CONTEXT PROVIDED TO THE ASSISTANT:\n{case['context']}")
    if case.get("expected_output"):
        parts.append(f"REFERENCE ANSWER (ground truth):\n{case['expected_output']}")
    if case.get("criteria"):
        parts.append(f"CASE-SPECIFIC CRITERIA:\n{case['criteria']}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- pointwise
def pointwise_v0(case: dict, output: str) -> list[dict]:
    crit = ", ".join(V0_CRITERIA)
    user = (f"{_case_block(case)}\n\nASSISTANT RESPONSE:\n{output}\n\n"
            f"Rate the assistant response from 1 to 5 on each of: {crit}.\n"
            'Return JSON: {"criteria": {"<name>": {"score": <1-5>, "rationale": "..."}}, "overall_rationale": "..."}')
    return [{"role": "user", "content": user}]


CALIBRATION_EXAMPLES = """CALIBRATION EXAMPLES (use the full 1-5 range; most real responses are NOT all 5s):
Q: "What is the boiling point of water at sea level in Celsius? Answer in one word." REFERENCE: "100"
 - "100" -> correctness 5, instruction_following 5, conciseness 5
 - "Water boils at 100 degrees Celsius at sea level, which is a well-known physical constant that..." -> correctness 5, instruction_following 1 (asked for one word), conciseness 2
 - "Absolutely certain: 90." -> correctness 1, faithfulness 1 (confident tone is not evidence)
 - "Around 100, though it depends on pressure." -> correctness 4, instruction_following 2"""

V1_RULES = """RULES
1. Judge each criterion independently, using the anchors. Scores are integers 1-5.
2. Ground every score: in "evidence", quote the exact span of the RESPONSE (and of the REFERENCE/CONTEXT when
   relevant) that justifies it. If you cannot quote evidence for a claim being correct, do not give credit for it.
3. Length is not quality. Do not reward extra length unless it adds requested information. Penalise padding under
   conciseness.
4. Confidence is not evidence. Assertive, expert-sounding or emphatic wording must not raise correctness; check the
   claim against the REFERENCE/CONTEXT or established fact.
5. Style (formatting, politeness, markdown) affects only instruction_following/safety_tone, never correctness.
6. If a criterion truly does not apply, score it 5 and say "n/a" in the rationale.
7. For counting constraints (sentences, words, bullets, lines, JSON validity) use the AUTOMATED CHECKS line.
   "evidence" must be a verbatim quote, not a description."""


def pointwise_v1(case: dict, output: str) -> list[dict]:
    system = ("You are a meticulous, impartial evaluator of AI assistant responses. You output only JSON.\n\n"
              f"RUBRIC\n{render(V1_CRITERIA)}\n\n{CALIBRATION_EXAMPLES}\n\n{V1_RULES}")
    keys = ", ".join(f'"{c}": {{"evidence": "...", "rationale": "...", "score": <1-5>}}' for c in V1_CRITERIA)
    user = (f"{_case_block(case)}\n\nASSISTANT RESPONSE TO EVALUATE:\n<<<\n{output}\n>>>\n"
            f"{automated_checks(output)}\n\n"
            f'Return JSON: {{"criteria": {{{keys}}}, "overall_rationale": "<one sentence>"}}')
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- pairwise
def pairwise_v0(case: dict, first: str, second: str) -> list[dict]:
    user = (f"{_case_block(case)}\n\nRESPONSE A:\n{first}\n\nRESPONSE B:\n{second}\n\n"
            'Which response is better? Return JSON: {"winner": "A" | "B", "rationale": "..."}')
    return [{"role": "user", "content": user}]


def pairwise_v1(case: dict, first: str, second: str) -> list[dict]:
    system = ("You are an impartial judge comparing two AI assistant responses to the same request. "
              "You output only JSON.\n\n"
              f"RUBRIC (compare the two responses on each criterion)\n{render(V1_CRITERIA, with_anchors=False)}\n\n"
              f"{V1_RULES}\n7. The order in which the responses are shown is random and carries no information. "
              "Do not favour the first or the second position.\n"
              "9. Declare \"tie\" when the responses are equally good or differ only in style or length.")
    per = ", ".join(f'"{c}": "A" | "B" | "tie"' for c in V1_CRITERIA)
    user = (f"{_case_block(case)}\n\nRESPONSE A:\n<<<\n{first}\n>>>\nA {automated_checks(first)}\n\n"
            f"RESPONSE B:\n<<<\n{second}\n>>>\nB {automated_checks(second)}\n\n"
            "First compare per criterion, quoting evidence in the rationale, then decide.\n"
            f'Return JSON: {{"per_criterion": {{{per}}}, "rationale": "<2-3 sentences with quoted evidence>", '
            '"winner": "A" | "B" | "tie", "confidence": <0.0-1.0>}}')
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


POINTWISE = {"v0": (pointwise_v0, V0_CRITERIA), "v1": (pointwise_v1, V1_CRITERIA)}
PAIRWISE = {"v0": pairwise_v0, "v1": pairwise_v1}
