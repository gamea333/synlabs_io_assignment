"""Aggregate per-case verdicts into suite reports and an A/B decision."""
from collections import Counter

from judge.rubric import CRITERIA
from judge.stats import bootstrap_ci, mean, sign_test_p, stdev


def usage(records: list[dict]) -> dict:
    """Judge token/call accounting across pointwise and pairwise records."""
    metas = []
    for r in records:
        if "tokens" in r:
            metas.append(r)
        for k in ("order_AB", "order_BA"):
            if k in r:
                metas.append(r[k])
    return {"judge_calls": sum(m["tokens"]["calls"] for m in metas),
            "judge_prompt_tokens": sum(m["tokens"]["prompt"] for m in metas),
            "judge_completion_tokens": sum(m["tokens"]["completion"] for m in metas),
            "cache_hits": sum(m["tokens"]["cache_hits"] for m in metas),
            "parse_status": dict(Counter(m["parse_status"] for m in metas))}


def suite_report(pointwise: list[dict]) -> dict:
    ok = [p for p in pointwise if p["pass"] is not None]
    crit = sorted({c for p in ok for c in p["scores"]}, key=list(CRITERIA).index)
    return {
        "n_cases": len(pointwise), "n_scored": len(ok), "n_unparseable": len(pointwise) - len(ok),
        "pass_rate": round(mean([float(p["pass"]) for p in ok]), 4) if ok else None,
        "mean_weighted_score": round(mean([p["weighted"] for p in ok]), 3) if ok else None,
        "sd_weighted_score": round(stdev([p["weighted"] for p in ok]), 3) if ok else None,
        "mean_by_criterion": {c: round(mean([p["scores"].get(c) for p in ok]), 3) for c in crit},
        "failed_cases": [p["case_id"] for p in ok if not p["pass"]],
        "usage": usage(pointwise),
    }


def pairwise_report(pairs: list[dict]) -> dict:
    finals = Counter(p["final"] for p in pairs)
    a, b, t = finals.get("A", 0), finals.get("B", 0), finals.get("tie", 0) + finals.get(None, 0)
    n = len(pairs)
    both = [p for p in pairs if p.get("consistent") is not None]
    first_pos = [x for p in both for x in p["first_position_won"]]
    return {
        "n_pairs": n, "A_wins": a, "B_wins": b, "ties": t,
        "win_rate_A": round((a + 0.5 * t) / n, 4) if n else None,      # ties count half
        "win_rate_B": round((b + 0.5 * t) / n, 4) if n else None,
        "decisive_win_rate_B": round(b / (a + b), 4) if a + b else None,
        "sign_test_p": round(sign_test_p(b, a), 4),
        "position_flip_rate": round(mean([float(p["flipped"]) for p in both]), 4) if both else None,
        "first_position_win_rate": round(mean([float(x) for x in first_pos]), 4) if first_pos else None,
        "usage": usage(pairs),
    }


DECISION_RULE = (
    "Fixed before running: two independent signals. (1) Pairwise: debiased (both-order) decisive wins, "
    "two-sided sign test. (2) Pointwise: paired per-case difference in weighted rubric score, 95% bootstrap CI. "
    "A config WINS if at least one signal is significant in its favour (p < 0.05 / CI excludes 0) and the other "
    "signal does not significantly favour the opponent. Otherwise: 'no significant difference' -> keep incumbent A.")


def declare_winner(rep_a: dict, rep_b: dict, pw: dict, point_a: list[dict], point_b: list[dict],
                   alpha: float = 0.05) -> dict:
    by_a = {p["case_id"]: p["weighted"] for p in point_a if p["weighted"] is not None}
    diffs = [p["weighted"] - by_a[p["case_id"]] for p in point_b
             if p["weighted"] is not None and p["case_id"] in by_a]
    lo, hi = bootstrap_ci(diffs)
    md = mean(diffs) or 0.0

    pair_sig = None
    if pw["sign_test_p"] < alpha:
        pair_sig = "B" if pw["B_wins"] > pw["A_wins"] else "A"
    point_sig = "B" if lo > 0 else "A" if hi < 0 else None
    signals = {pair_sig, point_sig} - {None}
    winner = signals.pop() if len(signals) == 1 else None      # 0 signals -> none; 2 conflicting -> none

    return {"winner": {"A": "A", "B": "B"}.get(winner, "no significant difference"),
            "signals": {"pairwise": pair_sig or "n.s.", "pointwise": point_sig or "n.s."},
            "pointwise_mean_diff_B_minus_A": round(md, 3), "pointwise_diff_95ci": [round(lo, 3), round(hi, 3)],
            "pairwise_sign_test_p": pw["sign_test_p"],
            "pairwise_decisive_wins": {"A": pw["A_wins"], "B": pw["B_wins"], "ties": pw["ties"]},
            "pass_rate": {"A": rep_a["pass_rate"], "B": rep_b["pass_rate"]},
            "mean_weighted": {"A": rep_a["mean_weighted_score"], "B": rep_b["mean_weighted_score"]},
            "rule": DECISION_RULE}
