"""Bias measurements (before = naive v0 judge, after = hardened v1 judge) and judge validation.

  python -m judge.experiments probes        # pairwise + pointwise on known-answer probes, v0 and v1
  python -m judge.experiments retest        # test-retest of the v1 judge (fresh, uncached draws)
  python -m judge.experiments self          # self-enhancement: gpt-oss-120b judge vs qwen judge on mixed-family pairs
  python -m judge.experiments report        # assemble results/bias_report.json + results/judge_validation.json

Gold labels for probes are known by construction. To make Cohen's kappa meaningful the "good" response is
assigned to config A for even-indexed probes and to config B for odd-indexed ones, so gold labels vary.
"""
import argparse
import json
import logging
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from judge.config import get_settings
from judge.core import judge_pairwise, judge_pointwise
from judge.generate import generate
from judge.report import usage
from judge.stats import cohen_kappa, entropy, mean, stdev
from judge.suite import load_config, load_suite

log = logging.getLogger("exp")
RES = Path("results")
POINTWISE_PROBE_TYPES = ("padded", "confident_wrong")
SECOND_JUDGE = "qwen/qwen3.8-27b"        # same family as the generator -> used to measure self-enhancement


def load_probes() -> list[dict]:
    probes = yaml.safe_load(Path("suites/probes.yaml").read_text(encoding="utf-8"))["probes"]
    for i, p in enumerate(probes):
        p["case"]["id"] = p["id"]
        p["good_side"] = "A" if i % 2 == 0 else "B"
        p["A"], p["B"] = (p["good"], p["bad"]) if p["good_side"] == "A" else (p["bad"], p["good"])
    return probes


def _save(name: str, obj) -> None:
    RES.mkdir(exist_ok=True)
    (RES / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote results/%s", name)


def _load(name: str):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# --------------------------------------------------------------------------- runs
def run_probes() -> None:
    probes = load_probes()
    out = {"pairwise": {}, "pointwise": {}}
    for version in ("v0", "v1"):
        with ThreadPoolExecutor(2) as ex:
            pairs = list(ex.map(lambda p: judge_pairwise(p["case"], p["A"], p["B"], version), probes))
        out["pairwise"][version] = pairs
        log.info("probes pairwise %s done", version)
        with ThreadPoolExecutor(2) as ex:
            # pointwise only where it is the evidence: padded (verbosity) and confident_wrong (sycophancy).
            # verbose_wrong / control are covered pairwise; skipping them saves ~40 rate-limited calls.
            jobs = [(p, side) for p in probes if p["type"] in POINTWISE_PROBE_TYPES for side in ("good", "bad")]
            pw = list(ex.map(lambda x: judge_pointwise(x[0]["case"], x[0][x[1]], version), jobs))
        out["pointwise"][version] = [{"probe": p["id"], "side": side, **r} for (p, side), r in zip(jobs, pw)]
        log.info("probes pointwise %s done", version)
    _save("probes_raw.json", out)


def run_retest() -> None:
    """Fresh draws (sample=1 bypasses the cache) of the production v1 judge at its configured temperature."""
    probes = load_probes()
    with ThreadPoolExecutor(2) as ex:
        pairs = list(ex.map(lambda p: judge_pairwise(p["case"], p["A"], p["B"], "v1", sample=1), probes))
    ab = _load("ab_report.json")
    suite = {c["id"]: c for c in load_suite("suites/main.yaml")["cases"]}
    with ThreadPoolExecutor(2) as ex:
        pw = list(ex.map(lambda cid: judge_pointwise(suite[cid], ab["outputs_B"][cid], "v1", sample=1),
                         list(ab["outputs_B"])[::2]))          # 12 of 24 cases, every other one
    _save("retest_raw.json", {"pairwise_probes_v1_sample1": pairs, "pointwise_suiteB_v1_sample1": pw})


def run_self_enhancement() -> None:
    suite = load_suite("suites/main.yaml")["cases"][::2]   # 12 of 24 (every other case -> all task types)
    cfg_gpt, cfg_qwen = load_config("configs/gen_gptoss.yaml"), load_config("configs/gen_v2.yaml")
    with ThreadPoolExecutor(1) as ex:
        out_gpt = {g["case_id"]: g["output"] for g in ex.map(lambda c: generate(c, cfg_gpt), suite)}
        out_qwen = {g["case_id"]: g["output"] for g in ex.map(lambda c: generate(c, cfg_qwen), suite)}
    res = {}
    for judge_name, model in (("gptoss120b_judge", get_settings().judge_model), ("qwen_judge", SECOND_JUDGE)):
        # A = gpt-oss output, B = qwen output; both orders, hardened v1 prompt
        with ThreadPoolExecutor(2) as ex:
            res[judge_name] = list(ex.map(
                lambda c: judge_pairwise(c, out_gpt[c["id"]], out_qwen[c["id"]], "v1", model=model), suite))
        log.info("self-enhancement %s done", judge_name)
    _save("self_enhancement_raw.json", {"A": "gpt-oss-20b (OpenAI family)", "B": "qwen3.8-27b (Alibaba family)",
                                        "judges": {"gptoss120b_judge": get_settings().judge_model, "qwen_judge": SECOND_JUDGE},
                                        "outputs_A": out_gpt, "outputs_B": out_qwen, "verdicts": res})


# --------------------------------------------------------------------------- analysis
def _pair_metrics(pairs: list[dict], probes: list[dict]) -> dict:
    gold = {p["id"]: p["good_side"] for p in probes}
    ptype = {p["id"]: p["type"] for p in probes}
    single, final, gold_single, gold_final = [], [], [], []
    by_type = defaultdict(lambda: {"n": 0, "fooled_single": 0, "fooled_final": 0, "correct_final": 0})
    for r in pairs:
        cid, g = r["case_id"], gold[r["case_id"]]
        bad = "B" if g == "A" else "A"
        t = by_type[ptype[cid]]
        t["n"] += 1
        for v in (r["verdict_AB"], r["verdict_BA"]):               # each order = one single-order judgment
            single.append(v or "none")
            gold_single.append(g)
            t["fooled_single"] += (v == bad) / 2
        final.append(r["final"] or "none")
        gold_final.append(g)
        t["fooled_final"] += r["final"] == bad
        t["correct_final"] += r["final"] == g
    acc = lambda pred, gl: round(mean([float(p == q) for p, q in zip(pred, gl)]), 4)
    return {
        "n_pairs": len(pairs),
        "single_order_accuracy": acc(single, gold_single),
        "single_order_kappa": round(cohen_kappa(single, gold_single), 4),
        "debiased_accuracy": acc(final, gold_final),
        "debiased_kappa": round(cohen_kappa(final, gold_final), 4),
        "position_flip_rate": round(mean([float(r["flipped"]) for r in pairs]), 4),
        "first_position_win_rate": round(mean([float(x) for r in pairs for x in r["first_position_won"]]), 4),
        "debiased_tie_rate": round(mean([float(r["final"] == "tie") for r in pairs]), 4),
        "by_type": {k: {**v, "fooled_single_rate": round(v["fooled_single"] / v["n"], 4),
                        "fooled_final_rate": round(v["fooled_final"] / v["n"], 4),
                        "correct_final_rate": round(v["correct_final"] / v["n"], 4)} for k, v in by_type.items()},
        "usage": usage(pairs),
    }


def _pointwise_probe_metrics(rows: list[dict], probes: list[dict]) -> dict:
    ptype = {p["id"]: p["type"] for p in probes}
    by = defaultdict(dict)
    for r in rows:
        by[r["probe"]][r["side"]] = r
    out = defaultdict(lambda: {"n": 0, "bad_scored_higher_or_equal": 0, "bad_correctness_ge3": 0,
                               "good_minus_bad": []})
    for pid, sides in by.items():
        g, b = sides.get("good"), sides.get("bad")
        if not g or not b or g["weighted"] is None or b["weighted"] is None:
            continue
        t = out[ptype[pid]]
        t["n"] += 1
        t["bad_scored_higher_or_equal"] += b["weighted"] >= g["weighted"]
        t["bad_correctness_ge3"] += b["scores"].get("correctness", 0) >= 3
        t["good_minus_bad"].append(g["weighted"] - b["weighted"])
    return {k: {"n": v["n"], "fooled_rate": round(v["bad_scored_higher_or_equal"] / v["n"], 4),
                "bad_correctness_ge3_rate": round(v["bad_correctness_ge3"] / v["n"], 4),
                "mean_margin_good_minus_bad": round(mean(v["good_minus_bad"]), 3)} for k, v in out.items()}


def _clustering(rows: list[dict], common: list[str]) -> dict:
    scores = [r["scores"][c] for r in rows for c in common if c in r["scores"]]
    weighted = [mean([r["scores"][c] for c in common if c in r["scores"]]) for r in rows if r["scores"]]
    dist = Counter(scores)
    return {"n_responses": len(weighted), "n_scores": len(scores),
            "score_distribution": {k: dist.get(k, 0) for k in range(1, 6)},
            "share_of_5s": round(dist.get(5, 0) / len(scores), 4),
            "entropy_bits": round(entropy(scores), 3),
            "sd_of_mean_score": round(stdev(weighted), 3),
            "all_criteria_5_rate": round(mean([float(all(r["scores"][c] == 5 for c in common if c in r["scores"]))
                                               for r in rows if r["scores"]]), 4)}


def build_report() -> None:
    probes = load_probes()
    raw = _load("probes_raw.json")
    ab = _load("ab_report.json")
    rt = _load("retest_raw.json")
    se = _load("self_enhancement_raw.json")
    report, validation = {}, {}

    if raw:
        m0, m1 = _pair_metrics(raw["pairwise"]["v0"], probes), _pair_metrics(raw["pairwise"]["v1"], probes)
        p0 = _pointwise_probe_metrics(raw["pointwise"]["v0"], probes)
        p1 = _pointwise_probe_metrics(raw["pointwise"]["v1"], probes)
        report["position"] = {
            "mitigation": "every pair judged in both orders; final = shared verdict, else tie (judge/core.py judge_pairwise)",
            "before_v0_single_order": {k: m0[k] for k in ("position_flip_rate", "first_position_win_rate", "single_order_accuracy")},
            "v1_prompt_single_order": {k: m1[k] for k in ("position_flip_rate", "first_position_win_rate", "single_order_accuracy")},
            "after_v1_both_orders": {"effective_flip_rate": 0.0, "debiased_accuracy": m1["debiased_accuracy"],
                                     "inconsistent_pairs_resolved_as_tie": m1["debiased_tie_rate"]},
        }
        report["verbosity"] = {
            "mitigation": "conciseness criterion penalising unsupported length + 'length is not quality' rule + padded probes",
            "pairwise_fooled_rate_verbose_wrong": {"before_v0_single": m0["by_type"]["verbose_wrong"]["fooled_single_rate"],
                                                   "after_v1_debiased": m1["by_type"]["verbose_wrong"]["fooled_final_rate"]},
            "pairwise_fooled_rate_padded": {"before_v0_single": m0["by_type"]["padded"]["fooled_single_rate"],
                                            "after_v1_debiased": m1["by_type"]["padded"]["fooled_final_rate"]},
            "pointwise_padded_scored_ge_plain": {"before_v0": p0.get("padded", {}).get("fooled_rate"),
                                                 "after_v1": p1.get("padded", {}).get("fooled_rate")},
            "pointwise_margin_plain_minus_padded": {"before_v0": p0.get("padded", {}).get("mean_margin_good_minus_bad"),
                                                    "after_v1": p1.get("padded", {}).get("mean_margin_good_minus_bad")},
        }
        report["sycophancy"] = {
            "mitigation": "per-criterion evidence quotes + 'confidence is not evidence' rule + confidently-wrong probes",
            "pairwise_fooled_rate": {"before_v0_single": m0["by_type"]["confident_wrong"]["fooled_single_rate"],
                                     "after_v1_debiased": m1["by_type"]["confident_wrong"]["fooled_final_rate"]},
            "pointwise_wrong_answer_got_correctness_ge3": {
                "before_v0": p0.get("confident_wrong", {}).get("bad_correctness_ge3_rate"),
                "after_v1": p1.get("confident_wrong", {}).get("bad_correctness_ge3_rate")},
        }
        validation["gold_agreement_probes"] = {
            "n_pairs": m1["n_pairs"], "labels": "known by construction (good side alternates A/B)",
            "v0_single_order": {"accuracy": m0["single_order_accuracy"], "kappa": m0["single_order_kappa"]},
            "v1_single_order": {"accuracy": m1["single_order_accuracy"], "kappa": m1["single_order_kappa"]},
            "v1_debiased": {"accuracy": m1["debiased_accuracy"], "kappa": m1["debiased_kappa"]},
        }
        validation["adversarial"] = {t: {"v0_fooled_single": m0["by_type"][t]["fooled_single_rate"],
                                         "v1_fooled_debiased": m1["by_type"][t]["fooled_final_rate"],
                                         "v1_correct_debiased": m1["by_type"][t]["correct_final_rate"]}
                                     for t in m1["by_type"]}
        validation["pointwise_probes"] = {"v0": p0, "v1": p1}
        report["probe_metrics_full"] = {"v0": m0, "v1": m1}

    if ab and raw:
        common = ["correctness", "faithfulness", "completeness", "instruction_following", "safety_tone"]
        v1_rows = ab["pointwise_A"] + ab["pointwise_B"] + raw["pointwise"]["v1"]
        v0_rows = raw["pointwise"]["v0"] + (_load("suite_pointwise_v0.json") or [])
        report["score_clustering"] = {
            "mitigation": "anchored 1-5 rubric + few-shot calibration examples (judge/prompts.py); pairwise for A/B",
            "note": "same responses where possible; computed on the 5 criteria both versions share",
            "before_v0": _clustering(v0_rows, common), "after_v1": _clustering(v1_rows, common)}

    if se:
        def win_rate(pairs, side):
            f = Counter(p["final"] for p in pairs)
            return round((f.get(side, 0) + 0.5 * f.get("tie", 0)) / len(pairs), 4), dict(f)
        # A = gpt-oss-20b output (OpenAI family), B = qwen output (Alibaba family)
        g_own, g_f = win_rate(se["verdicts"]["gptoss120b_judge"], "A")   # OpenAI judge -> OpenAI outputs
        q_own, q_f = win_rate(se["verdicts"]["qwen_judge"], "B")         # Alibaba judge -> Alibaba outputs
        g_for_openai, q_for_openai = g_own, round(1 - q_own, 4)
        report["self_enhancement"] = {
            "mitigation": "production judge gpt-oss-120b (OpenAI) is a different family from the production "
                          "generator qwen3.8-27b (Alibaba)",
            "setup": "12 prompts (every other suite case) answered by gpt-oss-20b (A, OpenAI) and qwen3.8-27b (B, Alibaba) with the "
                     "same prompt; judged pairwise (v1, both orders) by gpt-oss-120b and by qwen3.8-27b",
            "gptoss_judge_win_rate_for_openai_outputs": g_for_openai, "gptoss_judge_finals": g_f,
            "qwen_judge_win_rate_for_openai_outputs": q_for_openai, "qwen_judge_finals": q_f,
            "qwen_judge_win_rate_for_alibaba_outputs": q_own,
            "self_preference_gap": round(g_for_openai - q_for_openai, 4),
            "interpretation": "gap = how much more the OpenAI-family judge favours OpenAI outputs than the "
                              "Alibaba-family judge does. > 0 means at least one judge favours its own family.",
        }

    if rt and raw and ab:
        base = {r["case_id"]: r for r in raw["pairwise"]["v1"]}
        flips = [float(r["final"] != base[r["case_id"]]["final"]) for r in rt["pairwise_probes_v1_sample1"]]
        single = [float(r[k] != base[r["case_id"]][k]) for r in rt["pairwise_probes_v1_sample1"]
                  for k in ("verdict_AB", "verdict_BA")]
        pbase = {r["case_id"]: r for r in ab["pointwise_B"]}
        pw = rt["pointwise_suiteB_v1_sample1"]
        validation["test_retest"] = {
            "temperature": get_settings().judge_temperature,
            "pairwise_final_flip_rate": round(mean(flips), 4),
            "pairwise_single_order_flip_rate": round(mean(single), 4),
            "pointwise_pass_flip_rate": round(mean([float(r["pass"] != pbase[r["case_id"]]["pass"]) for r in pw]), 4),
            "pointwise_mean_abs_weighted_diff": round(mean([abs(r["weighted"] - pbase[r["case_id"]]["weighted"])
                                                            for r in pw if r["weighted"] and pbase[r["case_id"]]["weighted"]]), 3),
            "n_pairwise": len(flips), "n_pointwise": len(pw)}

    _save("bias_report.json", report)
    _save("judge_validation.json", validation)
    print(json.dumps({"bias": {k: v for k, v in report.items() if k != "probe_metrics_full"},
                      "validation": {k: v for k, v in validation.items() if k != "pointwise_probes"}}, indent=1))


def run_suite_v0() -> None:
    """Naive judge on the same A/B outputs -> 'before' side of score clustering."""
    ab = _load("ab_report.json")
    suite = {c["id"]: c for c in load_suite("suites/main.yaml")["cases"]}
    jobs = [(cid, out) for key in ("outputs_A", "outputs_B") for cid, out in ab[key].items()]
    with ThreadPoolExecutor(2) as ex:
        rows = list(ex.map(lambda j: judge_pointwise(suite[j[0]], j[1], "v0"), jobs))
    _save("suite_pointwise_v0.json", rows)


def _short(text: str, n: int = 150) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n] + "..."


def _print_pair(r: dict, label: str, a_text: str, b_text: str, gold: str | None = None) -> None:
    print(f"--- {label} ---")
    print(f"  config A: {_short(a_text)}")
    print(f"  config B: {_short(b_text)}")
    print(f"  order 1 (shown A first, B second) -> winner {r['verdict_AB']}: {_short(r['order_AB']['verdict']['rationale'], 230)}")
    print(f"  order 2 (shown B first, A second) -> judge said {r['order_BA']['winner_shown']} "
          f"= config {r['verdict_BA']}: {_short(r['order_BA']['verdict']['rationale'], 230)}")
    print(f"  consistent across orders: {r['consistent']}   flipped: {r['flipped']}   FINAL (debiased): {r['final']}"
          + (f"   GOLD: {gold}" + ("  -> CORRECT" if r['final'] == gold else "  -> WRONG") if gold else ""))


def show(kind: str, item: str) -> None:
    """Screenshot-friendly view of one pair judged in both orders, plus the aggregate rates."""
    probes = {p["id"]: p for p in load_probes()}
    if kind == "probe":
        raw, p = _load("probes_raw.json"), probes[item]
        print(f"PROBE {item}  type={p['type']}  question: {p['case']['input']}  reference: {p['case']['expected_output']}")
        print(f"good answer is config {p['good_side']}; bad = {'padded/verbose/confident' if p['type'] != 'control' else 'wrong'}\n")
        for v in ("v0", "v1"):
            r = next(x for x in raw["pairwise"][v] if x["case_id"] == item)
            _print_pair(r, f"judge {v} ({'naive' if v == 'v0' else 'hardened'})", p["A"], p["B"], p["good_side"])
            print()
        val = _load("judge_validation.json")["adversarial"][p["type"]]
        print(f"ALL {p['type']} probes: v0 fooled (single-order) {val['v0_fooled_single']:.0%}  ->  "
              f"v1 fooled (debiased) {val['v1_fooled_debiased']:.0%}, v1 correct {val['v1_correct_debiased']:.0%}")
    elif kind == "ab":
        ab = _load("ab_report.json")
        r = next(x for x in ab["pairs"] if x["case_id"] == item)
        case = next(c for c in load_suite("suites/main.yaml")["cases"] if c["id"] == item)
        print(f"A/B CASE {item}: {case['input']}   (A = prompt_v1, B = prompt_v2, judge v1)\n")
        _print_pair(r, "same pair, both orders", ab["outputs_A"][item], ab["outputs_B"][item])
        pos = _load("bias_report.json")["position"]
        print(f"\nPOSITION BIAS - real A/B suite (24 pairs, v1 judge): single-order flip rate "
              f"{ab['pairwise']['position_flip_rate']:.1%}, first-position win rate {ab['pairwise']['first_position_win_rate']:.1%}")
        print(f"POSITION BIAS - probes (22 pairs): naive v0 flip rate {pos['before_v0_single_order']['position_flip_rate']:.1%} "
              f"-> hardened v1 {pos['v1_prompt_single_order']['position_flip_rate']:.1%}; "
              f"after both-order debiasing the effective flip rate is 0% (flips become ties)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["probes", "retest", "self", "suite_v0", "report", "all", "show"])
    ap.add_argument("kind", nargs="?", choices=["probe", "ab"], help="for 'show'")
    ap.add_argument("item", nargs="?", help="probe id or suite case id, for 'show'")
    a = ap.parse_args()
    if a.what == "show":
        show(a.kind, a.item)
        return
    t = time.time()
    steps = {"probes": run_probes, "retest": run_retest, "self": run_self_enhancement,
             "suite_v0": run_suite_v0, "report": build_report}
    for name in (["probes", "suite_v0", "self", "retest", "report"] if a.what == "all" else [a.what]):
        log.info("=== %s", name)
        steps[name]()
    log.info("done in %.0fs", time.time() - t)


if __name__ == "__main__":
    main()
