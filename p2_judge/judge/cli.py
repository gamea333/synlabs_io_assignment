"""CLI.

  python -m judge.cli run      --suite suites/main.yaml --gen configs/gen_v2.yaml          # suite in -> report out
  python -m judge.cli run      --suite suites/with_outputs.json                            # judge given model_output
  python -m judge.cli compare  --suite suites/main.yaml --a configs/gen_v1.yaml --b configs/gen_v2.yaml
  python -m judge.cli replay   --hash <request_hash>      # show a logged judge prompt + raw response
  python -m judge.cli reparse                             # re-run the parser over every logged judge response
"""
import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from judge.config import get_settings
from judge.core import judge_pairwise, judge_pointwise
from judge.generate import generate
from judge.report import declare_winner, pairwise_report, suite_report
from judge.suite import load_config, load_suite

log = logging.getLogger("cli")


def _setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _meta(**extra) -> dict:
    s = get_settings()
    return {"run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "judge_model": s.judge_model,
            "judge_temperature": s.judge_temperature, "generator_model": s.generator_model, **extra}


def _outputs(cases: list[dict], gen_cfg: dict | None) -> dict[str, str]:
    if gen_cfg is None:
        missing = [c["id"] for c in cases if not c.get("model_output")]
        if missing:
            raise SystemExit(f"cases without model_output and no --gen config: {missing}")
        return {c["id"]: c["model_output"] for c in cases}
    with ThreadPoolExecutor(1) as ex:   # generator (Qwen) has a 1K output-token/min cap: no point in parallel
        gens = list(ex.map(lambda c: generate(c, gen_cfg), cases))
    return {g["case_id"]: g["output"] for g in gens}


def _pointwise_all(cases, outputs, version, workers=2):
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda c: judge_pointwise(c, outputs[c["id"]], version), cases))
    for r in res:
        log.info("pointwise %s %-8s weighted=%s pass=%s parse=%s", version, r["case_id"], r["weighted"],
                 r["pass"], r["parse_status"])
    return res


def cmd_run(args):
    suite = load_suite(args.suite)
    gen = load_config(args.gen) if args.gen else None
    outputs = _outputs(suite["cases"], gen)
    pw = _pointwise_all(suite["cases"], outputs, args.judge_version)
    out = {**_meta(suite=suite["name"], generator_config=gen, judge_version=args.judge_version),
           "report": suite_report(pw), "outputs": outputs, "verdicts": pw}
    _write(args.out, out)
    print(json.dumps(out["report"], indent=2))


def cmd_compare(args):
    suite = load_suite(args.suite)
    cfg_a, cfg_b = load_config(args.a), load_config(args.b)
    out_a, out_b = _outputs(suite["cases"], cfg_a), _outputs(suite["cases"], cfg_b)
    pa = _pointwise_all(suite["cases"], out_a, args.judge_version)
    pb = _pointwise_all(suite["cases"], out_b, args.judge_version)
    with ThreadPoolExecutor(2) as ex:
        pairs = list(ex.map(lambda c: judge_pairwise(c, out_a[c["id"]], out_b[c["id"]], args.judge_version),
                            suite["cases"]))
    for p in pairs:
        log.info("pairwise %-8s AB=%s BA=%s final=%s", p["case_id"], p["verdict_AB"], p.get("verdict_BA"), p["final"])
    rep_a, rep_b, rep_pw = suite_report(pa), suite_report(pb), pairwise_report(pairs)
    decision = declare_winner(rep_a, rep_b, rep_pw, pa, pb)
    out = {**_meta(suite=suite["name"], judge_version=args.judge_version),
           "config_A": cfg_a, "config_B": cfg_b,
           "report_A": rep_a, "report_B": rep_b, "pairwise": rep_pw, "decision": decision,
           "outputs_A": out_a, "outputs_B": out_b, "pointwise_A": pa, "pointwise_B": pb, "pairs": pairs}
    _write(args.out, out)
    print(json.dumps({"A": rep_a, "B": rep_b, "pairwise": rep_pw, "decision": decision}, indent=2))


def cmd_decide(args):
    """Recompute reports + decision from a saved compare run (no LLM calls)."""
    d = json.loads(Path(args.report).read_text(encoding="utf-8"))
    d["report_A"], d["report_B"] = suite_report(d["pointwise_A"]), suite_report(d["pointwise_B"])
    d["pairwise"] = pairwise_report(d["pairs"])
    d["decision"] = declare_winner(d["report_A"], d["report_B"], d["pairwise"], d["pointwise_A"], d["pointwise_B"])
    _write(args.report, d)
    print(json.dumps({"A": d["report_A"], "B": d["report_B"], "pairwise": d["pairwise"], "decision": d["decision"]},
                     indent=2))


def cmd_replay(args):
    for line in Path(get_settings().call_log).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["request_hash"].startswith(args.hash):
            print(f"request_hash={rec['request_hash']}  ts={rec['ts']}  tag={rec['tag']}  model={rec['model']}  "
                  f"temperature={rec['temperature']}  json_mode={rec.get('json_mode')}\n")
            for m in rec["messages"]:
                body = m["content"]
                if args.brief and m["role"] == "system":
                    lines = body.splitlines()
                    body = "\n".join(lines[:6] + [f"  ... [{len(lines) - 6} more lines: rubric anchors, "
                                                  "calibration examples, rules - see full log] ..."])
                print(f"===== {m['role'].upper()} =====\n{body}\n")
            print(f"===== RAW RESPONSE ({rec['model']}, {rec['tokens']}) =====\n{rec['response']}")
            return
    raise SystemExit("hash not found")


def cmd_reparse(_args):
    from collections import Counter

    from judge.parsing import ParseError, parse_json
    st = Counter()
    for line in Path(get_settings().call_log).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["role"] != "judge":
            continue
        try:
            st[parse_json(rec["response"])[1]] += 1
        except ParseError:
            st["unparseable_first_try"] += 1
    print(json.dumps(dict(st), indent=2))


def _write(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", p)


def main():
    _setup_logging()
    ap = argparse.ArgumentParser(prog="judge.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--suite", required=True)
    r.add_argument("--gen", help="generator config; omit to judge the suite's model_output fields")
    r.add_argument("--judge-version", default="v1", choices=["v0", "v1"])
    r.add_argument("--out", default="results/run_report.json")
    r.set_defaults(func=cmd_run)
    c = sub.add_parser("compare")
    c.add_argument("--suite", required=True)
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--judge-version", default="v1", choices=["v0", "v1"])
    c.add_argument("--out", default="results/ab_report.json")
    c.set_defaults(func=cmd_compare)
    d = sub.add_parser("decide", help="recompute report + winner from a saved compare run")
    d.add_argument("--report", default="results/ab_report.json")
    d.set_defaults(func=cmd_decide)
    rp = sub.add_parser("replay")
    rp.add_argument("--hash", required=True)
    rp.add_argument("--brief", action="store_true", help="abbreviate the long system prompt for screenshots")
    rp.set_defaults(func=cmd_replay)
    sub.add_parser("reparse").set_defaults(func=cmd_reparse)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
