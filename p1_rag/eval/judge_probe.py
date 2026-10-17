"""Sanity-check the faithfulness judge: does it catch answers we deliberately corrupt?

  python -m eval.judge_probe   -> results/judge_probe.json

For each answered question in results/eval_results.json we re-retrieve the same context and judge
three variants:
  original   - the real system answer (expected: high faithfulness)
  num_swap   - first number in the answer replaced by a wrong one (expected: < 1)
  fabricated - original + an appended, cited, but unsupported claim (expected: < 1)
A judge that scores the corrupted variants 1.0 is not measuring faithfulness.
"""
import json
import re
import statistics
from pathlib import Path

from app.config import get_settings
from app.rag import _format_context, retrieve
from eval import judge

FABRICATED = " This approach was first deployed commercially by IBM in 1987 on a cluster of 4,096 GPUs [1]."


def swap_number(text: str) -> str | None:
    m = re.search(r"(?<![\w.])(\d+(?:[.,]\d+)?)", re.sub(r"\[\d+\]", "", text))
    if not m:
        return None
    old = m.group(1)
    new = str(int(float(old.replace(",", ""))) * 3 + 7)
    return text.replace(old, new, 1)


WORDS = {"two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
         "nine": "9", "ten": "10"}


def main() -> None:
    s = get_settings()
    res = json.loads(Path("results/eval_results.json").read_text(encoding="utf-8"))
    rows = []
    for p in res["answers"]["per_question"]:
        if not p["answered"]:
            continue
        hits, _ = retrieve(p["question"], res["config"]["k"])
        ctx = _format_context([h for h in hits if h.score >= s.min_score])
        original = p["answer"]
        numeric = re.sub(r"\b(" + "|".join(WORDS) + r")\b", lambda m: WORDS[m.group(1).lower()],
                         original, flags=re.I)
        variants = {"original": original, "fabricated": original + FABRICATED}
        swapped = swap_number(numeric)
        if swapped:
            variants["num_swap"] = swapped
        row = {"id": p["id"]}
        for name, text in variants.items():
            f, audit = judge.faithfulness(text, ctx)
            row[name] = {"faithfulness": f, "text": text,
                         "unsupported": [c["claim"] for c in audit["parsed"].get("claims", [])
                                         if c.get("verdict", "").lower() != "supported"]}
        rows.append(row)
        print(p["id"], {k: v["faithfulness"] for k, v in row.items() if k != "id"})

    def summary(name):
        vals = [r[name]["faithfulness"] for r in rows if name in r and r[name]["faithfulness"] is not None]
        return {"n": len(vals), "mean_faithfulness": round(statistics.mean(vals), 4),
                "caught_rate": round(sum(v < 1.0 for v in vals) / len(vals), 4)}

    out = {"judge_model": s.judge_model,
           "summary": {n: summary(n) for n in ("original", "num_swap", "fabricated")},
           "note": "caught_rate = fraction of variants scored < 1.0. For 'original' this is the false-alarm rate.",
           "rows": rows}
    Path("results/judge_probe.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
