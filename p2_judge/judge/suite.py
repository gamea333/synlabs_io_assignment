"""Load and validate a test suite (JSON or YAML).

Case schema: {id, input, system_prompt, model_output?, expected_output?, criteria?, context?, tags?}
model_output may be absent when outputs are produced by a generator config (A/B runs).
"""
import json
from pathlib import Path

import yaml

REQUIRED = ("id", "input")


def load_suite(path: str | Path) -> dict:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if p.suffix.lower() in (".yaml", ".yml") else json.loads(text)
    cases = data["cases"] if isinstance(data, dict) else data
    ids = set()
    for c in cases:
        missing = [k for k in REQUIRED if not c.get(k)]
        if missing:
            raise ValueError(f"case {c.get('id')!r} missing {missing}")
        if c["id"] in ids:
            raise ValueError(f"duplicate case id {c['id']!r}")
        ids.add(c["id"])
        c.setdefault("system_prompt", "")
    return {"name": data.get("name", p.stem) if isinstance(data, dict) else p.stem,
            "description": data.get("description", "") if isinstance(data, dict) else "", "cases": cases}


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
