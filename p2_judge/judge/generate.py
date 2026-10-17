"""Produce model_output for each case under a generator config (prompt version and/or model)."""
from judge.llm import chat


def generate(case: dict, config: dict) -> dict:
    system = "\n\n".join(x for x in (config.get("preamble", ""), case.get("system_prompt", "")) if x)
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": case["input"] + (f"\n\nContext:\n{case['context']}" if case.get("context") else "")}]
    call = chat("generator", messages, model=config.get("model"), temperature=config.get("temperature"),
                max_tokens=config.get("max_tokens", 800), tag=f"generate:{config['name']}:{case['id']}")
    return {"case_id": case["id"], "config": config["name"], "model": call.model, "output": call.text.strip(),
            "tokens": {"prompt": call.prompt_tokens, "completion": call.completion_tokens},
            "latency_ms": call.latency_ms, "cache_hit": call.cache_hit}
