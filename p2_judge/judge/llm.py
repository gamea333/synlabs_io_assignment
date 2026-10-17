"""OpenAI-compatible client with an audit log that doubles as a replay cache.

Every request is hashed (model, messages, temperature, max_tokens, json_mode, sample). The log line holds
the full prompt, the raw response text, token usage, latency and throttle time. A later identical
request is served from the log (cache_hit=True) — that is what makes a run auditable AND replayable.
`sample` lets us deliberately bypass the cache (test-retest draws, temperature > 0).
"""
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError

from judge.config import get_settings

log = logging.getLogger("llm")
_lock = threading.Lock()


@dataclass
class Call:
    text: str
    model: str
    role: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    throttle_ms: float
    cache_hit: bool
    request_hash: str


@lru_cache
def _client(role: str) -> OpenAI:
    s = get_settings()
    base, key = ((s.judge_base_url, s.judge_api_key) if role == "judge"
                 else (s.generator_base_url, s.generator_api_key))
    return OpenAI(base_url=base, api_key=key or "unused", max_retries=0, timeout=90)


@lru_cache
def _cache() -> dict:
    path = Path(get_settings().call_log)
    cache = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("ok"):
                cache[rec["request_hash"]] = rec
    return cache


def _append(rec: dict) -> None:
    path = Path(get_settings().call_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def request_hash(model: str, messages: list[dict], temperature: float, max_tokens: int,
                 json_mode: bool, sample: int) -> str:
    blob = json.dumps([model, messages, temperature, max_tokens, json_mode, sample], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def chat(role: str, messages: list[dict], model: str | None = None, temperature: float | None = None,
         max_tokens: int = 1200, json_mode: bool = False, sample: int = 0, tag: str = "",
         max_attempts: int = 10) -> Call:
    s = get_settings()
    model = model or (s.judge_model if role == "judge" else s.generator_model)
    temperature = (s.judge_temperature if role == "judge" else s.generator_temperature) \
        if temperature is None else temperature
    h = request_hash(model, messages, temperature, max_tokens, json_mode, sample)

    hit = _cache().get(h)
    if hit:
        return Call(hit["response"], model, role, hit["tokens"]["prompt"], hit["tokens"]["completion"],
                    hit["latency_ms"], 0.0, True, h)

    kwargs = {}
    if "gpt-oss" in model:
        kwargs["reasoning_effort"] = "low"
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    throttle_ms = 0.0
    for attempt in range(max_attempts):
        t0 = time.perf_counter()
        try:
            r = _client(role).chat.completions.create(model=model, messages=messages, temperature=temperature,
                                                      max_tokens=max_tokens, **kwargs)
        except (RateLimitError, APIConnectionError, InternalServerError) as e:
            if attempt == max_attempts - 1 or "Request too large" in str(e):
                raise   # a request that can never fit the provider's per-minute cap must not be retried
            wait = min(2 ** attempt, 30)
            if isinstance(e, RateLimitError) and e.response is not None:
                try:
                    wait = float(e.response.headers.get("retry-after", wait)) + 0.5
                except ValueError:
                    pass
            log.warning("%s on %s, waiting %.1fs", type(e).__name__, model, wait)
            time.sleep(wait)
            throttle_ms += (time.perf_counter() - t0) * 1000
            continue
        except Exception as e:
            # e.g. Groq json_validate_failed (400): the model produced invalid JSON in JSON mode.
            # Log it as an auditable failed call and return the provider's failed_generation if present
            # so the robust parser still gets a chance.
            body = getattr(e, "body", None) or {}
            failed = (body.get("error", {}) if isinstance(body, dict) else {}).get("failed_generation", "")
            latency = (time.perf_counter() - t0) * 1000
            _append({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "request_hash": h, "ok": False, "role": role,
                     "model": model, "tag": tag, "messages": messages, "temperature": temperature,
                     "response": failed, "error": str(e)[:500],
                     "tokens": {"prompt": 0, "completion": 0}, "latency_ms": round(latency, 1)})
            if failed:
                return Call(failed, model, role, 0, 0, round(latency, 1), round(throttle_ms, 1), False, h)
            raise
        latency = (time.perf_counter() - t0) * 1000
        text = r.choices[0].message.content or ""
        u = r.usage
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "request_hash": h, "ok": True, "role": role,
               "model": model, "tag": tag, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens, "json_mode": json_mode, "sample": sample, "response": text,
               "tokens": {"prompt": u.prompt_tokens if u else 0, "completion": u.completion_tokens if u else 0},
               "latency_ms": round(latency, 1), "throttle_ms": round(throttle_ms, 1)}
        _append(rec)
        _cache()[h] = rec
        return Call(text, model, role, rec["tokens"]["prompt"], rec["tokens"]["completion"],
                    rec["latency_ms"], rec["throttle_ms"], False, h)
    raise AssertionError("unreachable")
