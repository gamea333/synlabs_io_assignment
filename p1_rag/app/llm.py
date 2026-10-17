"""Thin client for any OpenAI-compatible chat endpoint (Groq, Ollama, OpenAI...)."""
import logging
import time
from dataclasses import dataclass
from functools import lru_cache

from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError

from app.config import get_settings

log = logging.getLogger("llm")


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    gen_ms: float = 0.0        # time spent in the successful call (true generation latency)
    throttle_ms: float = 0.0   # time spent waiting on 429s / retries (provider quota, not our system)
    retries: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@lru_cache
def get_client() -> OpenAI:
    s = get_settings()
    # Retries are done by hand below so rate-limit waits are measured, not hidden in latency.
    return OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key or "unused", max_retries=0, timeout=60)


def _retry_after(err: RateLimitError, attempt: int) -> float:
    headers = err.response.headers if err.response is not None else {}
    for h in ("retry-after",):
        if h in headers:
            try:
                return float(headers[h]) + 0.5
            except ValueError:
                pass
    return min(2 ** attempt, 30)


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.0,
         max_tokens: int = 1024, json_mode: bool = False, max_attempts: int = 8) -> LLMResult:
    model = model or get_settings().llm_model
    kwargs = {}
    if "gpt-oss" in model:
        kwargs["reasoning_effort"] = "low"   # reasoning tokens are billed; keep them small
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    throttle_ms = 0.0
    for attempt in range(max_attempts):
        t0 = time.perf_counter()
        try:
            r = get_client().chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
        except (RateLimitError, APIConnectionError, InternalServerError) as e:
            if attempt == max_attempts - 1:
                raise
            wait = _retry_after(e, attempt) if isinstance(e, RateLimitError) else min(2 ** attempt, 30)
            log.warning("%s on %s, waiting %.1fs (attempt %d)", type(e).__name__, model, wait, attempt + 1)
            time.sleep(wait)
            throttle_ms += (time.perf_counter() - t0) * 1000
            continue
        gen_ms = (time.perf_counter() - t0) * 1000
        u = r.usage
        return LLMResult((r.choices[0].message.content or "").strip(), model,
                         u.prompt_tokens if u else 0, u.completion_tokens if u else 0,
                         round(gen_ms, 1), round(throttle_ms, 1), attempt)
    raise AssertionError("unreachable")
