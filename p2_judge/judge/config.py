"""Runtime config from env / .env. Judge and generator have separate keys so they can differ."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    judge_base_url: str = "https://api.groq.com/openai/v1"
    judge_api_key: str = ""
    judge_model: str = "qwen/qwen3.8-27b"
    judge_temperature: float = 0.0

    generator_base_url: str = "https://api.groq.com/openai/v1"
    generator_api_key: str = ""
    generator_model: str = "openai/gpt-oss-20b"
    generator_temperature: float = 0.3

    call_log: str = "logs/calls.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
