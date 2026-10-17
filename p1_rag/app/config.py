"""All runtime config comes from env / .env — nothing secret is hardcoded."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-oss-20b"
    judge_model: str = "qwen/qwen3.8-27b"

    embed_model: str = "BAAI/bge-small-en-v1.5"

    chunk_size: int = 256      # tokens (cl100k); chosen by eval/sweep.py
    chunk_overlap: int = 32    # tokens (12.5%)

    top_k: int = 5
    min_score: float = 0.50    # gate-1 cosine floor; see .env.example for calibration

    lancedb_uri: str = "data/lancedb"
    lancedb_table: str = "chunks"
    log_path: str = "logs/queries.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
