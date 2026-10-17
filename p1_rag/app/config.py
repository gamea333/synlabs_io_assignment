"""All runtime config comes from env / .env — nothing secret is hardcoded."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "llama-3.1-8b-instant"
    judge_model: str = "llama-3.3-70b-versatile"

    embed_model: str = "BAAI/bge-small-en-v1.5"

    chunk_size: int = 512      # tokens (cl100k)
    chunk_overlap: int = 64    # tokens

    top_k: int = 5
    min_score: float = 0.55    # cosine similarity floor for "relevant context"

    lancedb_uri: str = "data/lancedb"
    lancedb_table: str = "chunks"
    log_path: str = "logs/queries.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
