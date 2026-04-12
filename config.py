"""
config.py
─────────
Centralised, type-safe configuration loaded from environment variables /
.env file via Pydantic Settings.  Import `settings` everywhere — never
read os.environ directly.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ──────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI secret key")

    # ── Model config ────────────────────────────────────────────────────
    llm_model: str = Field("gpt-4o", description="Chat model for scoring & parsing")
    embedding_model: str = Field(
        "text-embedding-3-small", description="Embedding model for vector store"
    )
    llm_temperature: float = Field(0.1, ge=0.0, le=1.0)

    # ── Chunking ────────────────────────────────────────────────────────
    chunk_size: int = Field(512, gt=0)
    chunk_overlap: int = Field(50, ge=0)
    top_k_retrieval: int = Field(8, gt=0)

    # ── Scoring ─────────────────────────────────────────────────────────
    min_score_threshold: float = Field(
        30.0, description="Scores below this are flagged as 'no match'"
    )
    min_confidence_threshold: float = Field(
        0.3, description="Confidence below this triggers 'insufficient data' warning"
    )
    scoring_runs: int = Field(
        1,
        description="Number of independent scoring calls to average (set 3 for prod)",
    )

    # ── ChromaDB ────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field("./chroma_db")

    # ── Logging ─────────────────────────────────────────────────────────
    log_level: str = Field("INFO")


# Singleton — import this everywhere
settings = Settings()