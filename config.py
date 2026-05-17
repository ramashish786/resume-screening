from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: str = Field(..., description="OpenAI secret key")

    llm_model: str = Field("gpt-4o")
    embedding_model: str = Field("text-embedding-3-small")
    llm_temperature: float = Field(0.1, ge=0.0, le=1.0)

    chunk_size: int = Field(512, gt=0)
    chunk_overlap: int = Field(50, ge=0)
    top_k_retrieval: int = Field(8, gt=0)

    min_score_threshold: float = Field(30.0)
    min_confidence_threshold: float = Field(0.3)
    scoring_runs: int = Field(1)  # set to 3 in prod for self-consistency averaging

    chroma_persist_dir: str = Field("./chroma_db")

    smtp_host: str = Field("smtp.gmail.com")
    smtp_port: int = Field(587)
    smtp_user: str = Field("")
    smtp_password: str = Field("")
    smtp_from_name: str = Field("Resume Scoring Agent")

    log_level: str = Field("INFO")


# Singleton — import this everywhere
settings = Settings()