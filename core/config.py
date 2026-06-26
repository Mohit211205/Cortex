from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Cortex"
    DEBUG: bool = False
    SECRET_KEY: str = "change-this-in-production"

    # Database — Railway gives postgresql://, asyncpg needs postgresql+asyncpg://
    DATABASE_URL: str = "postgresql+asyncpg://cortex:cortex@localhost:5432/cortex"

    # Redis — Railway gives a full REDIS_URL
    REDIS_URL: Optional[str] = None
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # JWT
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"

    # HNSW index path
    HNSW_INDEX_PATH: str = "data/hnsw.index"

    # Embeddings
    EMBEDDER_DEVICE: str = "cpu"

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # Rate limiting
    RATE_LIMIT: int = 100
    RATE_LIMIT_WINDOW: int = 60

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Fix DATABASE_URL prefix for asyncpg
        if self.DATABASE_URL.startswith("postgresql://"):
            object.__setattr__(self, "DATABASE_URL",
                self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1))
        elif self.DATABASE_URL.startswith("postgres://"):
            object.__setattr__(self, "DATABASE_URL",
                self.DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1))

        # Use REDIS_URL if provided (Railway sets this)
        if self.REDIS_URL:
            object.__setattr__(self, "CELERY_BROKER_URL", self.REDIS_URL + "/1")
            object.__setattr__(self, "CELERY_RESULT_BACKEND", self.REDIS_URL + "/2")

    class Config:
        env_file = ".env"


settings = Settings()
