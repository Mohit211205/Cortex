from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Cortex"
    DEBUG: bool = False
    SECRET_KEY: str = "change-this-in-production"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://cortex:cortex@localhost:5432/cortex"

    # Redis
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

    class Config:
        env_file = ".env"


settings = Settings()
