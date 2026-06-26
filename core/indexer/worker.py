"""
Celery worker for async repository indexing.
Broker: Redis
"""

from celery import Celery
from core.config import settings

celery_app = Celery(
    "cortex",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, max_retries=3)
def index_repository_task(self, repo_id: str, github_url: str):
    """
    Async Celery task for indexing a repository.
    Retries up to 3 times on failure with exponential backoff.
    """
    import asyncio
    from core.indexer.pipeline import IndexingPipeline

    try:
        pipeline = IndexingPipeline()
        asyncio.run(pipeline.run(repo_id, github_url))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
