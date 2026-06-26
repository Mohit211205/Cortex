import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.dependencies import get_current_user
from db.session import get_db
from db.models import Repository, User

router = APIRouter(prefix="/repositories", tags=["ingest"])


class IngestRequest(BaseModel):
    github_url: str
    name: Optional[str] = None


class RepoResponse(BaseModel):
    id: str
    name: str
    github_url: str
    status: str


@router.post("", response_model=RepoResponse, status_code=202)
async def ingest_repository(
    body: IngestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    name = body.name or body.github_url.rstrip("/").split("/")[-1]

    repo = Repository(
        owner_id=current_user.id,
        name=name,
        github_url=body.github_url,
        status="pending",
    )
    db.add(repo)
    await db.commit()
    await db.refresh(repo)

    # Kick off async indexing
    background_tasks.add_task(_run_indexing, str(repo.id), body.github_url)

    return RepoResponse(id=str(repo.id), name=repo.name, github_url=repo.github_url, status=repo.status)


@router.get("/{repo_id}/status")
async def get_repo_status(
    repo_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Repository).where(Repository.id == repo_id, Repository.owner_id == current_user.id)
    )
    repo = result.scalar_one_or_none()
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    return {"id": str(repo.id), "name": repo.name, "status": repo.status, "indexed_at": repo.indexed_at}


@router.delete("/{repo_id}", status_code=204)
async def delete_repository(
    repo_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Repository).where(Repository.id == repo_id, Repository.owner_id == current_user.id)
    )
    repo = result.scalar_one_or_none()
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    from core.cache import QueryCache
    QueryCache().invalidate_repo(repo_id)

    await db.delete(repo)
    await db.commit()


async def _run_indexing(repo_id: str, github_url: str):
    """Background task: clone repo, chunk files, embed, index into HNSW."""
    from core.indexer.pipeline import IndexingPipeline
    pipeline = IndexingPipeline()
    await pipeline.run(repo_id, github_url)
