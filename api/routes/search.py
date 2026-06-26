import time
from typing import List

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.dependencies import get_current_user, rate_limit
from core.cache import QueryCache
from core.embeddings import get_embedder
import core.state as state
from db.session import get_db
from db.models import CodeChunk, SearchLog, User

router = APIRouter(prefix="/search", tags=["search"])

_cache = QueryCache()


class SearchResult(BaseModel):
    chunk_id: str
    file_path: str
    content: str
    language: str
    start_line: int
    end_line: int
    score: float
    repo_id: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    latency_ms: int
    cached: bool


@router.post("", response_model=SearchResponse, dependencies=[Depends(rate_limit)])
async def semantic_search(
    query: str,
    k: int = Query(default=10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    start = time.time()

    # Check cache
    cached = _cache.get(query, k)
    if cached:
        latency = int((time.time() - start) * 1000)
        return SearchResponse(query=query, results=cached, latency_ms=latency, cached=True)

    # Embed query
    embedder = get_embedder()
    query_vec = embedder.embed(query)

    # Search HNSW index
    hits = state.hnsw_index.search(query_vec, k=k)

    # Fetch chunk metadata from DB
    results = []
    for node_id, dist in hits:
        result = await db.execute(
            select(CodeChunk).where(CodeChunk.embedding_id == str(node_id))
        )
        chunk = result.scalar_one_or_none()
        if chunk:
            results.append(SearchResult(
                chunk_id=str(chunk.id),
                file_path=chunk.file_path,
                content=chunk.chunk_content,
                language=chunk.language or "",
                start_line=chunk.start_line or 0,
                end_line=chunk.end_line or 0,
                score=round(1.0 - dist, 4),
                repo_id=str(chunk.repo_id),
            ))

    latency = int((time.time() - start) * 1000)

    # Cache results
    _cache.set(query, k, [r.dict() for r in results])

    # Log search
    log = SearchLog(
        user_id=current_user.id,
        query=query,
        latency_ms=latency,
        result_count=len(results),
    )
    db.add(log)
    await db.commit()

    return SearchResponse(query=query, results=results, latency_ms=latency, cached=False)


@router.get("/history")
async def search_history(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = Query(default=20, ge=1, le=100),
):
    result = await db.execute(
        select(SearchLog)
        .where(SearchLog.user_id == current_user.id)
        .order_by(SearchLog.created_at.desc())
        .limit(limit)
    )
    logs = result.scalars().all()
    return [
        {
            "query": l.query,
            "latency_ms": l.latency_ms,
            "result_count": l.result_count,
            "created_at": l.created_at,
        }
        for l in logs
    ]
