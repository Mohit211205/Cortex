import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator

from core.config import settings
from core.hnsw import HNSWIndex
from core.embeddings import CodeBERTEmbedder
from db.models import Base
from db.session import engine

# Global HNSW index — loaded at startup, shared across requests
hnsw_index: HNSWIndex = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global hnsw_index

    # Create DB tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Load or create HNSW index
    if os.path.exists(settings.HNSW_INDEX_PATH):
        hnsw_index = HNSWIndex.load(settings.HNSW_INDEX_PATH)
    else:
        hnsw_index = HNSWIndex(dim=CodeBERTEmbedder.DIM)

    yield

    # Persist index on shutdown
    os.makedirs(os.path.dirname(settings.HNSW_INDEX_PATH), exist_ok=True)
    hnsw_index.save(settings.HNSW_INDEX_PATH)


app = FastAPI(
    title="Cortex — Semantic Code Search",
    description="Search your codebase by meaning, not keywords.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)

# Routers
from api.routes import auth, search, ingest

app.include_router(auth.router)
app.include_router(search.router)
app.include_router(ingest.router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_size": len(hnsw_index) if hnsw_index else 0,
    }


# Serve web UI
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")
