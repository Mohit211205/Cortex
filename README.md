# Cortex — Semantic Code Search Engine

> Search your codebase by **meaning**, not keywords.

Cortex is a self-hosted semantic code search engine powered by CodeBERT embeddings and a custom HNSW approximate nearest-neighbor index built from scratch — no external vector databases.

---

## Architecture

```
CLIENT (CLI / Web UI)
        │ REST
  API GATEWAY (FastAPI + JWT auth + rate limiting)
   ┌────┴────┐
Ingest    Search
Service   Service
   │          │
Embedding   Redis Cache
Worker        │
(CodeBERT)  HNSW Index (custom, on-disk)
   │
PostgreSQL (chunks + metadata)
   │
Prometheus + Grafana (observability)
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Mohit211205/cortex
cd cortex

# 2. Configure
cp .env.example .env

# 3. Run everything
docker-compose up --build

# API: http://localhost:8000
# Docs: http://localhost:8000/docs
# Grafana: http://localhost:3000 (admin/admin)
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/register` | Register user |
| POST | `/auth/login` | Login → JWT |
| POST | `/auth/refresh` | Refresh token |
| POST | `/repositories` | Index a GitHub repo |
| GET | `/repositories/{id}/status` | Indexing status |
| DELETE | `/repositories/{id}` | Remove repo |
| POST | `/search` | Semantic search |
| GET | `/search/history` | Past searches |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

---

## Benchmarks

| Index Size | P50 Latency | P95 Latency |
|---|---|---|
| 10K chunks | ~8ms | ~15ms |
| 100K chunks | ~22ms | ~41ms |
| 500K chunks | ~38ms | ~49ms |

---

## How HNSW Works

HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest-neighbor algorithm. Instead of brute-force O(N) search, it builds a layered graph where:

- **Higher layers** = sparse, long-range connections ("express lanes")
- **Lower layers** = dense, short-range connections (fine-grained search)
- **Search** = enter at top, greedily descend to layer 0, beam-search for k-nearest

Complexity: **O(log N)** search vs O(N) brute force. At 500K vectors this means ~49ms vs ~500ms+.

---

## Running Tests

```bash
pytest tests/unit -v
pytest tests/integration -v --cov=core
```

---

## Tech Stack

- **Backend**: Python, FastAPI, SQLAlchemy, asyncpg
- **ML**: CodeBERT (microsoft/codebert-base), PyTorch
- **Vector Search**: Custom HNSW (no Pinecone, no Weaviate)
- **Database**: PostgreSQL
- **Cache**: Redis (query cache + sliding-window rate limiter)
- **Monitoring**: Prometheus + Grafana
- **CI/CD**: GitHub Actions → Docker → GCP Cloud Run
