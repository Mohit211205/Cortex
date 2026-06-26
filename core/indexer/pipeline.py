"""
Indexing pipeline:
1. Clone GitHub repo to temp dir
2. Walk all code files
3. Chunk each file (AST-based for Python, sliding window for others)
4. Embed each chunk via CodeBERT
5. Insert into HNSW index
6. Persist chunk metadata to PostgreSQL
7. Update repo status
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from sqlalchemy import update

from core.embeddings import get_embedder, chunk_file, chunk_hash
from db.models import CodeChunk, Repository
from db.session import AsyncSessionLocal

SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".java", ".go", ".cpp", ".c", ".rs"}
MAX_FILE_SIZE_KB = 500


class IndexingPipeline:
    async def run(self, repo_id: str, github_url: str):
        async with AsyncSessionLocal() as db:
            # Mark as indexing
            await db.execute(
                update(Repository).where(Repository.id == repo_id).values(status="indexing")
            )
            await db.commit()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                self._clone(github_url, tmpdir)
                await self._index_directory(repo_id, tmpdir)

            async with AsyncSessionLocal() as db:
                await db.execute(
                    update(Repository)
                    .where(Repository.id == repo_id)
                    .values(status="ready", indexed_at=datetime.utcnow())
                )
                await db.commit()

        except Exception as e:
            async with AsyncSessionLocal() as db:
                await db.execute(
                    update(Repository).where(Repository.id == repo_id).values(status="failed")
                )
                await db.commit()
            raise e

    def _clone(self, github_url: str, dest: str):
        subprocess.run(
            ["git", "clone", "--depth=1", github_url, dest],
            check=True,
            capture_output=True,
            timeout=120,
        )

    async def _index_directory(self, repo_id: str, directory: str):
        from api.main import hnsw_index

        embedder = get_embedder()
        node_id_counter = len(hnsw_index)

        async with AsyncSessionLocal() as db:
            for root, _, files in os.walk(directory):
                for fname in files:
                    fpath = Path(root) / fname
                    if fpath.suffix not in SUPPORTED_EXTENSIONS:
                        continue
                    if fpath.stat().st_size > MAX_FILE_SIZE_KB * 1024:
                        continue

                    try:
                        content = fpath.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        continue

                    rel_path = str(fpath.relative_to(directory))
                    chunks = chunk_file(content, fname)

                    texts = [c.content for c in chunks]
                    embeddings = embedder.embed_batch(texts)

                    for chunk, embedding in zip(chunks, embeddings):
                        node_id = node_id_counter
                        node_id_counter += 1

                        hnsw_index.add(node_id, embedding)

                        db_chunk = CodeChunk(
                            repo_id=repo_id,
                            file_path=rel_path,
                            chunk_content=chunk.content,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                            language=chunk.language,
                            embedding_id=str(node_id),
                            chunk_type=chunk.chunk_type,
                        )
                        db.add(db_chunk)

            await db.commit()

        # Persist updated HNSW index to disk
        from core.config import settings
        os.makedirs(os.path.dirname(settings.HNSW_INDEX_PATH), exist_ok=True)
        hnsw_index.save(settings.HNSW_INDEX_PATH)
