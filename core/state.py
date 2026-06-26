"""
Global shared state — holds the HNSW index singleton.
Imported by both api/main.py and any module that needs the index.
Avoids circular imports.
"""

from __future__ import annotations
from typing import Optional
from core.hnsw import HNSWIndex

# Set at startup by api/main.py lifespan
hnsw_index: Optional[HNSWIndex] = None
