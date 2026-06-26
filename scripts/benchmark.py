"""
Benchmark HNSW search latency at different index sizes.
Run: python scripts/benchmark.py
"""

import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.hnsw import HNSWIndex

DIM = 768
SIZES = [1_000, 10_000, 50_000]
K = 10
QUERIES = 100


def benchmark(n: int):
    print(f"\n--- Index size: {n:,} vectors ---")
    idx = HNSWIndex(dim=DIM, M=16, ef_construction=200, ef_search=50)

    # Build index
    print(f"Building index...", end=" ", flush=True)
    t0 = time.time()
    for i in range(n):
        v = np.random.randn(DIM).astype(np.float32)
        idx.add(i, v)
    build_time = time.time() - t0
    print(f"{build_time:.1f}s")

    # Search latency
    latencies = []
    for _ in range(QUERIES):
        q = np.random.randn(DIM).astype(np.float32)
        t0 = time.time()
        idx.search(q, k=K)
        latencies.append((time.time() - t0) * 1000)

    latencies.sort()
    p50 = latencies[int(QUERIES * 0.50)]
    p95 = latencies[int(QUERIES * 0.95)]
    p99 = latencies[int(QUERIES * 0.99)]

    print(f"Search latency over {QUERIES} queries:")
    print(f"  P50: {p50:.1f}ms")
    print(f"  P95: {p95:.1f}ms")
    print(f"  P99: {p99:.1f}ms")


if __name__ == "__main__":
    print("Cortex HNSW Benchmark")
    print("=" * 40)
    for size in SIZES:
        benchmark(size)
    print("\nDone.")
