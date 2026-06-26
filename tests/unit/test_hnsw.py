import numpy as np
import pytest
from core.hnsw import HNSWIndex


def make_index(dim=64):
    return HNSWIndex(dim=dim, M=8, ef_construction=50, ef_search=20)


def random_vec(dim=64):
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_empty_search():
    idx = make_index()
    results = idx.search(random_vec(), k=5)
    assert results == []


def test_single_insert_search():
    idx = make_index()
    v = random_vec()
    idx.add(0, v)
    results = idx.search(v, k=1)
    assert len(results) == 1
    assert results[0][0] == 0
    assert results[0][1] < 1e-4   # distance to itself is ~0


def test_multiple_inserts():
    idx = make_index()
    vecs = [random_vec() for _ in range(50)]
    for i, v in enumerate(vecs):
        idx.add(i, v)
    assert len(idx) == 50


def test_nearest_neighbour_correct():
    idx = make_index()
    target = random_vec()
    # Add target with small noise variants and unrelated vectors
    idx.add(0, target)
    for i in range(1, 20):
        idx.add(i, random_vec())

    results = idx.search(target, k=1)
    assert results[0][0] == 0   # target itself should be closest


def test_k_results():
    idx = make_index()
    for i in range(30):
        idx.add(i, random_vec())
    results = idx.search(random_vec(), k=10)
    assert len(results) <= 10


def test_save_load(tmp_path):
    idx = make_index()
    for i in range(20):
        idx.add(i, random_vec())

    path = str(tmp_path / "test.index")
    idx.save(path)

    loaded = HNSWIndex.load(path)
    assert len(loaded) == 20

    q = random_vec()
    r1 = idx.search(q, k=5)
    r2 = loaded.search(q, k=5)
    assert [x[0] for x in r1] == [x[0] for x in r2]
