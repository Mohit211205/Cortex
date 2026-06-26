import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from api.main import app


@pytest_asyncio.fixture
async def auth_client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/auth/register", json={
            "email": "ingest@example.com",
            "password": "password123"
        })
        token = r.json()["access_token"]
        c.headers.update({"Authorization": f"Bearer {token}"})
        yield c


@pytest.mark.anyio
async def test_ingest_requires_auth():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/repositories", json={"github_url": "https://github.com/test/repo"})
        assert r.status_code == 403


@pytest.mark.anyio
async def test_ingest_repo(auth_client):
    r = await auth_client.post("/repositories", json={
        "github_url": "https://github.com/test/repo",
        "name": "test-repo"
    })
    assert r.status_code == 202
    data = r.json()
    assert data["status"] == "pending"
    assert "id" in data


@pytest.mark.anyio
async def test_get_repo_status(auth_client):
    r = await auth_client.post("/repositories", json={
        "github_url": "https://github.com/test/repo2",
        "name": "test-repo-2"
    })
    repo_id = r.json()["id"]

    r = await auth_client.get(f"/repositories/{repo_id}/status")
    assert r.status_code == 200
    assert r.json()["id"] == repo_id


@pytest.mark.anyio
async def test_delete_repo(auth_client):
    r = await auth_client.post("/repositories", json={
        "github_url": "https://github.com/test/repo3",
        "name": "test-repo-3"
    })
    repo_id = r.json()["id"]

    r = await auth_client.delete(f"/repositories/{repo_id}")
    assert r.status_code == 204
