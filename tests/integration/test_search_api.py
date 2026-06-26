import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from api.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.anyio
async def test_register_and_login(client):
    # Register
    r = await client.post("/auth/register", json={
        "email": "test@example.com",
        "password": "testpassword123"
    })
    assert r.status_code == 201
    assert "access_token" in r.json()

    # Login
    r = await client.post("/auth/login", json={
        "email": "test@example.com",
        "password": "testpassword123"
    })
    assert r.status_code == 200
    assert "access_token" in r.json()


@pytest.mark.anyio
async def test_search_requires_auth(client):
    r = await client.post("/search", params={"query": "binary search"})
    assert r.status_code == 403


@pytest.mark.anyio
async def test_register_duplicate_email(client):
    payload = {"email": "dup@example.com", "password": "pass123"}
    await client.post("/auth/register", json=payload)
    r = await client.post("/auth/register", json=payload)
    assert r.status_code == 400
