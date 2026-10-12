"""
Unit tests for FastAPI REST Endpoints.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from vsl.api.server import app


@pytest.mark.asyncio
async def test_get_system_stats():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/system_stats")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "device" in data
    assert data["input_features"] == 258


@pytest.mark.asyncio
async def test_get_dictionary_api():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/dictionary")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total"] == 20
    assert len(data["data"]) == 20
