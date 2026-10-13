"""
Unit tests for Camera routes.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from vsl.api.server import app


@pytest.mark.asyncio
async def test_get_cameras_api():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/cameras")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "total" in data
    assert "cameras" in data
    assert isinstance(data["cameras"], list)
