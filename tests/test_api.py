"""
Unit tests for FastAPI REST Endpoints (both Full Server and Serverless Vercel App).
"""

import pytest
from httpx import AsyncClient, ASGITransport
from vsl.api.server import app
from vsl.serverless.app import app as serverless_app


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


@pytest.mark.asyncio
async def test_demo_sequence_and_timeline_api():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        seq_resp = await ac.get("/api/predict_video/demo_sequence/Xin chao")
        assert seq_resp.status_code == 200
        seq_data = seq_resp.json()
        assert seq_data["status"] == "success"
        assert len(seq_data["frames"]) == 30
        assert seq_data["prediction"]["name"] == "Xin chao"

        # Test direct window prediction endpoint
        pred_resp = await ac.post("/api/predict", json={"frames": seq_data["frames"]})
        assert pred_resp.status_code == 200
        assert pred_resp.json()["name"] == "Xin chao"

        tl_resp = await ac.post("/api/predict_video/demo_timeline")
        assert tl_resp.status_code == 200
        tl_data = tl_resp.json()
        assert tl_data["status"] == "success"
        assert len(tl_data["timeline"]) == 5
        assert "Xin chào" in tl_data["sentence"]

        asset_resp = await ac.get("/assets/model_benchmark.png")
        assert asset_resp.status_code == 200


@pytest.mark.asyncio
async def test_serverless_vercel_app():
    async with AsyncClient(transport=ASGITransport(app=serverless_app), base_url="http://test") as ac:
        home = await ac.get("/")
        assert home.status_code == 200
        assert 'data-mode="serverless"' in home.text

        health = await ac.get("/api/health")
        assert health.status_code == 200
        assert health.json()["mode"] == "serverless"

        dict_resp = await ac.get("/api/dictionary")
        assert dict_resp.status_code == 200
        assert dict_resp.json()["total"] == 20

        seq_resp = await ac.get("/api/predict_video/demo_sequence/Viet Nam")
        assert seq_resp.status_code == 200
        seq_data = seq_resp.json()
        assert seq_data["prediction"]["name"] == "Viet Nam"

        pred_resp = await ac.post("/api/predict", json={"frames": seq_data["frames"]})
        assert pred_resp.status_code == 200
        assert pred_resp.json()["name"] == "Viet Nam"

        tl_resp = await ac.post("/api/predict_timeline", json={"frames": seq_data["frames"], "fps": 30.0})
        assert tl_resp.status_code == 200
        assert "Việt Nam" in tl_resp.json()["sentence"]
