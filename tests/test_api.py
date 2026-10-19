"""API integration tests for FastAPI Endpoints."""
import pytest
from starlette.testclient import TestClient
from src.main import app

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["movies_loaded"] >= 30
    assert data["tracks_loaded"] >= 30

def test_api_presets(client):
    res = client.get("/api/presets")
    assert res.status_code == 200
    data = res.json()
    assert len(data["presets"]) >= 5

def test_api_analyze_mood(client):
    payload = {"mood_text": "exhausted but want something uplifting"}
    res = client.post("/api/analyze-mood", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["dominant_emotion"] in ["Uplift", "Joy"]
    assert data["valence"] >= 0.70

def test_api_recommend(client):
    payload = {
        "mood_text": "exhausted after a long week but want something uplifting",
        "novelty_weight": 0.20,
        "diversity_weight": 0.35,
        "top_k": 6
    }
    res = client.post("/api/recommend", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "mood_analysis" in data
    assert "recommendations" in data
    assert len(data["recommendations"]["movies"]) == 6
    assert len(data["recommendations"]["music"]) == 6

    # Verify explainability layer presence on all cards
    for m in data["recommendations"]["movies"]:
        assert "explanation" in m
        assert len(m["explanation"]) > 0
    for t in data["recommendations"]["music"]:
        assert "explanation" in t
        assert len(t["explanation"]) > 0

    assert data["meta"]["latency_ms"] < 250

def test_api_feedback_and_session_update(client):
    # Test movie rating
    fb_payload = {
        "item_type": "movie",
        "item_id": "1",
        "rating": 5.0
    }
    res = client.post("/api/feedback", json=fb_payload)
    assert res.status_code == 200
    assert res.json()["status"] == "success"

    # Test music like
    music_fb = {
        "item_type": "music",
        "item_id": "track_01",
        "action": "like"
    }
    res_m = client.post("/api/feedback", json=music_fb)
    assert res_m.status_code == 200

    # Verify session feedback
    sess_res = client.get("/api/session-feedback")
    assert sess_res.status_code == 200
    sess_data = sess_res.json()
    assert sess_data["movies"].get("1") == 5.0
    assert sess_data["music"].get("track_01") == 1

def test_api_metrics(client):
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "mean_ndcg_at_5" in data
    assert "mean_map_at_5" in data
    assert "diversity_vs_accuracy_tradeoff" in data
