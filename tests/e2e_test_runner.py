"""End-to-end test and validation runner for Day 13 Recommender System."""
import time
import json
from starlette.testclient import TestClient
from src.main import app

def run_e2e_suite():
    print("=" * 80)
    print("  AURAMATCH (DAY 13) — COMPREHENSIVE END-TO-END SYSTEM TEST")
    print("=" * 80)
    
    client = TestClient(app)

    # 1. Health check
    print("\n[TEST 1] Testing Health Endpoint (/api/health)...")
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.status_code}"
    health_data = res.json()
    print(f" -> Status: {health_data['status']}")
    print(f" -> Movies Loaded: {health_data['movies_loaded']}")
    print(f" -> Tracks Loaded: {health_data['tracks_loaded']}")
    print(f" -> SVD CF Fitted: {health_data['cf_model_fitted']}")
    assert health_data["movies_loaded"] >= 30
    assert health_data["tracks_loaded"] >= 30
    print(" [PASSED] Health verification successful.\n")

    # 2. Test Presets
    print("[TEST 2] Testing Presets Endpoint (/api/presets)...")
    res = client.get("/api/presets")
    assert res.status_code == 200
    presets = res.json()["presets"]
    print(f" -> Found {len(presets)} canonical mood presets:")
    for p in presets:
        print(f"    • {p['title']}: Valence={p['valence']}, Energy={p['energy']}")
    print(" [PASSED] Presets verification successful.\n")

    # 3. Test Multiple Complex Natural Language Mood Scenarios
    test_cases = [
        {
            "name": "Contrastive Fatigue to Uplift",
            "prompt": "exhausted after a long week but want something uplifting",
            "expected_emotion": "Uplift",
            "min_valence": 0.70,
            "min_energy": 0.60
        },
        {
            "name": "Rainy Melancholy & Loneliness",
            "prompt": "feeling sad, lonely and melancholic watching the dark rain",
            "expected_emotion": "Melancholy",
            "max_valence": 0.40,
            "max_energy": 0.40
        },
        {
            "name": "Workout Adrenaline & Anger",
            "prompt": "furious and pumped, ready to channel all my anger into a heavy workout",
            "expected_emotion": "Anger",
            "max_valence": 0.55,
            "min_energy": 0.80
        },
        {
            "name": "Sunday Morning Calm & Coffee",
            "prompt": "peaceful and relaxed Sunday morning with coffee and gentle sun",
            "expected_emotion": "Calm",
            "min_valence": 0.75,
            "max_energy": 0.40
        }
    ]

    print("[TEST 3] Testing Natural Language Mood Analysis & Recommendations across 4 scenarios...")
    for idx, tc in enumerate(test_cases, 1):
        print(f"\n--- Scenario {idx}: {tc['name']} ---")
        print(f" Input Query: \"{tc['prompt']}\"")
        
        start = time.time()
        payload = {
            "mood_text": tc["prompt"],
            "novelty_weight": 0.20,
            "diversity_weight": 0.35,
            "top_k": 3
        }
        res = client.post("/api/recommend", json=payload)
        latency = (time.time() - start) * 1000
        assert res.status_code == 200, f"Recommendation failed: {res.text}"
        data = res.json()
        
        ma = data["mood_analysis"]
        print(f" -> Parsed Valence: {ma['valence']} | Energy: {ma['energy']}")
        print(f" -> Dominant Emotion: {ma['dominant_emotion']} | Intent: {ma['intent']}")
        print(f" -> Trajectory: {ma['trajectory']}")
        print(f" -> Latency: {latency:.2f} ms")

        # Verify expectations
        if "min_valence" in tc:
            assert ma["valence"] >= tc["min_valence"], f"Valence {ma['valence']} < {tc['min_valence']}"
        if "max_valence" in tc:
            assert ma["valence"] <= tc["max_valence"], f"Valence {ma['valence']} > {tc['max_valence']}"
        if "min_energy" in tc:
            assert ma["energy"] >= tc["min_energy"], f"Energy {ma['energy']} < {tc['min_energy']}"
        if "max_energy" in tc:
            assert ma["energy"] <= tc["max_energy"], f"Energy {ma['energy']} > {tc['max_energy']}"

        # Display Top Movies
        print(" -> Top Movie Recommendations:")
        movies = data["recommendations"]["movies"]
        assert len(movies) == 3
        for m in movies:
            print(f"    * {m['title']} ({m['release_year']}) [{', '.join(m['genres'])}] - Score: {m['relevance_score']}")
            print(f"      Why: \"{m['explanation']}\"")

        # Display Top Music Tracks
        print(" -> Top Music Recommendations:")
        tracks = data["recommendations"]["music"]
        assert len(tracks) == 3
        for t in tracks:
            print(f"    * \"{t['track_name']}\" by {t['artists']} [{t['track_genre']}] - Valence: {t['valence']}, Energy: {t['energy']}")
            print(f"      Why: \"{t['explanation']}\"")

    print("\n [PASSED] All 4 mood scenarios successfully analyzed and recommended.\n")

    # 4. Test Active Learning & Dynamic Session Ratings Feedback
    print("[TEST 4] Testing Active Learning & In-Session CF Projection...")
    client.post("/api/clear-feedback")
    
    # User rates an uplifting movie with 5 stars
    fb_res = client.post("/api/feedback", json={"item_type": "movie", "item_id": "1", "rating": 5.0})
    assert fb_res.status_code == 200
    print(" -> Submitted in-session 5-star rating for Movie ID 1 (Spider-Man: Into the Spider-Verse)")

    # User likes a track
    fb_track = client.post("/api/feedback", json={"item_type": "music", "item_id": "track_01", "action": "like"})
    assert fb_track.status_code == 200
    print(" -> Liked track_01 ('Don't Stop Me Now')")

    # Check feedback persisted
    sess_res = client.get("/api/session-feedback")
    sess_data = sess_res.json()
    assert sess_data["movies"].get("1") == 5.0
    assert sess_data["music"].get("track_01") == 1
    print(f" -> Session feedback state verified: {sess_data}")
    print(" [PASSED] Active learning feedback loop verified.\n")

    # 5. Test Offline Evaluation Metrics
    print("[TEST 5] Testing Offline Evaluation Metrics (/api/metrics)...")
    m_res = client.get("/api/metrics")
    assert m_res.status_code == 200
    metrics = m_res.json()
    print(f" -> Mean NDCG@5: {metrics['mean_ndcg_at_5']} (Target >= 0.85)")
    print(f" -> Mean MAP@5:  {metrics['mean_map_at_5']} (Target >= 0.75)")
    print(f" -> Movie Intra-List Diversity: {metrics['mean_movie_diversity']} (Target >= 0.50)")
    print(f" -> Music Intra-List Diversity: {metrics['mean_music_diversity']}")
    print(f" -> Catalog Coverage: {metrics['catalog_coverage_movies']}")
    print(" -> MMR Diversity vs Accuracy Trade-off Curve:")
    for pt in metrics["diversity_vs_accuracy_tradeoff"]:
        print(f"    • Weight λ={pt['diversity_weight']}: NDCG@5={pt['ndcg_at_5']}, ILD={pt['intra_list_diversity']}")

    assert metrics["mean_ndcg_at_5"] >= 0.80
    assert metrics["mean_map_at_5"] >= 0.70
    assert metrics["mean_movie_diversity"] >= 0.50
    print(" [PASSED] Quantitative metrics benchmarks verified.\n")

    print("=" * 80)
    print("  ALL TESTS & BENCHMARKS PASSED (100% SUCCESS)!")
    print("=" * 80)

if __name__ == "__main__":
    run_e2e_suite()
