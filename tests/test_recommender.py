"""Unit tests for Recommender Engine: SVD Collaborative Filtering, Content Filtering, and Hybrid MMR Re-ranking."""
import pytest
import pandas as pd
from src.data_loader import load_data
from src.collaborative_model import CollaborativeFilteringModel
from src.content_filter import ContentFilter
from src.hybrid_reranker import HybridReranker
from src.explainer import RecommendationExplainer
from src.evaluator import OfflineEvaluator

@pytest.fixture(scope="module")
def data():
    return load_data()

def test_collaborative_filtering_fit_and_predict(data):
    df_movies, df_ratings, _ = data
    cf = CollaborativeFilteringModel(n_factors=5, random_state=42)
    cf.fit(df_ratings, df_movies)

    assert cf.is_fitted
    # Test prediction for known user
    pred = cf.predict_user_movie(user_id=1, movie_id=1)
    assert 1.0 <= pred <= 5.0

    # Test cold start fallback
    cold_pred = cf.predict_user_movie(user_id=99999, movie_id=1)
    assert 1.0 <= cold_pred <= 5.0

def test_active_learning_session_projection(data):
    df_movies, df_ratings, _ = data
    cf = CollaborativeFilteringModel(n_factors=5, random_state=42)
    cf.fit(df_ratings, df_movies)

    # Empty session ratings -> cold start Bayesian priors
    all_mids = list(df_movies["movie_id"])
    cold_scores = cf.predict_session_user({}, all_mids)
    assert len(cold_scores) == len(all_mids)

    # Dynamic session ratings (user gives 5 stars to comedy/uplifting)
    session_ratings = {1: 5.0, 2: 5.0, 3: 5.0}
    personalized_scores = cf.predict_session_user(session_ratings, all_mids)
    assert len(personalized_scores) == len(all_mids)
    # The personalized score for movie 1 should remain high
    assert personalized_scores[1] >= 3.5

def test_content_filter_scores(data):
    df_movies, _, df_tracks = data
    cf_filter = ContentFilter()

    # Target: High Valence, High Energy (Joy)
    scored_tracks = cf_filter.score_music_tracks(df_tracks, target_valence=0.9, target_energy=0.85, dominant_emotion="Joy")
    assert "content_score" in scored_tracks.columns
    # Top track should have high score
    assert scored_tracks["content_score"].max() > 0.70

    scored_movies = cf_filter.score_movies(df_movies, target_valence=0.9, target_energy=0.85, dominant_emotion="Joy")
    assert "content_score" in scored_movies.columns
    assert scored_movies["content_score"].max() > 0.70

def test_hybrid_reranker_and_mmr_diversity(data):
    df_movies, df_ratings, df_tracks = data
    cf_filter = ContentFilter()
    cf = CollaborativeFilteringModel(n_factors=5, random_state=42).fit(df_ratings, df_movies)
    reranker = HybridReranker()

    scored_movies = cf_filter.score_movies(df_movies, target_valence=0.85, target_energy=0.8, dominant_emotion="Joy")
    cf_scores = cf.predict_session_user({}, list(df_movies["movie_id"]))

    # Test top-k size
    top_picks = reranker.rerank_movies(scored_movies, cf_scores, novelty_weight=0.2, diversity_weight=0.35, top_k=5)
    assert len(top_picks) == 5
    assert all("relevance_score" in p for p in top_picks)
    assert all("novelty_score" in p for p in top_picks)

def test_explainer_rationales():
    explainer = RecommendationExplainer()
    sample_movie = {
        "title": "Singin' in the Rain",
        "vote_average": 8.3,
        "mood_valence": 0.92,
        "mood_energy": 0.78,
        "genres": ["Comedy", "Musical"],
        "cf_rating": 4.5,
        "novelty_score": 0.2
    }
    mood_info = {"dominant_emotion": "Uplift", "intent": "mood_regulation"}
    explanation = explainer.explain_movie(sample_movie, mood_info)
    assert isinstance(explanation, str)
    assert len(explanation) > 10

def test_offline_evaluator_metrics(data):
    df_movies, df_ratings, df_tracks = data
    cf = CollaborativeFilteringModel(n_factors=5, random_state=42).fit(df_ratings, df_movies)
    cf_filter = ContentFilter()
    reranker = HybridReranker()

    metrics = OfflineEvaluator.evaluate_benchmark(reranker, cf_filter, cf, df_movies, df_tracks)
    assert metrics["mean_ndcg_at_5"] > 0.60
    assert metrics["mean_map_at_5"] > 0.50
    assert metrics["mean_movie_diversity"] > 0.30
    assert len(metrics["diversity_vs_accuracy_tradeoff"]) == 6
