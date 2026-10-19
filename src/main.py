"""FastAPI Recommendation Service for Mood-Based Movie and Music Recommender."""
import os
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.data_loader import load_data
from src.mood_analyzer import MoodAnalyzer
from src.collaborative_model import CollaborativeFilteringModel
from src.content_filter import ContentFilter
from src.hybrid_reranker import HybridReranker
from src.explainer import RecommendationExplainer
from src.evaluator import OfflineEvaluator

app = FastAPI(
    title="Mood-Based Movie and Music Recommender API",
    description="Hybrid recommendation engine mapping natural language mood to cinematic and sonic experiences.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory engine state
df_movies, df_ratings, df_tracks = load_data()
mood_analyzer = MoodAnalyzer()
cf_model = CollaborativeFilteringModel(n_factors=8, random_state=42)
cf_model.fit(df_ratings, df_movies)
content_filter = ContentFilter()
reranker = HybridReranker()
explainer = RecommendationExplainer()

# Active-learning session state: in-memory user feedback
session_user_ratings: Dict[int, float] = {}  # movie_id -> rating (1.0 to 5.0)
session_track_likes: Dict[str, int] = {}     # track_id -> 1 (liked) or -1 (disliked)

# Pre-computed evaluation metrics cache
metrics_cache = OfflineEvaluator.evaluate_benchmark(reranker, content_filter, cf_model, df_movies, df_tracks)

# Request / Response Schemas
class MoodRequest(BaseModel):
    mood_text: str = Field(..., json_schema_extra={"example": "exhausted after a long week but want something uplifting"})
    valence_override: Optional[float] = Field(None, ge=0.0, le=1.0)
    energy_override: Optional[float] = Field(None, ge=0.0, le=1.0)
    novelty_weight: float = Field(0.20, ge=0.0, le=1.0)
    diversity_weight: float = Field(0.35, ge=0.0, le=1.0)
    top_k: int = Field(6, ge=1, le=20)

class FeedbackRequest(BaseModel):
    item_type: str = Field(..., description="'movie' or 'music'", json_schema_extra={"example": "movie"})
    item_id: str = Field(..., json_schema_extra={"example": "1"})
    rating: Optional[float] = Field(None, ge=1.0, le=5.0)
    action: Optional[str] = Field(None, description="'like' or 'dislike'", json_schema_extra={"example": "like"})

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "movies_loaded": len(df_movies),
        "tracks_loaded": len(df_tracks),
        "ratings_loaded": len(df_ratings),
        "cf_model_fitted": cf_model.is_fitted
    }

@app.get("/api/presets")
def get_mood_presets():
    return {
        "presets": [
            {
                "title": "Exhausted & Need Uplift",
                "text": "exhausted after a brutal week but want something uplifting to recharge",
                "valence": 0.88,
                "energy": 0.72,
                "icon": "sparkles"
            },
            {
                "title": "Rainy Night Melancholy",
                "text": "feeling sad and melancholic watching rain through the window, need catharsis",
                "valence": 0.25,
                "energy": 0.28,
                "icon": "cloud-rain"
            },
            {
                "title": "Sunday Morning Chill",
                "text": "drinking coffee on a warm peaceful Sunday morning with gentle acoustic vibes",
                "valence": 0.82,
                "energy": 0.35,
                "icon": "coffee"
            },
            {
                "title": "Late Night Drive & Cyberpunk",
                "text": "driving alone through neon streets at midnight with intense pulsing energy",
                "valence": 0.45,
                "energy": 0.85,
                "icon": "zap"
            },
            {
                "title": "Gym Beast Mode & Anger",
                "text": "furious and pumped, ready to channel all my anger into a heavy workout",
                "valence": 0.20,
                "energy": 0.95,
                "icon": "flame"
            },
            {
                "title": "Cozy Romance & Tenderness",
                "text": "cozy evening in love, looking for something sweet, warm and romantic",
                "valence": 0.86,
                "energy": 0.42,
                "icon": "heart"
            }
        ]
    }

@app.post("/api/analyze-mood")
def analyze_mood_endpoint(req: MoodRequest):
    """Parses free-text mood into emotional coordinates."""
    analysis = mood_analyzer.analyze(req.mood_text)
    if req.valence_override is not None:
        analysis["valence"] = float(round(req.valence_override, 3))
    if req.energy_override is not None:
        analysis["energy"] = float(round(req.energy_override, 3))
    return analysis

@app.post("/api/recommend")
def recommend_endpoint(req: MoodRequest):
    """End-to-end hybrid recommendation for movies and music."""
    start_time = time.time()
    
    # 1. Natural Language Mood Analysis
    mood_info = mood_analyzer.analyze(req.mood_text)
    target_valence = req.valence_override if req.valence_override is not None else mood_info["valence"]
    target_energy = req.energy_override if req.energy_override is not None else mood_info["energy"]
    dominant_emotion = mood_info["dominant_emotion"]

    # 2. Content-Based Scoring
    scored_movies_df = content_filter.score_movies(df_movies, target_valence, target_energy, dominant_emotion)
    scored_tracks_df = content_filter.score_music_tracks(df_tracks, target_valence, target_energy, dominant_emotion)

    # 3. Collaborative Filtering with in-session user projection
    movie_id_list = [int(mid) for mid in df_movies["movie_id"]]
    cf_scores = cf_model.predict_session_user(session_user_ratings, movie_id_list)

    # 4. Hybrid Re-Ranking with MMR Diversity and Novelty Modulation
    rec_movies = reranker.rerank_movies(
        df_movies=scored_movies_df,
        cf_scores=cf_scores,
        novelty_weight=req.novelty_weight,
        diversity_weight=req.diversity_weight,
        top_k=req.top_k
    )

    rec_music = reranker.rerank_music(
        df_tracks=scored_tracks_df,
        novelty_weight=req.novelty_weight,
        diversity_weight=req.diversity_weight,
        top_k=req.top_k
    )

    # 5. Explainability Layer
    for movie in rec_movies:
        movie["explanation"] = explainer.explain_movie(movie, mood_info)
        movie["user_rating"] = session_user_ratings.get(movie["movie_id"], None)

    for track in rec_music:
        track["explanation"] = explainer.explain_music(track, mood_info)
        track["liked"] = session_track_likes.get(track["track_id"], None)

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "mood_analysis": {
            "input_text": req.mood_text,
            "valence": round(target_valence, 3),
            "energy": round(target_energy, 3),
            "dominant_emotion": dominant_emotion,
            "intent": mood_info["intent"],
            "trajectory": mood_info["trajectory"],
            "explanation": mood_info["explanation"]
        },
        "recommendations": {
            "movies": rec_movies,
            "music": rec_music
        },
        "meta": {
            "latency_ms": latency_ms,
            "novelty_weight": req.novelty_weight,
            "diversity_weight": req.diversity_weight,
            "active_ratings_count": len(session_user_ratings),
            "active_track_feedback_count": len(session_track_likes)
        }
    }

@app.post("/api/feedback")
def record_feedback(fb: FeedbackRequest):
    """Records real-time user ratings and likes, dynamically adapting the CF model."""
    if fb.item_type == "movie":
        try:
            m_id = int(fb.item_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid movie ID")
        
        if fb.rating is not None:
            session_user_ratings[m_id] = float(fb.rating)
        elif fb.action == "like":
            session_user_ratings[m_id] = 5.0
        elif fb.action == "dislike":
            session_user_ratings[m_id] = 1.0

        return {
            "status": "success",
            "message": f"Recorded rating {session_user_ratings[m_id]} for movie {m_id}",
            "active_session_ratings": session_user_ratings
        }

    elif fb.item_type == "music":
        t_id = str(fb.item_id)
        if fb.action == "like":
            session_track_likes[t_id] = 1
        elif fb.action == "dislike":
            session_track_likes[t_id] = -1
        
        return {
            "status": "success",
            "message": f"Recorded feedback '{fb.action}' for track {t_id}",
            "active_track_likes": session_track_likes
        }
    
    raise HTTPException(status_code=400, detail="item_type must be 'movie' or 'music'")

@app.get("/api/session-feedback")
def get_session_feedback():
    return {
        "movies": session_user_ratings,
        "music": session_track_likes
    }

@app.post("/api/clear-feedback")
def clear_feedback():
    session_user_ratings.clear()
    session_track_likes.clear()
    return {"status": "cleared"}

@app.get("/api/metrics")
def get_offline_metrics():
    """Returns offline evaluation metrics (NDCG@K, MAP@K, ILD, Tradeoff curve)."""
    return metrics_cache

# Static files mount
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Recommender API is active. UI not found in static/."}
