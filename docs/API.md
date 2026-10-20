# AuraMatch REST API Reference

The AuraMatch service exposes a clean, asynchronous REST API documented via OpenAPI / Swagger.

- **Base URL:** `http://localhost:8013`
- **Interactive Documentation:** `http://localhost:8013/docs`

---

## Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health check, catalog counts, and SVD model status. |
| `GET` | `/api/presets` | List canonical mood scenarios with predefined coordinates. |
| `POST` | `/api/analyze-mood` | Parses free-text mood into Valence, Energy, and Intent. |
| `POST` | `/api/recommend` | Main hybrid recommendation endpoint for movies and tracks. |
| `POST` | `/api/feedback` | Records star ratings, likes, or dislikes for active learning. |
| `GET` | `/api/session-feedback` | Retrieves current in-session ratings and likes. |
| `POST` | `/api/clear-feedback` | Resets current user session feedback state. |
| `GET` | `/api/metrics` | Retrieves offline benchmark scores and MMR trade-off curve. |

---

## Endpoint Details

### 1. `POST /api/recommend`
Generates ranked movie and music recommendations with transparent rationales.

#### Request Body
```json
{
  "mood_text": "exhausted after a long week but want something uplifting",
  "valence_override": null,
  "energy_override": null,
  "novelty_weight": 0.20,
  "diversity_weight": 0.35,
  "top_k": 6
}
```

#### Response (200 OK)
```json
{
  "mood_analysis": {
    "input_text": "exhausted after a long week but want something uplifting",
    "valence": 0.88,
    "energy": 0.75,
    "dominant_emotion": "Uplift",
    "intent": "mood_regulation",
    "trajectory": "From Exhaustion -> Uplift",
    "explanation": "Detected contrastive mood: currently feeling Exhaustion, with a clear desire to transition toward Uplift (Target Valence: 0.88, Energy: 0.75)."
  },
  "recommendations": {
    "movies": [
      {
        "movie_id": 8,
        "title": "Ferris Bueller's Day Off",
        "release_year": 1986,
        "genres": ["Comedy"],
        "overview": "A high school wise guy is determined to have a day off from school...",
        "director": "John Hughes",
        "vote_average": 7.8,
        "vote_count": 22000,
        "poster_url": "https://images.unsplash.com/...",
        "content_score": 0.95,
        "cf_rating": 3.8,
        "novelty_score": 0.18,
        "relevance_score": 0.5766,
        "explanation": "Guides your transition toward uplift with vibrant comedy energy (90% positivity) and an uplifting resolution."
      }
    ],
    "music": [
      {
        "track_id": "track_07",
        "track_name": "Happy",
        "artists": "Pharrell Williams",
        "album_name": "G I R L",
        "track_genre": "pop",
        "valence": 0.96,
        "energy": 0.82,
        "tempo": 160.0,
        "danceability": 0.65,
        "popularity": 83,
        "explanation": "Picks you up with radiant 96% harmonic valence and an infectious 160 BPM groove."
      }
    ]
  },
  "meta": {
    "latency_ms": 12.4,
    "novelty_weight": 0.2,
    "diversity_weight": 0.35,
    "active_ratings_count": 0,
    "active_track_feedback_count": 0
  }
}
```

---

### 2. `POST /api/feedback`
Records in-session ratings to dynamically personalize subsequent collaborative filtering queries.

#### Request Body (Movie Star Rating)
```json
{
  "item_type": "movie",
  "item_id": "1",
  "rating": 5.0
}
```

#### Request Body (Track Like)
```json
{
  "item_type": "music",
  "item_id": "track_01",
  "action": "like"
}
```

#### Response (200 OK)
```json
{
  "status": "success",
  "message": "Recorded rating 5.0 for movie 1",
  "active_session_ratings": {
    "1": 5.0
  }
}
```

---

### 3. `GET /api/metrics`
Returns offline evaluation scores, catalog coverage, and MMR sensitivity curve.

#### Response (200 OK)
```json
{
  "mean_ndcg_at_5": 0.916,
  "mean_map_at_5": 0.968,
  "mean_movie_diversity": 0.836,
  "mean_music_diversity": 0.107,
  "mean_novelty": 0.342,
  "catalog_coverage_movies": "32 / 32 (100%)",
  "catalog_coverage_music": "32 / 32 (100%)",
  "diversity_vs_accuracy_tradeoff": [
    { "diversity_weight": 0.0, "ndcg_at_5": 1.0, "intra_list_diversity": 0.708 },
    { "diversity_weight": 0.2, "ndcg_at_5": 0.927, "intra_list_diversity": 0.893 },
    { "diversity_weight": 0.4, "ndcg_at_5": 0.916, "intra_list_diversity": 0.908 },
    { "diversity_weight": 0.6, "ndcg_at_5": 0.893, "intra_list_diversity": 0.880 },
    { "diversity_weight": 0.8, "ndcg_at_5": 0.860, "intra_list_diversity": 0.894 },
    { "diversity_weight": 1.0, "ndcg_at_5": 0.860, "intra_list_diversity": 0.894 }
  ]
}
```
