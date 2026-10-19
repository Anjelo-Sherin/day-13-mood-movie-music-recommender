"""Hybrid Re-Ranker balancing Mood Relevance, Collaborative Matrix Factorization, Novelty, and MMR Diversity."""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Set

class HybridReranker:
    """Balances Content-Based Mood Fit, SVD Collaborative Filtering, Novelty, and MMR Diversity."""

    def rerank_movies(
        self,
        df_movies: pd.DataFrame,
        cf_scores: Dict[int, float],
        novelty_weight: float = 0.20,
        diversity_weight: float = 0.35,
        top_k: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Merges content mood scores and SVD collaborative predicted ratings,
        applies novelty scaling, and re-ranks using Maximal Marginal Relevance (MMR).
        """
        candidates = []
        for _, row in df_movies.iterrows():
            m_id = int(row["movie_id"])
            content_sc = float(row.get("content_score", 0.5))
            predicted_r = float(cf_scores.get(m_id, 3.5))
            norm_cf = predicted_r / 5.0

            # Base hybrid score: 55% mood alignment + 45% collaborative rating
            base_score = 0.55 * content_sc + 0.45 * norm_cf

            # Novelty: inverse of vote count/popularity
            vote_count = float(row.get("vote_count", 20000))
            # Log-scaled novelty where lower vote counts have higher novelty
            novelty_score = 1.0 - (np.log1p(vote_count) / np.log1p(100000))
            novelty_score = float(np.clip(novelty_score, 0.05, 0.95))

            # Blend novelty according to user slider
            blended_score = (1.0 - novelty_weight) * base_score + (novelty_weight * novelty_score)

            candidates.append({
                "movie_id": m_id,
                "title": row["title"],
                "release_year": int(row["release_year"]),
                "genres": row["genres"].split("|"),
                "overview": row["overview"],
                "director": row["director"],
                "vote_average": float(row["vote_average"]),
                "vote_count": int(vote_count),
                "poster_url": row["poster_url"],
                "mood_valence": float(row["mood_valence"]),
                "mood_energy": float(row["mood_energy"]),
                "mood_tags": row["mood_tags"].split("|"),
                "content_score": round(content_sc, 3),
                "cf_rating": round(predicted_r, 2),
                "novelty_score": round(novelty_score, 3),
                "relevance_score": round(blended_score, 4)
            })

        # Apply Maximal Marginal Relevance (MMR) for genre & director diversity
        return self._apply_mmr(
            candidates=candidates,
            sim_func=self._movie_similarity,
            diversity_lambda=1.0 - diversity_weight,
            top_k=top_k
        )

    def rerank_music(
        self,
        df_tracks: pd.DataFrame,
        novelty_weight: float = 0.20,
        diversity_weight: float = 0.35,
        top_k: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Re-ranks Spotify music tracks using Audio Fitness, Popularity/Novelty tradeoff,
        and MMR for artist and genre diversity.
        """
        candidates = []
        for _, row in df_tracks.iterrows():
            content_sc = float(row.get("content_score", 0.5))
            pop = float(row.get("popularity", 80))
            norm_pop = pop / 100.0

            # Base score: 85% audio fitness + 15% popularity
            base_score = 0.85 * content_sc + 0.15 * norm_pop

            # Novelty: inverse of popularity
            novelty_score = 1.0 - norm_pop
            blended_score = (1.0 - novelty_weight) * base_score + (novelty_weight * novelty_score)

            candidates.append({
                "track_id": row["track_id"],
                "track_name": row["track_name"],
                "artists": row["artists"],
                "album_name": row["album_name"],
                "track_genre": row["track_genre"],
                "valence": float(row["valence"]),
                "energy": float(row["energy"]),
                "tempo": float(row["tempo"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
                "popularity": int(pop),
                "duration_ms": int(row["duration_ms"]),
                "cover_art": row["cover_art"],
                "mood_tags": str(row.get("mood_tags", "")).split("|"),
                "content_score": round(content_sc, 3),
                "novelty_score": round(novelty_score, 3),
                "relevance_score": round(blended_score, 4)
            })

        # Apply MMR for artist and genre diversity
        return self._apply_mmr(
            candidates=candidates,
            sim_func=self._music_similarity,
            diversity_lambda=1.0 - diversity_weight,
            top_k=top_k
        )

    def _apply_mmr(
        self,
        candidates: List[Dict[str, Any]],
        sim_func,
        diversity_lambda: float,
        top_k: int
    ) -> List[Dict[str, Any]]:
        r"""
        Maximal Marginal Relevance (MMR) greedy selection:
        MMR = argmax_{d in C \ S} [ lambda * Rel(d) - (1 - lambda) * max_{s in S} Sim(d, s) ]
        """
        if not candidates:
            return []

        remaining = candidates.copy()
        # Sort remaining by initial relevance
        remaining.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        selected: List[Dict[str, Any]] = []
        
        # Pick the absolute best item first
        selected.append(remaining.pop(0))

        while len(selected) < top_k and remaining:
            best_mmr_score = -float("inf")
            best_idx = 0

            for i, cand in enumerate(remaining):
                rel = cand["relevance_score"]
                # Maximum similarity to any already-selected item
                max_sim = max(sim_func(cand, sel) for sel in selected)
                mmr_val = (diversity_lambda * rel) - ((1.0 - diversity_lambda) * max_sim)

                if mmr_val > best_mmr_score:
                    best_mmr_score = mmr_val
                    best_idx = i

            picked = remaining.pop(best_idx)
            selected.append(picked)

        return selected

    @staticmethod
    def _movie_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        """Calculates Jaccard genre similarity and director overlap between two movies."""
        genres_a = set(a.get("genres", []))
        genres_b = set(b.get("genres", []))
        jaccard = len(genres_a & genres_b) / max(1, len(genres_a | genres_b))
        same_director = 0.3 if a.get("director") == b.get("director") else 0.0
        return float(min(1.0, 0.7 * jaccard + same_director))

    @staticmethod
    def _music_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        """Calculates artist and audio similarity between two tracks."""
        same_artist = 0.5 if a.get("artists") == b.get("artists") else 0.0
        same_genre = 0.3 if a.get("track_genre") == b.get("track_genre") else 0.0
        # Audio feature delta
        v_diff = abs(a.get("valence", 0.5) - b.get("valence", 0.5))
        e_diff = abs(a.get("energy", 0.5) - b.get("energy", 0.5))
        audio_sim = max(0.0, 1.0 - np.sqrt(v_diff**2 + e_diff**2))
        return float(min(1.0, same_artist + same_genre + (0.2 * audio_sim)))
