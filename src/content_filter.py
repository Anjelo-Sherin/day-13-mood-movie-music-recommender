"""Content-Based Filtering for Music Audio Features and Film Genre/Mood Alignments."""
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Mood to Music Genre affinities
MUSIC_GENRE_AFFINITIES = {
    "Joy": ["pop", "dance", "classic rock", "rock", "soul"],
    "Excitement": ["edm", "dance", "synthwave", "hip-hop", "rock"],
    "Uplift": ["pop", "rock", "classic rock", "dance", "soul"],
    "Calm": ["ambient", "acoustic", "classical", "chillwave", "indie folk", "soul"],
    "Romance": ["r-n-b", "soul", "acoustic", "pop", "indie folk"],
    "Sadness": ["indie folk", "indie", "ambient", "classical", "art rock", "pop"],
    "Melancholy": ["indie", "art rock", "classical", "indie folk", "ambient"],
    "Exhaustion": ["ambient", "classical", "chillwave", "acoustic"],
    "Anger": ["metal", "alternative metal", "hip-hop", "breakbeat", "rock"],
    "Stress": ["ambient", "classical", "chillwave", "acoustic", "indie folk"],
    "Anxiety": ["ambient", "classical", "chillwave"],
    "Dark/Intense": ["synth-pop", "breakbeat", "metal", "synthwave", "hip-hop"]
}

# Mood to Movie Genre weights
MOVIE_GENRE_AFFINITIES = {
    "Joy": {"Animation": 0.9, "Comedy": 1.0, "Adventure": 0.8, "Musical": 0.9, "Drama": 0.3, "Crime": 0.1, "Horror": 0.0},
    "Excitement": {"Action": 1.0, "Adventure": 0.9, "Sci-Fi": 0.8, "Comedy": 0.7, "Drama": 0.3, "Romance": 0.2},
    "Uplift": {"Comedy": 0.9, "Animation": 0.8, "Drama": 0.7, "Family": 0.9, "Musical": 0.8, "Crime": 0.1},
    "Calm": {"Family": 0.9, "Romance": 0.8, "Fantasy": 0.7, "Animation": 0.7, "Comedy": 0.6, "Action": 0.0, "Horror": 0.0},
    "Romance": {"Romance": 1.0, "Drama": 0.8, "Comedy": 0.7, "Fantasy": 0.6},
    "Sadness": {"Drama": 1.0, "Romance": 0.6, "Mystery": 0.5, "Action": 0.1, "Comedy": 0.0},
    "Melancholy": {"Drama": 1.0, "Romance": 0.7, "Sci-Fi": 0.6, "Mystery": 0.6, "Comedy": 0.1},
    "Exhaustion": {"Comedy": 0.7, "Family": 0.8, "Animation": 0.8, "Drama": 0.4},
    "Anger": {"Action": 1.0, "Crime": 0.9, "Thriller": 0.9, "Drama": 0.7, "Comedy": 0.1},
    "Stress": {"Comedy": 0.9, "Family": 0.8, "Animation": 0.8, "Fantasy": 0.7},
    "Anxiety": {"Family": 0.8, "Animation": 0.8, "Comedy": 0.7, "Thriller": 0.0},
    "Dark/Intense": {"Thriller": 1.0, "Crime": 0.9, "Sci-Fi": 0.8, "Drama": 0.8, "Action": 0.8, "Comedy": 0.0}
}

class ContentFilter:
    """Computes content-based mood alignment scores for tracks and movies."""

    def score_music_tracks(
        self,
        df_tracks: pd.DataFrame,
        target_valence: float,
        target_energy: float,
        dominant_emotion: str
    ) -> pd.DataFrame:
        """
        Scores music tracks based on Euclidean distance in normalized (Valence, Energy, Tempo)
        plus genre affinity bonus.
        """
        scored_df = df_tracks.copy()
        preferred_genres = MUSIC_GENRE_AFFINITIES.get(dominant_emotion, ["pop", "rock", "indie"])

        # Target BPM heuristic: high energy targets faster BPM
        # 0.0 energy -> 60 BPM, 1.0 energy -> 160 BPM
        target_bpm = 60.0 + (target_energy * 100.0)

        scores = []
        for _, row in scored_df.iterrows():
            v = float(row["valence"])
            e = float(row["energy"])
            tempo = float(row.get("tempo", 110.0))
            genre = str(row.get("track_genre", "")).lower()

            # Normalized tempo distance (clamped to 60-180 range)
            norm_tempo = (np.clip(tempo, 60.0, 180.0) - 60.0) / 120.0
            norm_target_bpm = (target_bpm - 60.0) / 120.0

            # Feature distance calculation
            v_dist = abs(v - target_valence)
            e_dist = abs(e - target_energy)
            tempo_dist = abs(norm_tempo - norm_target_bpm)

            # Audio fitness (1.0 = perfect match, 0.0 = opposite spectrum)
            audio_dist = np.sqrt(0.45 * (v_dist ** 2) + 0.40 * (e_dist ** 2) + 0.15 * (tempo_dist ** 2))
            audio_fitness = max(0.0, 1.0 - audio_dist)

            # Genre affinity multiplier
            genre_boost = 1.15 if any(g in genre for g in preferred_genres) else 1.0
            final_audio_score = float(np.clip(audio_fitness * genre_boost, 0.0, 1.0))

            scores.append(final_audio_score)

        scored_df["content_score"] = scores
        return scored_df

    def score_movies(
        self,
        df_movies: pd.DataFrame,
        target_valence: float,
        target_energy: float,
        dominant_emotion: str
    ) -> pd.DataFrame:
        """
        Scores movies based on intrinsic emotional valence/energy proximity
        and genre-psychology affinity matrix.
        """
        scored_df = df_movies.copy()
        genre_weights = MOVIE_GENRE_AFFINITIES.get(dominant_emotion, {})

        scores = []
        for _, row in scored_df.iterrows():
            mv = float(row["mood_valence"])
            me = float(row["mood_energy"])
            genres = str(row["genres"]).split("|")

            # 2D Affective Euclidean distance in valence-energy space
            dist = np.sqrt(((mv - target_valence) ** 2) + ((me - target_energy) ** 2))
            affective_fitness = max(0.0, 1.0 - (dist / np.sqrt(2)))

            # Genre affinity matching
            genre_scores = [genre_weights.get(g, 0.5) for g in genres]
            avg_genre_score = float(np.mean(genre_scores)) if genre_scores else 0.5

            # Combined content score (70% affective fitness + 30% genre match)
            final_score = float(0.70 * affective_fitness + 0.30 * avg_genre_score)
            scores.append(final_score)

        scored_df["content_score"] = scores
        return scored_df
