"""Collaborative Filtering Engine using Matrix Factorization (TruncatedSVD) with Cold-Start Handling."""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from typing import Dict, List, Tuple, Optional

class CollaborativeFilteringModel:
    """Matrix Factorization model using TruncatedSVD on User-Item rating matrices."""

    def __init__(self, n_factors: int = 10, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
        self.user_item_matrix = None
        self.user_ids = []
        self.movie_ids = []
        self.user_id_to_idx = {}
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.U = None
        self.Vt = None
        self.global_mean = 3.5
        self.movie_stats = {}  # movieId -> {mean_rating, count, bayesian_score}
        self.is_fitted = False

    def fit(self, df_ratings: pd.DataFrame, df_movies: pd.DataFrame):
        """Fits the TruncatedSVD model on the rating matrix and computes Bayesian cold-start priors."""
        self.global_mean = float(df_ratings["rating"].mean())
        
        # Calculate Bayesian weighted scores for cold start
        # WR = (v / (v + m)) * R + (m / (v + m)) * C
        m_threshold = 10  # minimum vote count prior
        grouped = df_ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
        for _, row in grouped.iterrows():
            m_id = int(row["movieId"])
            r_mean = float(row["mean"])
            r_count = float(row["count"])
            bayesian = (r_count / (r_count + m_threshold)) * r_mean + (m_threshold / (r_count + m_threshold)) * self.global_mean
            self.movie_stats[m_id] = {
                "mean_rating": r_mean,
                "count": r_count,
                "bayesian_score": bayesian
            }

        # Also populate any movie in df_movies that has no ratings yet
        for _, m_row in df_movies.iterrows():
            m_id = int(m_row["movie_id"])
            if m_id not in self.movie_stats:
                # Use vote_average if available scaled to 5-star
                vote_avg_5 = float(m_row.get("vote_average", 7.0)) / 2.0
                self.movie_stats[m_id] = {
                    "mean_rating": vote_avg_5,
                    "count": 5,
                    "bayesian_score": (5 / (5 + m_threshold)) * vote_avg_5 + (m_threshold / (5 + m_threshold)) * self.global_mean
                }

        # Pivot to User-Item Matrix
        pivot = df_ratings.pivot(index="userId", columns="movieId", values="rating")
        self.user_ids = list(pivot.index)
        self.movie_ids = list(pivot.columns)
        self.user_id_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.idx_to_movie_id = {i: mid for mid, i in self.movie_id_to_idx.items()}

        # Mean-center impute missing values for SVD stability
        matrix_imputed = pivot.fillna(self.global_mean).values

        # Adjust factor components if smaller than matrix dimensions
        actual_factors = min(self.n_factors, min(matrix_imputed.shape) - 1)
        self.svd = TruncatedSVD(n_components=actual_factors, random_state=self.random_state)
        
        # Decompose R ~ U * Sigma * Vt
        user_factors = self.svd.fit_transform(matrix_imputed)
        self.U = user_factors
        self.Vt = self.svd.components_
        self.is_fitted = True
        return self

    def predict_user_movie(self, user_id: int, movie_id: int) -> float:
        """Predicts rating for a given user and movie."""
        if not self.is_fitted:
            return self.global_mean

        if user_id in self.user_id_to_idx and movie_id in self.movie_id_to_idx:
            u_idx = self.user_id_to_idx[user_id]
            m_idx = self.movie_id_to_idx[movie_id]
            # Dot product of user factor and item factor
            pred = np.dot(self.U[u_idx], self.Vt[:, m_idx])
            return float(np.clip(pred, 1.0, 5.0))
        
        # Cold start fallback to Bayesian weighted prior
        if movie_id in self.movie_stats:
            return float(self.movie_stats[movie_id]["bayesian_score"])
        return self.global_mean

    def predict_session_user(self, session_ratings: Dict[int, float], movie_ids: List[int]) -> Dict[int, float]:
        """
        Active learning & Session personalizer:
        Projects in-session user ratings into latent space to generate predicted ratings for all candidate movies.
        If session_ratings is empty (cold start), returns Bayesian popularity priors.
        """
        if not self.is_fitted:
            return {mid: self.global_mean for mid in movie_ids}

        if not session_ratings:
            # Cold-start: return normalized Bayesian priors
            scores = {}
            for mid in movie_ids:
                if mid in self.movie_stats:
                    scores[mid] = self.movie_stats[mid]["bayesian_score"]
                else:
                    scores[mid] = self.global_mean
            return scores

        # Construct sparse session user vector in the training movie space
        session_vec = np.full(len(self.movie_ids), self.global_mean)
        has_overlap = False
        for mid, r in session_ratings.items():
            if mid in self.movie_id_to_idx:
                session_vec[self.movie_id_to_idx[mid]] = r
                has_overlap = True

        if not has_overlap:
            return {mid: self.movie_stats.get(mid, {}).get("bayesian_score", self.global_mean) for mid in movie_ids}

        # Project session vector into latent factor space
        # Center around global mean for latent offset
        centered_vec = session_vec - self.global_mean
        u_latent = np.dot(centered_vec, self.Vt.T)
        reconstructed_offset = np.dot(u_latent, self.Vt)
        reconstructed = self.global_mean + reconstructed_offset
        
        scores = {}
        for mid in movie_ids:
            if mid in session_ratings:
                # Explicit in-session feedback takes precedence
                scores[mid] = float(session_ratings[mid])
            elif mid in self.movie_id_to_idx:
                idx = self.movie_id_to_idx[mid]
                # Blend reconstructed latent score with Bayesian prior
                rec_val = float(np.clip(reconstructed[idx], 1.0, 5.0))
                prior = self.movie_stats.get(mid, {}).get("bayesian_score", self.global_mean)
                scores[mid] = round(0.70 * rec_val + 0.30 * prior, 2)
            else:
                scores[mid] = self.movie_stats.get(mid, {}).get("bayesian_score", self.global_mean)
        return scores
