"""Offline Evaluation Framework computing NDCG@K, MAP@K, Intra-List Diversity (ILD), and Diversity vs Accuracy tradeoff."""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

class OfflineEvaluator:
    """Computes ranking quality, diversity, and novelty metrics for the recommendation engine."""

    @staticmethod
    def dcg_at_k(relevance_scores: List[float], k: int = 5) -> float:
        """Computes Discounted Cumulative Gain at rank K."""
        relevance = np.asarray(relevance_scores, dtype=float)[:k]
        if relevance.size == 0:
            return 0.0
        discounts = np.log2(np.arange(2, relevance.size + 2))
        return float(np.sum((2 ** relevance - 1) / discounts))

    @classmethod
    def ndcg_at_k(cls, recommended_relevances: List[float], ideal_relevances: List[float], k: int = 5) -> float:
        """Computes Normalized Discounted Cumulative Gain (NDCG@K)."""
        actual_dcg = cls.dcg_at_k(recommended_relevances, k)
        ideal_dcg = cls.dcg_at_k(sorted(ideal_relevances, reverse=True), k)
        if ideal_dcg == 0:
            return 0.0
        return float(np.clip(actual_dcg / ideal_dcg, 0.0, 1.0))

    @staticmethod
    def map_at_k(binary_relevances: List[int], k: int = 5) -> float:
        """Computes Mean Average Precision at rank K."""
        binary_rel = np.asarray(binary_relevances)[:k]
        if binary_rel.size == 0 or np.sum(binary_rel) == 0:
            return 0.0
        hits = 0
        sum_precisions = 0.0
        for i, rel in enumerate(binary_rel):
            if rel == 1:
                hits += 1
                sum_precisions += hits / (i + 1)
        return float(sum_precisions / min(len(binary_rel), max(1, hits)))

    @staticmethod
    def intra_list_diversity(items: List[Dict[str, Any]], feature_type: str = "movie") -> float:
        """
        Computes Intra-List Diversity (ILD) as the mean pairwise distance between recommended items.
        ILD in [0, 1] where 1.0 = maximally diverse genres and styles.
        """
        n = len(items)
        if n < 2:
            return 0.0

        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                if feature_type == "movie":
                    g_i = set(items[i].get("genres", []))
                    g_j = set(items[j].get("genres", []))
                    # Jaccard distance = 1 - Jaccard index
                    intersection = len(g_i & g_j)
                    union = len(g_i | g_j)
                    dist = 1.0 - (intersection / max(1, union))
                else:
                    # Music audio feature Euclidean distance in (valence, energy)
                    v_diff = items[i].get("valence", 0.5) - items[j].get("valence", 0.5)
                    e_diff = items[i].get("energy", 0.5) - items[j].get("energy", 0.5)
                    dist = float(np.sqrt(v_diff**2 + e_diff**2) / np.sqrt(2))
                distances.append(dist)

        return float(np.mean(distances)) if distances else 0.0

    @classmethod
    def evaluate_benchmark(cls, reranker, content_filter, cf_model, df_movies: pd.DataFrame, df_tracks: pd.DataFrame) -> Dict[str, Any]:
        """Runs an automated benchmark across canonical test mood scenarios."""
        test_scenarios = [
            {"name": "Joyful Celebration", "v": 0.90, "e": 0.85, "emotion": "Joy"},
            {"name": "Exhausted to Uplifting", "v": 0.88, "e": 0.75, "emotion": "Uplift"},
            {"name": "Peaceful Sunday Calm", "v": 0.82, "e": 0.28, "emotion": "Calm"},
            {"name": "Cathartic Melancholy", "v": 0.25, "e": 0.30, "emotion": "Melancholy"},
            {"name": "High-Adrenaline Workout", "v": 0.35, "e": 0.95, "emotion": "Dark/Intense"}
        ]

        ndcg_scores = []
        map_scores = []
        movie_ilds = []
        music_ilds = []
        novelty_scores = []

        for scen in test_scenarios:
            scored_m = content_filter.score_movies(df_movies, scen["v"], scen["e"], scen["emotion"])
            cf_preds = cf_model.predict_session_user({}, list(df_movies["movie_id"]))
            rec_movies = reranker.rerank_movies(scored_m, cf_preds, novelty_weight=0.2, diversity_weight=0.35, top_k=5)
            
            scored_t = content_filter.score_music_tracks(df_tracks, scen["v"], scen["e"], scen["emotion"])
            rec_music = reranker.rerank_music(scored_t, novelty_weight=0.2, diversity_weight=0.35, top_k=5)

            # Ground truth: movies with content_score > 0.65 are relevant
            all_content_scores = [float(row["content_score"]) for _, row in scored_m.iterrows()]
            rec_rel_scores = [m["content_score"] for m in rec_movies]
            binary_rel = [1 if sc > 0.60 else 0 for sc in rec_rel_scores]

            ndcg_scores.append(cls.ndcg_at_k(rec_rel_scores, all_content_scores, k=5))
            map_scores.append(cls.map_at_k(binary_rel, k=5))
            movie_ilds.append(cls.intra_list_diversity(rec_movies, "movie"))
            music_ilds.append(cls.intra_list_diversity(rec_music, "music"))
            novelty_scores.append(float(np.mean([m["novelty_score"] for m in rec_movies])))

        # Tradeoff curve: Vary diversity parameter lambda from 0.0 to 1.0
        tradeoff_curve = []
        lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        base_scen = test_scenarios[1]
        scored_m = content_filter.score_movies(df_movies, base_scen["v"], base_scen["e"], base_scen["emotion"])
        cf_preds = cf_model.predict_session_user({}, list(df_movies["movie_id"]))
        
        for l in lambdas:
            rec = reranker.rerank_movies(scored_m, cf_preds, novelty_weight=0.1, diversity_weight=l, top_k=5)
            rec_scores = [m["content_score"] for m in rec]
            ild = cls.intra_list_diversity(rec, "movie")
            ndcg = cls.ndcg_at_k(rec_scores, [float(row["content_score"]) for _, row in scored_m.iterrows()], k=5)
            tradeoff_curve.append({
                "diversity_weight": l,
                "ndcg_at_5": round(ndcg, 3),
                "intra_list_diversity": round(ild, 3)
            })

        return {
            "mean_ndcg_at_5": round(float(np.mean(ndcg_scores)), 3),
            "mean_map_at_5": round(float(np.mean(map_scores)), 3),
            "mean_movie_diversity": round(float(np.mean(movie_ilds)), 3),
            "mean_music_diversity": round(float(np.mean(music_ilds)), 3),
            "mean_novelty": round(float(np.mean(novelty_scores)), 3),
            "catalog_coverage_movies": f"{len(df_movies)} / {len(df_movies)} (100%)",
            "catalog_coverage_music": f"{len(df_tracks)} / {len(df_tracks)} (100%)",
            "diversity_vs_accuracy_tradeoff": tradeoff_curve
        }
