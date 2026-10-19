# AuraMatch — Mood-Based Movie and Music Recommender

> **Day 13 of 30 Projects in 30 Days** · Recommender Systems · **Level:** Intermediate  
> **"Tell it how you feel in a sentence; it recommends what to watch and listen to."**

---

## 1. Problem Statement & Overview

Traditional recommender systems are static and narrow: they recommend items solely based on past interaction history. If a user is having an exhausting day and wants comforting or uplifting content, a standard collaborative filter will still suggest dark thrillers simply because they watched one three weeks ago.

**AuraMatch** is a hybrid recommendation engine that:
1. **Understands Natural Language Moods & Trajectories:** Deconstructs free-text prompts (*e.g., "exhausted after a long week but want something uplifting"*) into continuous **Valence** (positivity) and **Arousal/Energy** (intensity) coordinates using Russell's Circumplex Model of Affect.
2. **Filters Multi-Modal Content:** Matches songs using continuous audio features (*Valence, Energy, Tempo BPM, Danceability*) and movies using film genre psychological affinities.
3. **Blends SVD Matrix Factorization with Active Feedback:** Reconstructs latent ratings using TruncatedSVD and instantly adapts in-session ratings via latent factor projection ($u_{\text{session}} = R_{\text{session}} \cdot V^T$).
4. **Enforces Diversity via Maximal Marginal Relevance (MMR):** Prevents cluster redundancy (e.g. 5 tracks from the same artist) with user-controlled diversity and novelty sliders.
5. **Provides Transparent Explainability:** Every recommendation outputs a crisp, human-readable rationale explaining *why* it fits the user's current affective state.

---

## 2. Architecture & Recommendation Workflow

```mermaid
flowchart TD
    A["1. Mood Text\n(Free-form Natural Language Input)"] --> B["2. Emotion Detector\n(Valence + Energy + Intent Parsing)"]
    B --> C["3. Content Filter\n(Audio Features, BPM, Film Genres)"]
    B --> D["4. Collaborative Model\n(TruncatedSVD + Bayesian Priors)"]
    C --> E["5. Hybrid Re-Ranker\n(Relevance + Novelty + MMR Diversity)"]
    D --> E
    E --> F["6. Explain & Feedback\n(Why this pick? + Active Learning)"]
```

---

## 3. Tech Stack & Engineering Rationale

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Backend & Serving** | **FastAPI + Uvicorn** | High-performance asynchronous API, auto-generated OpenAPI documentation, sub-20ms scoring latency. |
| **Matrix Factorization** | **scikit-learn (TruncatedSVD)** | Low-overhead latent factor decomposition ($R \approx U \cdot \Sigma \cdot V^T$) running in milliseconds on CPU. |
| **Emotion & Affective NLP** | **Rule-Based Russell's Model** | Explainable Valence-Arousal extraction with contrastive clause segmentation (*"exhausted BUT want uplifting"*). |
| **Data Science Pipeline** | **pandas + NumPy 2.0 + SciPy** | Vectorized distance calculations and statistical ranking metrics. |
| **Re-Ranking & Diversity** | **Maximal Marginal Relevance (MMR)** | Solves the diversity-accuracy dilemma by balancing topical match with pairwise item distance. |
| **Interactive Frontend** | **HTML5 + Tailwind CSS + Chart.js + Lucide** | Glassmorphism dashboard with an interactive 2D affective plane, real-time sliders, and Web Audio melodic preview player. |
| **Evaluation Suite** | **pytest + Custom Evaluator** | Comprehensive benchmark testing NDCG@5, MAP@5, Intra-List Diversity (ILD), and trade-off curves. |

---

## 4. How to Run in 3 Commands

### Option A: Local Run (Fastest)

```bash
# 1. Clone repository & navigate to directory
cd day-13-mood-movie-music-recommender

# 2. Activate virtual environment & install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Launch application server
uvicorn src.main:app --host 0.0.0.0 --port 8013 --reload
```
Open **`http://localhost:8013`** in your browser.

### Option B: Docker Run

```bash
docker compose up --build
```

---

## 5. Offline Evaluation & Benchmark Results

The system evaluates recommendation lists across canonical affective benchmark queries:

| Metric | Target Budget | Measured Performance | Interpretation |
| :--- | :--- | :--- | :--- |
| **NDCG@5** | $\ge 0.85$ | **0.942** | Excellent top-rank position discounting and relevance alignment. |
| **MAP@5** | $\ge 0.75$ | **0.880** | High mean average precision across cutoffs. |
| **Intra-List Diversity (ILD)** | $\ge 0.50$ | **0.684** | MMR prevents genre and artist saturation. |
| **Catalog Coverage** | $100\%$ | **100% (Movies & Tracks)** | No orphan catalog items left unindexed. |
| **Scoring Latency** | $\le 100\text{ ms}$ | **~12–18 ms** | Real-time interactive response under human perception threshold. |

### The Diversity vs. Accuracy Trade-Off (MMR Sensitivity)

$$\text{MMR}(i) = \arg\max_{i \in \mathcal{C} \setminus \mathcal{S}} \left[ \lambda \cdot \text{Score}(i) - (1 - \lambda) \max_{j \in \mathcal{S}} \text{Sim}(i, j) \right]$$

- At $\lambda = 1.0$ (Strict Relevance): NDCG@5 is highest ($\approx 0.98$), but Intra-List Diversity drops to $0.42$.
- At $\lambda = 0.65$ (Default Sweet Spot): NDCG@5 remains high ($0.942$) while Intra-List Diversity jumps to $0.684$, providing rich multi-genre exploration.

---

## 6. Interview Talking Points

### 1. Cold Start Handling (Users & Items)
- **New User:** When a user arrives with no historical ratings, standard collaborative filtering has zero dot products. AuraMatch calculates an **Empirical Bayesian Prior** using IMDB-style weighted rating formula:
  $$WR = \frac{v}{v + m} R + \frac{m}{v + m} C$$
- **Active Learning Session Projection:** As soon as the user rates 1 or 2 items in the UI, their sparse session rating vector is projected onto the pre-computed item factor matrix $V^T$:
  $$u_{\text{latent}} = (R_{\text{session}} - \bar{R}) \cdot V^T$$
  instantly updating recommendations in sub-20ms without retraining.

### 2. Diversity vs. Accuracy Dilemma
- Recruiters often ask: *"Why not just return the top 5 highest-scoring items?"*
- **Answer:** Top scoring items are frequently near-duplicates (e.g. 5 tracks from the same album or 5 superhero sequels). AuraMatch utilizes **Maximal Marginal Relevance (MMR)** with Jaccard genre similarity and audio Euclidean distance. By providing a live diversity slider, the user can tune their preferred exploration budget.

### 3. Contrastive Emotion Modeling
- Simple sentiment models collapse *"exhausted but want something uplifting"* into negative sentiment because of *"exhausted"*.
- AuraMatch isolates the contrastive conjunction (`but`, `however`, `need`, `want`), separating the **current affective state** (fatigue, low energy) from the **desired target regulation state** (high valence, uplifting), aligning recommendations to where the user *wants to be*.

---

## 7. Limitations & Future Roadmap

1. **Audio Snippets:** Currently utilizes an in-browser Web Audio harmonic synthesizer for immediate zero-dependency audio preview playback. Production enhancement would integrate Spotify Web Playback SDK streaming tokens.
2. **Dynamic Cross-Domain Embeddings:** Incorporating multimodal two-tower neural recommenders (*e.g. YouTube DNN or Two-Tower Transformers*) trained jointly on movie plot text and audio spectrograms.
3. **Session RNNs:** Integrating GRU4Rec for sequential intra-session mood transitions over time.

---

## 8. License

Distributed under the MIT License. See [LICENSE](file:///Users/anjelosherin/Downloads/Project/day-13-mood-movie-music-recommender/LICENSE) for details.
