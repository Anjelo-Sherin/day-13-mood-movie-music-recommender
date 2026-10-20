# Local Setup and Installation Guide

This guide walks you through setting up, configuring, running, and testing **AuraMatch** on your local machine.

---

## 1. Prerequisites

- **Python:** Version `3.9` through `3.12` installed.
- **Git:** Version `2.30+` installed.
- **Docker & Docker Compose (Optional):** If you prefer running containerized services.
- **RAM:** Minimum 512 MB required (lightweight, runs completely on CPU).

---

## 2. Quickstart Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/Anjelo-Sherin/day-13-mood-movie-music-recommender.git
cd day-13-mood-movie-music-recommender
```

### Step 2: Create a Virtual Environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Environment Variables (Optional)
The application works 100% offline out-of-the-box using the included MovieLens and Spotify datasets. If you wish to configure custom ports or external API keys:
```bash
cp .env.example .env
```
Key configuration fields:
```ini
PORT=8013
HOST=0.0.0.0
DEBUG=True
```

---

## 3. Running the Application

### Option A: Direct Uvicorn Launch
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8013 --reload
```
Once launched, navigate to:
- **Web Dashboard:** [http://localhost:8013](http://localhost:8013)
- **Interactive Swagger Docs:** [http://localhost:8013/docs](http://localhost:8013/docs)
- **OpenAPI JSON Spec:** [http://localhost:8013/openapi.json](http://localhost:8013/openapi.json)

### Option B: Docker Container
```bash
# Build and run with docker compose
docker compose up --build

# Run in background
docker compose up -d

# Stop container
docker compose down
```

---

## 4. Running the Test Suite

AuraMatch includes both automated pytest unit tests and an end-to-end benchmark validation script.

### Run Automated Unit & API Tests
```bash
pytest tests/ -v
```

### Run Full Multi-Scenario E2E Validation
```bash
PYTHONPATH=. python3 tests/e2e_test_runner.py
```
This script exercises:
1. Health and dataset sanity check.
2. Canonical preset retrieval.
3. 4 natural language mood queries (Contrastive, Melancholic, Adrenaline, Calm).
4. Active learning session rating persistence.
5. Quantitative benchmark metrics (NDCG@5, MAP@5, ILD).

---

## 5. Troubleshooting & FAQ

#### `Address already in use` error
If port 8013 is occupied by another process:
```bash
# Find and terminate process on macOS/Linux
lsof -ti :8013 | xargs kill -9
```
Or start on a different port:
```bash
uvicorn src.main:app --port 8080 --reload
```

#### Audio playback is silent
AuraMatch utilizes the Web Audio API to synthesize harmonious chord previews directly inside modern browsers (Chrome, Firefox, Safari, Edge). If you hear no sound, click anywhere on the page to un-mute the browser's audio context.
