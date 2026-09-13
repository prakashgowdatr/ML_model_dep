# PM2.5 Forecaster — LSTM Air-Quality Prediction API

[![CI](https://github.com/prakashgowdatr/ML_model_dep/actions/workflows/ci.yml/badge.svg)](https://github.com/prakashgowdatr/ML_model_dep/actions/workflows/ci.yml)
[![CD](https://github.com/prakashgowdatr/ML_model_dep/actions/workflows/cd.yml/badge.svg)](https://github.com/prakashgowdatr/ML_model_dep/actions/workflows/cd.yml)

> **Learning objective:** CI/CD with GitHub Actions.  
> The ML model is intentionally simple. The engineering pipeline is the point.

---

## Project Overview

This project predicts the **next hour's PM2.5 concentration** (air pollution) using the previous 24 hours of weather measurements. A single-layer LSTM is wrapped in a FastAPI REST API, containerised with Docker, and deployed through a fully automated GitHub Actions CI/CD pipeline.

```
Push code → GitHub Actions → Lint → Test → Docker Build → Push to GHCR → Deploy
```

---

## Architecture

```
data/raw/PRSA_data.csv
        │
        ▼
src/data/preprocessing.py      ← clean, split, scale, build windows
        │
        ▼
src/models/lstm.py              ← LSTMForecaster(hidden=64, layers=1)
        │
        ▼
src/training/train.py           ← train, early stop, checkpoint
        │
        ▼
artifacts/model.pt              ← saved weights
artifacts/scaler.pkl            ← saved MinMaxScaler
        │
        ▼
api/main.py (FastAPI)           ← POST /predict
        │
        ▼
Dockerfile                      ← containerise the API
        │
        ▼
.github/workflows/ci.yml        ← lint + test + docker build
.github/workflows/cd.yml        ← push image to GHCR
```

---

## Why LSTM for Time Series?

PM2.5 follows sequential patterns — yesterday's pollution predicts today's. LSTMs maintain a **hidden state** that "remembers" context across the 24-hour window, unlike a plain linear model which would treat each hour independently.

A single-layer LSTM is sufficient here because:
- The dataset is clean and structured
- We're predicting just one step ahead
- Keeping it simple makes the engineering pipeline the focus

---

## Dataset

**UCI Beijing PM2.5 Data** — hourly measurements from 2010–2014.

| Column | Description |
|--------|-------------|
| `pm25` | PM2.5 concentration (µg/m³) |
| `temperature` | Temperature (°C) |
| `humidity` | Dew point as humidity proxy |
| `wind_speed` | Wind speed (m/s) |
| `pressure` | Atmospheric pressure (hPa) |

Rows: ~43,000 · Split: 70% train / 15% val / 15% test (chronological)

---

## Project Structure

```
.
├── .github/
│   └── workflows/
│       ├── ci.yml          ← Lint → Test → Docker build
│       └── cd.yml          ← Build → Tag → Push to GHCR
├── api/
│   └── main.py             ← FastAPI app
├── data/raw/               ← Raw CSV goes here (gitignored)
├── artifacts/              ← model.pt + scaler.pkl (gitignored)
├── src/
│   ├── config.py           ← Loads config.yaml
│   ├── data/preprocessing.py
│   ├── models/lstm.py
│   ├── training/train.py
│   ├── evaluation/evaluate.py
│   └── inference/predict.py
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
├── scripts/
│   └── download_data.py
├── config.yaml             ← All hyperparameters live here
├── requirements.txt
├── Dockerfile
└── LEARNING_GUIDE.md
```

---

## Local Setup

```bash
# 1. Clone
git clone https://github.com/prakashgowdatr/ML_model_dep.git
cd ML_model_dep

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Step-by-Step Usage

### 1. Download Data
```bash
python scripts/download_data.py
# → data/raw/PRSA_data.csv
```

### 2. Train the Model
```bash
python src/training/train.py
# → artifacts/model.pt
# → artifacts/scaler.pkl
# → artifacts/metrics.json
```

### 3. Evaluate
```bash
python src/evaluation/evaluate.py
# → prints MAE / RMSE / R² on the test set
```

### 4. Run the API Locally
```bash
uvicorn api.main:app --reload
# → http://localhost:8000/docs  (Swagger UI)
```

### 5. Run Tests
```bash
pytest tests/ -v
```

### 6. Build & Run Docker
```bash
# Build
docker build -t pm25-forecaster .

# Run
docker run -p 8000:8000 pm25-forecaster

# Test
curl http://localhost:8000/health
```

---

## API Reference

### `GET /health`
```json
{"status": "ok"}
```

### `GET /model-info`
Returns model config and test metrics.

### `POST /predict`

**Request:**
```json
{
  "observations": [
    {"pm25": 120, "temperature": 27.5, "humidity": 70, "wind_speed": 2.1, "pressure": 1008},
    ... (24 entries total, one per hour)
  ]
}
```

**Response:**
```json
{"predicted_pm25": 115.4}
```

**Error (wrong count):**
```json
{"detail": [{"msg": "Exactly 24 observations are required, got 5."}]}
```

---

## CI Pipeline Explained

```
Push to main / Open PR
        ↓
┌─────────────┐
│    lint     │  Ruff checks formatting + code style (~10s)
└──────┬──────┘
       │ passes
       ↓
┌─────────────┐
│    test     │  pytest runs all unit tests (~30s)
└──────┬──────┘
       │ passes
       ↓
┌──────────────────┐
│  docker-build    │  Builds image + smoke tests /health (~2min)
└──────────────────┘
```

Every step must pass. If lint fails, tests don't run. If tests fail, Docker doesn't build. **This protects `main` from broken code.**

---

## CD Pipeline Explained

CD runs only on push to `main` (after CI passes):

```
1. Checkout repository
2. Log in to GHCR using GITHUB_TOKEN (auto-provided, no setup)
3. Build Docker image
4. Tag it:
   • :latest          → always the newest
   • :sha-a1b2c3d     → pinned to this exact commit
   • :main            → branch name
5. Push all tags to ghcr.io/prakashgowdatr/ml_model_dep
```

Your image is now at:
```
ghcr.io/prakashgowdatr/ml_model_dep:latest
```

---

## How GHCR Works

**GitHub Container Registry (GHCR)** is a Docker image registry, similar to Docker Hub, but built into GitHub.

- Free for public repositories
- Authentication uses your existing `GITHUB_TOKEN`
- Images are linked to your repository automatically
- Pull with: `docker pull ghcr.io/prakashgowdatr/ml_model_dep:latest`

---

## Deploying to Production

After CD pushes the image, deploy it anywhere:

**Option A — Cloud VM (EC2, GCP Compute, DigitalOcean):**
```bash
ssh user@your-server
docker pull ghcr.io/prakashgowdatr/ml_model_dep:latest
docker stop pm25 && docker rm pm25
docker run -d --name pm25 -p 80:8000 ghcr.io/prakashgowdatr/ml_model_dep:latest
```

**Option B — Railway / Render:**  
Enter your GHCR image URL in their dashboard — they pull and deploy automatically.

---

## Changing Hyperparameters

Edit `config.yaml` — no code changes needed:

```yaml
model:
  hidden_size: 128   # was 64

training:
  epochs: 100        # was 50
  learning_rate: 0.0005
```

Then retrain: `python src/training/train.py`

---

## See Also

- [LEARNING_GUIDE.md](./LEARNING_GUIDE.md) — 6-phase explanation of every component
