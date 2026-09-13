# Learning Guide — PM2.5 Forecaster

Read this after you've run the project end-to-end at least once.  
Each phase builds on the previous one.

---

## Phase 1 — Machine Learning

### How the time-series data is structured

Raw data looks like this (each row = one hour):

```
hour | pm25 | temperature | humidity | wind_speed | pressure
  0  | 129  |    2.0      |   -16    |   4.0      |  1020
  1  | 148  |    2.0      |   -15    |   4.0      |  1020
  2  | 159  |    2.0      |   -11    |   5.0      |  1021
...
```

Time order matters. You cannot shuffle this data — future values would leak into training.

### How the 24-hour sliding window works

Think of it as a moving camera that looks at 24 rows at a time:

```
Window 0:  rows  0–23  → predict row 24's PM2.5
Window 1:  rows  1–24  → predict row 25's PM2.5
Window 2:  rows  2–25  → predict row 26's PM2.5
...
```

Each window becomes one training sample. With 43,000 rows and a window of 24, you get ~43,000 training samples.

### LSTM input/output shapes

```python
# Input to the model:
X.shape  # (batch_size, 24, 5)
          #  batch=64 samples
          #  24 = hours of history
          #  5  = features (pm25, temp, humidity, wind, pressure)

# What happens inside:
Input (64, 24, 5)
  → LSTM  → processes each of the 24 time steps sequentially
          → maintains a "hidden state" that carries information forward
  → takes the LAST hidden state (64, 64) — summary of all 24 hours
  → Linear(64 → 1)
  → Output (64,) — one prediction per sample in the batch
```

### How training works (the training loop)

```python
for epoch in range(50):          # repeat 50 times over all data
    for batch in dataloader:     # process 64 samples at a time
        pred = model(X_batch)    # forward pass
        loss = criterion(pred, y_batch)  # how wrong were we?
        loss.backward()          # compute gradients (backprop)
        optimizer.step()         # adjust weights to reduce loss
```

The model learns by repeatedly making predictions, measuring how wrong they are, and adjusting its weights in the direction that reduces error.

### How inference works

At prediction time:
1. Take 24 real observations from the API request
2. Scale them using the saved `scaler.pkl` (same scaling as training)
3. Pass through the LSTM
4. Un-scale the output back to real PM2.5 units
5. Return the number

The key insight: **the scaler must be the same one fit during training**. That's why we save it to `artifacts/scaler.pkl`.

---

## Phase 2 — FastAPI

### How FastAPI loads the model

```python
# In api/main.py:
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()    ← runs ONCE when the server starts
    yield               ← server handles requests here
```

Loading happens once at startup. Every incoming request reuses the already-loaded model in memory. If you loaded the model on every request, each prediction would take 1-2 seconds of disk I/O.

### How `/predict` works step by step

```
POST /predict
    │
    ▼
Pydantic validates request body
    │ 24 observations? all fields present? all numbers?
    ▼
obs_dicts = [obs.model_dump() for obs in request.observations]
    │
    ▼
src/inference/predict.predict(obs_dicts)
    │  → build numpy array (24, 5)
    │  → scale with saved scaler
    │  → run LSTM
    │  → un-scale
    │  → return float
    ▼
PredictResponse(predicted_pm25=115.4)
    │
    ▼
{"predicted_pm25": 115.4}
```

### How Pydantic validates input

```python
class PredictRequest(BaseModel):
    observations: List[Observation]

    @field_validator("observations")
    def must_be_24(cls, v):
        if len(v) != 24:
            raise ValueError("Exactly 24 observations required")
        return v
```

FastAPI automatically runs this validation before your function runs. Wrong count, missing field, or wrong type → HTTP 422 (Unprocessable Entity) with a clear error message. You never write `if len(obs) != 24: return error` manually.

---

## Phase 3 — Docker

### What the Dockerfile does, line by line

```dockerfile
FROM python:3.11-slim           # start with a pre-built Python environment
ENV PYTHONUNBUFFERED=1          # logs appear immediately (important in containers)
WORKDIR /app                    # all subsequent commands run from /app

COPY requirements.txt .         # copy this FIRST (caching trick below)
RUN pip install -r requirements.txt  # install deps (cached unless requirements change)

COPY config.yaml .              # copy app files
COPY src/ src/
COPY api/ api/
COPY artifacts/ artifacts/      # includes the trained model!

EXPOSE 8000                     # document the port (doesn't open it)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**The caching trick**: Docker builds layers in order. If `requirements.txt` hasn't changed, Docker reuses the cached `pip install` layer and skips it. By copying `requirements.txt` before `src/`, changing your Python code doesn't re-trigger a full `pip install`.

### What is a Docker image?

An **image** is a read-only snapshot of:
- A base operating system (slim Debian Linux)
- Python 3.11
- Your dependencies
- Your code
- Your model weights

Think of it as a shipping container blueprint. You build it once and run it anywhere.

### What is a Docker container?

A **container** is a running instance of an image — the actual executing process. You can run 10 containers from the same image simultaneously (for load balancing).

```
Image (blueprint)    →    Container (running instance)
   model.pt                 actual API serving requests
   scaler.pkl               port 8000 open
   uvicorn                  ~50MB RAM
```

### How the API runs inside the container

```bash
docker run -p 8000:8000 pm25-forecaster
```

- `-p 8000:8000` maps your laptop's port 8000 → container's port 8000
- Inside: uvicorn starts, loads model, listens on 0.0.0.0:8000
- From outside: `curl localhost:8000/health` goes through the port mapping

---

## Phase 4 — Continuous Integration (CI)

### What happens when you push code

```
git push origin main
    │
    ▼
GitHub receives the push
    │
    ▼
GitHub reads .github/workflows/ci.yml
    │
    ▼
Spins up an ubuntu-latest runner (a clean virtual machine)
    │
    ▼
Job: lint
  - checkout code
  - install ruff
  - ruff check src/ api/ tests/    ← fails if code has issues
  - ruff format --check            ← fails if formatting is wrong

Job: test  (only runs if lint passed)
  - checkout code
  - pip install -r requirements.txt
  - pytest tests/ -v               ← fails if any test fails

Job: docker-build  (only runs if test passed)
  - docker build                   ← fails if Dockerfile is broken
  - docker run + curl /health      ← fails if container crashes
```

### Why does the order matter?

Lint runs first because it's fastest (~10s). If there's a typo, you find out in 10 seconds, not 3 minutes. Tests run before Docker because no point building an image if the code is broken.

**Fast feedback = fewer wasted minutes.**

### Explaining every CI concept

| Concept | Example in ci.yml | What it means |
|---------|-------------------|---------------|
| Workflow | The entire `ci.yml` file | A collection of automated jobs |
| Trigger | `on: push: branches: [main]` | What causes the workflow to run |
| Job | `lint:`, `test:`, `docker-build:` | A group of steps on one machine |
| Runner | `runs-on: ubuntu-latest` | The VM that executes the job |
| Step | `- name: Run pytest` | One command or action in a job |
| Action | `uses: actions/checkout@v4` | Pre-built reusable step from marketplace |
| Env var | `PYTHONPATH: ${{ github.workspace }}` | Variable available to all steps |
| Artifact | `upload-artifact` step | File saved from the workflow run |
| Job dep | `needs: lint` | Forces sequential execution |

---

## Phase 5 — Continuous Delivery (CD)

### Docker image tagging

When CD runs, the image gets three tags:

```
ghcr.io/you/repo:latest          ← always points to newest
ghcr.io/you/repo:main            ← points to main branch build
ghcr.io/you/repo:sha-a1b2c3d     ← pinned to exact commit (never changes)
```

Why the SHA tag? Rollbacks. If `latest` breaks production, you run:
```bash
docker run ghcr.io/you/repo:sha-previousgoodcommit
```
And you're instantly back to the last good version.

### How GHCR authentication works

```yaml
- uses: docker/login-action@v3
  with:
    registry: ghcr.io
    username: ${{ github.actor }}           # your GitHub username
    password: ${{ secrets.GITHUB_TOKEN }}   # auto-created by GitHub
```

`GITHUB_TOKEN` is a temporary token GitHub creates for each workflow run. It has exactly the permissions you declared in the `permissions:` block. You never need to create or rotate it — GitHub handles it.

### How the image gets pushed

```
docker/build-push-action → BuildKit builds all layers
                         → pushes layers to GHCR
                         → attaches all three tags
```

Anyone with access to your GitHub repo can now pull:
```bash
docker pull ghcr.io/you/repo:latest
```

### Deploying to AWS / Azure / GCP (the next step)

The CD workflow you have is the foundation. To auto-deploy to a server, add one more job after `build-and-push`:

```yaml
deploy:
  needs: build-and-push
  steps:
    - name: Deploy to server
      uses: appleboy/ssh-action@v1
      with:
        host: ${{ secrets.SERVER_HOST }}
        username: ${{ secrets.SERVER_USER }}
        key: ${{ secrets.SSH_PRIVATE_KEY }}
        script: |
          docker pull ghcr.io/${{ github.repository }}:latest
          docker stop pm25 && docker rm pm25
          docker run -d --name pm25 -p 80:8000 ghcr.io/${{ github.repository }}:latest
```

This SSH's into your server and runs the pull + restart commands. The SSH key is stored as a GitHub secret.

---

## Phase 6 — What Would Change for Real Production

This project is intentionally simplified for learning. Here's what a real ML production service adds:

| Concern | Simple version (this project) | Production version |
|---------|-------------------------------|-------------------|
| **Model versioning** | One `model.pt` file | MLflow or W&B tracks every experiment |
| **Data validation** | None | Great Expectations or Pandera checks schema |
| **Model validation** | None | Challenger/champion A/B testing before deploy |
| **Monitoring** | None | Prometheus metrics, drift detection |
| **Logging** | print() | Structured JSON logs → Datadog/CloudWatch |
| **Secrets** | GITHUB_TOKEN | Vault, AWS Secrets Manager |
| **Rollback** | Manual `docker run :sha-xxx` | Automated health check → auto-rollback |
| **Scaling** | 1 container | Kubernetes HPA or AWS ECS auto-scaling |
| **Retraining** | Manual script | Scheduled pipeline (Airflow, Prefect) |
| **Artifacts** | Local `artifacts/` | S3, GCS, or Azure Blob Storage |
| **Config** | `config.yaml` in repo | Environment-specific configs, feature flags |

The key takeaway: **the CI/CD pipeline structure stays the same**. You just add more stages and validations. The `ci.yml` / `cd.yml` pattern you learned here is used by teams at every scale.

---

## Incremental Build Order

Build one piece at a time and test it before moving to the next:

```
Step 1  config.yaml + src/config.py
        → verify: python -c "from src.config import CFG; print(CFG.model.hidden_size)"

Step 2  scripts/download_data.py
        → verify: python scripts/download_data.py → data/raw/PRSA_data.csv exists

Step 3  src/data/preprocessing.py
        → verify: python src/data/preprocessing.py → prints shapes, scaler saved

Step 4  src/models/lstm.py
        → verify: python -c "from src.models.lstm import LSTMForecaster; m=LSTMForecaster(5); import torch; print(m(torch.randn(2,24,5)).shape)"

Step 5  src/training/train.py
        → verify: python src/training/train.py → model.pt created

Step 6  src/evaluation/evaluate.py
        → verify: python src/evaluation/evaluate.py → MAE/RMSE/R² printed

Step 7  src/inference/predict.py  (no standalone test — used by API)

Step 8  api/main.py
        → verify: uvicorn api.main:app --reload → curl localhost:8000/health

Step 9  tests/
        → verify: pytest tests/ -v → all green

Step 10 Dockerfile
        → verify: docker build -t pm25 . && docker run -p 8000:8000 pm25

Step 11 Push to GitHub
        → verify: CI workflow goes green in Actions tab

Step 12 Check GHCR
        → verify: ghcr.io/YOU/repo:latest appears under Packages
```
