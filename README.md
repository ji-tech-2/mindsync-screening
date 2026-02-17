# MindSync - Mental Health Prediction API (Inference Service)

**Independent microservice** untuk serving ML predictions menggunakan Flask, scikit-learn, dan Google Gemini AI.

[![CI](https://github.com/YOUR_USERNAME/mindsync-model-flask/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/mindsync-model-flask/actions/workflows/ci.yml)
[![Artifact Build](https://github.com/YOUR_USERNAME/mindsync-model-flask/actions/workflows/artifact.yml/badge.svg)](https://github.com/YOUR_USERNAME/mindsync-model-flask/actions/workflows/artifact.yml)
[![CD](https://github.com/YOUR_USERNAME/mindsync-model-flask/actions/workflows/cd.yml/badge.svg)](https://github.com/YOUR_USERNAME/mindsync-model-flask/actions/workflows/cd.yml)
[![codecov](https://codecov.io/gh/YOUR_USERNAME/mindsync-model-flask/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/mindsync-model-flask)

## 📊 Code Quality Standards

| Metric                | Standard    | Status      |
| --------------------- | ----------- | ----------- |
| Code Coverage         | ≥ 80%       | ✅ Enforced |
| Cyclomatic Complexity | ≤ 15        | ✅ Enforced |
| Code Duplication      | None        | ✅ Enforced |
| Security Issues       | No critical | ✅ Enforced |
| Code Quality Score    | ≥ 8.0/10    | ✅ Enforced |

> **Quality Assurance**: All code is automatically checked in CI/CD with actual metrics displayed (coverage %, Pylint score, complexity values, issue counts). See workflow runs for detailed quality reports.

> **Note**: Service ini adalah microservice yang terpisah dan independen. Model artifacts di-download dari Weights & Biases yang di-upload oleh training service secara terpisah.

## 🏗️ Architecture Overview

- **Training Service**: `mindsync-model-training` - Independent service, training model dan upload ke W&B
- **Inference Service**: `mindsync-model-flask` (service ini) - Independent service, download dari W&B dan serving predictions
- **Communication**: Via Weights & Biases artifact storage (no direct connection)

## 📁 Project Structure (Updated)

```
mindsync-model-flask/
├── flaskr/                    # Main application package
│   ├── __init__.py           # Application factory
│   ├── db.py                 # Database models (PostgreSQL)
│   ├── model.py              # ML model & preprocessing (+ W&B download)
│   ├── cache.py              # Valkey/Redis caching
│   ├── ai.py                 # Gemini AI integration
│   ├── predict.py            # Prediction routes
│   ├── templates/            # HTML templates (future)
│   └── static/               # Static files (future)
├── artifacts/                 # ML model artifacts (downloaded from W&B)
│   ├── model.pkl             # ⬇️ Downloaded from W&B
│   ├── preprocessor.pkl      # ⬇️ Downloaded from W&B
│   ├── model_coefficients.csv    # ⬇️ Downloaded from W&B
│   ├── feature_importance.csv    # ⬇️ Downloaded from W&B
│   └── healthy_cluster_avg.csv   # 📌 LOCAL FILE (preserved, not overwritten)
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_core_endpoints.py      # Health, predict, result, advice tests
│   └── test_weekly_daily_endpoints.py  # Weekly/daily suggestion tests
├── notebook/                  # Jupyter notebooks
│   └── final_FINAL.ipynb
├── download_artifacts.py      # Script to download from W&B
├── .env                       # Environment variables
├── .dockerignore              # Docker ignore rules
├── Dockerfile                 # Docker configuration
├── wsgi.py                    # Application entry point
├── requirements.txt           # Python dependencies (includes wandb)
└── README.md
```

## 🔄 Model Artifacts Management

### Automatic Download at Startup

Model artifacts automatically download from W&B when Flask app starts (see `flaskr/model.py`).

**Note**: `healthy_cluster_avg.csv` is a **local file** that will **NOT** be overwritten by W&B downloads. Other artifacts (model.pkl, preprocessor.pkl, coefficients, etc.) will be updated with latest from W&B.

### CI/CD Workflow

**Automatic Deployment on Push to Main**:

```
Developer creates PR → Tests run → Merge to main
    ↓
GitHub Actions: deploy.yml auto-triggers
    ↓
Download LATEST model from W&B
    ↓
Run tests
    ↓
Build Docker image ✅
```

**Triggers**:

- `push` to `main` branch
- `workflow_dispatch` - Manual trigger with version option

### Standard Development Flow

```bash
# 1. Create feature branch
git checkout -b feature/improve-api

# 2. Make changes to flaskr/predict.py, etc.
# 3. Create PR and get review
# 4. Merge to main → Deployment auto-runs! ✅
# Latest model from W&B will be downloaded
```

### Quick Model Update (No Code Changes)

```bash
# Force deployment to get latest model from W&B
git commit --allow-empty -m "chore: update to latest model"
git push origin main
```

## 🔄 Model Artifacts Management

### Automatic Download at Startup

Service ini akan otomatis mencoba download artifacts dari W&B saat startup:

1. Saat `init_app()` dipanggil, service cek W&B untuk artifacts terbaru
2. Jika tersedia, download ke direktori `artifacts/`
3. Jika gagal, gunakan artifacts lokal (fallback)

### Manual Download

Download artifacts secara manual:

```bash
# Set environment variables terlebih dahulu
export WANDB_API_KEY=your-api-key
export WANDB_PROJECT=mindsync-model
export WANDB_ENTITY=your-username

# Run download script
python download_artifacts.py
```

### Environment Variables untuk W&B

```env
# Weights & Biases Configuration
WANDB_API_KEY=your-wandb-api-key
WANDB_PROJECT=mindsync-model
WANDB_ENTITY=your-wandb-username

# Artifact version ('latest' or specific version like 'v0', 'v1')
ARTIFACT_VERSION=latest

# Optional: Skip W&B download (use local artifacts only)
SKIP_WANDB_DOWNLOAD=false
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment (if not exists)
python -m venv .venv

# Activate (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (CMD)
.venv\Scripts\activate.bat
```

### 2. Configure Environment Variables

Edit [.env](.env) file:

```env
# Gemini AI
GEMINI_API_KEY=your_gemini_api_key_here

# Database (Optional for dev)
DATABASE_URL=postgresql://user:pass@host:port/dbname

# Cache (Optional)
VALKEY_URL=redis://localhost:6379

# Weights & Biases (Required for artifact download)
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=mindsync-model
WANDB_ENTITY=your_username
ARTIFACT_VERSION=latest
```

### 3. Download Model Artifacts

**First time setup - download artifacts from W&B:**

```bash
# Login to W&B (only needed once)
wandb login

# Download artifacts
python download_artifacts.py
```

### 4. Install & Run

**Option A: Development Mode (Flask Dev Server)**

```bash
# Set environment variables
set FLASK_APP=wsgi:app
set FLASK_DEBUG=True
flask run --host=0.0.0.0 --port=5000
```

**Option B: Production Mode (Gunicorn)**

```bash
# Run with gunicorn using configuration file
gunicorn --config gunicorn_config.py wsgi:app

# Or with custom settings
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 wsgi:app
```

**Option C: Manual Testing**

```bash
# Quick test with flask dev server
python -m flask --app wsgi:app run
```

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t mindsync-inference .
```

### Run Container (Standalone)

```bash
docker run -p 5000:5000 \
  -e WANDB_API_KEY=your-api-key \
  -e WANDB_PROJECT=mindsync-model \
  -e WANDB_ENTITY=your-username \
  -e GEMINI_API_KEY=your-gemini-key \
  mindsync-inference
```

### Docker Compose (With Dependencies)

Service ini punya `docker-compose.yml` sendiri yang include database dan cache:

```bash
# Start full stack
docker-compose up

# Start hanya inference (tanpa DB/cache)
docker-compose up inference
```

### CI/CD Integration

Service ini memiliki GitHub Actions workflow sendiri di `.github/workflows/`:

- **deploy.yml** - Build dan deploy inference service
- **test.yml** - Automated testing untuk PR dan commits

Setup secrets di repository:

- `WANDB_API_KEY`
- `WANDB_ENTITY`
- `GEMINI_API_KEY`

## 📡 API Endpoints

### Authentication

Most endpoints require a valid JWT token in the `auth_token` httpOnly cookie:

- **User Access**: Authenticate via the main authentication service to get JWT token
- **Guest Access**: Automatic identification via IP hash (no token needed for guest predictions)

Endpoints that require authentication will return `401 Unauthorized` without a valid token.

### Health Check

```
GET /
Response: {"status": "active", "message": "MindSync Model API is running."}
```

### Predict Mental Health Score

```
POST /predict
Content-Type: application/json

{
  "screen_time_hours": 8.5,
  "work_screen_hours": 6.0,
  "leisure_screen_hours": 2.5,
  "sleep_hours": 6.5,
  "sleep_quality_1_5": 3,
  "stress_level_0_10": 7,
  "productivity_0_100": 65,
  "exercise_minutes_per_week": 120,
  "social_hours_per_week": 5.0,
  "user_id": "optional-uuid"
}

Response: 202 Accepted
{
  "prediction_id": "uuid",
  "status": "processing",
  "message": "Prediction is being processed..."
}
```

### Get Prediction Result

```
GET /result/<prediction_id>

Response: 200 OK (when ready)
{
  "status": "ready",
  "result": {
    "prediction_score": 45.3,
    "health_level": "average",
    "wellness_analysis": {
      "areas_for_improvement": [...],
      "strengths": [...]
    },
    "advice": {
      "description": "...",
      "factors": {...}
    }
  },
  "created_at": "2026-01-22T...",
  "completed_at": "2026-01-22T..."
}
```

### Get User Streak

Returns sparse data showing whether the user took a screening on specific days/weeks, plus current streak counts:

- **Daily**: Returns Mon-Sun of the current week
- **Weekly**: Returns the last 7 weeks
- **Current Streaks**: Daily and weekly consecutive streak counts

**Authentication**: Requires valid JWT token in `auth_token` httpOnly cookie

```
GET /streak

Response: 200 OK
{
  "status": "success",
  "data": {
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "current_streak": {
      "daily": 5,
      "daily_last_date": "2026-02-16",
      "weekly": 2,
      "weekly_last_date": "2026-02-10"
    },
    "daily": [
      {
        "date": "2026-02-10",
        "label": "Mon",
        "has_screening": false
      },
      {
        "date": "2026-02-11",
        "label": "Tue",
        "has_screening": true
      },
      {
        "date": "2026-02-12",
        "label": "Wed",
        "has_screening": true
      },
      {
        "date": "2026-02-13",
        "label": "Thu",
        "has_screening": false
      },
      {
        "date": "2026-02-14",
        "label": "Fri",
        "has_screening": true
      },
      {
        "date": "2026-02-15",
        "label": "Sat",
        "has_screening": false
      },
      {
        "date": "2026-02-16",
        "label": "Sun",
        "has_screening": true
      }
    ],
    "weekly": [
      {
        "week_start": "2025-12-29",
        "week_end": "2026-01-04",
        "label": "Dec 29 - Jan 4",
        "has_screening": false
      },
      {
        "week_start": "2026-01-05",
        "week_end": "2026-01-11",
        "label": "Jan 5-11",
        "has_screening": true
      },
      {
        "week_start": "2026-01-12",
        "week_end": "2026-01-18",
        "label": "Jan 12-18",
        "has_screening": false
      },
      {
        "week_start": "2026-01-19",
        "week_end": "2026-01-25",
        "label": "Jan 19-25",
        "has_screening": true
      },
      {
        "week_start": "2026-01-26",
        "week_end": "2026-02-01",
        "label": "Jan 26 - Feb 1",
        "has_screening": false
      },
      {
        "week_start": "2026-02-02",
        "week_end": "2026-02-08",
        "label": "Feb 2-8",
        "has_screening": true
      },
      {
        "week_start": "2026-02-09",
        "week_end": "2026-02-15",
        "label": "Feb 9-15",
        "has_screening": true
      }
    ]
  }
}
```

### Get User History

**Authentication**: Requires valid JWT token in `auth_token` httpOnly cookie

```
GET /history

Response: 200 OK
{
  "count": 5,
  "data": [
    {
      "advice": {
        "description": "Great job maintaining good sleep habits...",
        "factors": {
           "Screen Time": {
              "advices": ["Try the 20-20-20 rule..."],
              "references": ["https://..."]
           }
        }
      }
      "created_at": "2026-02-04T10:00:00",
      "health_level": "healthy",
      "prediction_id": "uuid-1...",
      "prediction_score": 35.5,
      "wellness_analysis": {
        "areas_for_improvement": [
           {"feature": "Screen Time", "impact_score": 5.2}
        ]
        "strengths": [
           {"feature": "Sleep Quality", "impact_score": -10.5}
        ],
      },
    },
  ],
  "status": "success"
}
```

### Get Weekly Chart

**Authentication**: Requires valid JWT token in `auth_token` httpOnly cookie

```
GET /chart/weekly

Response: 200 OK
{
  "data": [
    {
      "date": "2026-02-04",
      "exercise_duration": 0.0,
      "has_data": true,
      "label": "Wed",
      "mental_health_index": 45.0,
      "productivity": 40.0,
      "screen_time": 12.0,
      "sleep_duration": 4.5,
      "sleep_quality": 2.0,
      "social_activity": 1.0,
      "stress_level": 8.0
    },
    {
      "date": "2026-02-05",
      "exercise_duration": 60.0,
      "has_data": true,
      "label": "Thu",
      "mental_health_index": 92.5,
      "productivity": 95.0,
      "screen_time": 4.0,
      "sleep_duration": 8.0,
      "sleep_quality": 5.0,
      "social_activity": 5.0,
      "stress_level": 1.5
    }
  ],
  "status": "success"
}
```

### Get Daily Suggestion

Returns AI-powered daily suggestions based on today's areas of improvement.

**Authentication**: Requires valid JWT token in `auth_token` httpOnly cookie

```
GET /daily-suggestion

Response: 200 OK
{
  "status": "success",
  "cached": false,
  "date": "2026-02-17",
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "stats": {
    "predictions_today": 3
  },
  "areas_of_improvement": [
    {
      "factor_name": "Screen Time",
      "impact_score": 5.2
    },
    {
      "factor_name": "Sleep Quality",
      "impact_score": 3.1
    }
  ],
  "suggestion": {
    "message": "Try to reduce screen time in the evening for better sleep quality.",
    "factors": {...}
  }
}
```

### Get Weekly Critical Factors

Returns the top 3 most frequent areas of improvement from the last week with AI analysis.

**Authentication**: Requires valid JWT token in `auth_token` httpOnly cookie

**Query Parameters**:

- `days` (optional): Number of days to look back (default: 7)

```
GET /weekly-critical-factors?days=7

Response: 200 OK
{
  "status": "success",
  "cached": false,
  "period": {
    "start_date": "2026-02-10T00:00:00",
    "end_date": "2026-02-17T23:59:59",
    "days": 7
  },
  "stats": {
    "total_predictions": 15,
    "user_id": "550e8400-e29b-41d4-a716-446655440000"
  },
  "top_critical_factors": [
    {
      "factor_name": "Screen Time",
      "count": 8,
      "avg_impact_score": 4.5
    },
    {
      "factor_name": "Sleep Quality",
      "count": 6,
      "avg_impact_score": 3.2
    },
    {
      "factor_name": "Stress Level",
      "count": 5,
      "avg_impact_score": 2.8
    }
  ],
  "advice": {
    "description": "Focus on managing screen time and improving sleep patterns for better mental health.",
    "factors": {...}
  }
}
```

### Generate AI Advice

Generate AI advice manually for a wellness analysis result.

```
POST /advice
Content-Type: application/json

{
  "prediction_score": 45.3,
  "mental_health_category": "average",
  "wellness_analysis": {
    "areas_for_improvement": [
      {"feature": "Screen Time", "impact_score": 5.2}
    ],
    "strengths": [
      {"feature": "Exercise", "impact_score": -8.5}
    ]
  }
}

Response: 200 OK
{
  "ai_advice": {
    "description": "Great job with exercise! Try reducing screen time...",
    "factors": {
      "Screen Time": {
        "advices": ["Try the 20-20-20 rule..."],
        "references": ["https://..."]
      }
    }
  },
  "status": "success"
}
```

## 🏗️ Architecture

### Application Factory Pattern

- Menggunakan `create_app()` factory function
- Modular blueprint-based routing
- Easy testing dan multiple configurations

### Components

- **DB Layer** (`db.py`): PostgreSQL models dengan SQLAlchemy
- **Model Layer** (`model.py`): Custom Ridge Regression + preprocessing
- **Cache Layer** (`cache.py`): Valkey/Redis untuk caching hasil
- **AI Layer** (`ai.py`): Google Gemini untuk AI advice
- **Routes** (`predict.py`): Blueprint untuk API endpoints

### Background Processing

- Threading untuk async prediction processing
- Status tracking: `processing` → `partial` → `ready`
- Fallback ke database jika cache tidak tersedia

## 🧪 Testing

### Manual API Testing

Gunakan script `test_predict.py` untuk menguji API secara manual:

```bash
# Pastikan server sudah berjalan di terminal lain (production mode)
gunicorn --config gunicorn_config.py wsgi:app

# Atau dengan Flask dev server
flask --app wsgi:app run

# Di terminal baru, jalankan test script
python test_predict.py
```

Script ini akan:

1. Mengirim prediction request ke `/predict`
2. Polling `/result/<prediction_id>` sampai selesai
3. Menampilkan hasil prediksi dan AI advice

### Testing dengan cURL

```bash
# Health check (no auth required)
curl http://localhost:5000/

# Submit prediction (guest or user)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"screen_time_hours": 8, "work_screen_hours": 6, "leisure_screen_hours": 2, "sleep_hours": 7, "sleep_quality_1_5": 3, "stress_level_0_10": 5, "productivity_0_100": 70, "exercise_minutes_per_week": 150, "social_hours_per_week": 10}'

# Optional: Submit with user_id (still supported)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"screen_time_hours": 8, "user_id": "550e8400-e29b-41d4-a716-446655440000", ...}'

# Check result (replace <prediction_id> with ID from response above)
curl http://localhost:5000/result/<prediction_id>

# Get user streak (requires JWT)
curl -b "auth_token=<your_jwt_token>" http://localhost:5000/streak

# Get user history (requires JWT)
curl -b "auth_token=<your_jwt_token>" http://localhost:5000/history

# Get weekly chart (requires JWT)
curl -b "auth_token=<your_jwt_token>" http://localhost:5000/chart/weekly
```

**Note**: Replace `<your_jwt_token>` with a valid JWT token from the authentication service. The token is passed via httpOnly cookie for security.

> **Note:** Automated pytest tests belum diimplementasikan. Untuk kontribusi test suite, silakan buat `tests/` directory dan tambahkan pytest ke requirements.

## � Code Quality & Coverage

This project enforces enterprise-grade code quality standards using industry-standard tools.

### Quality Standards

| Metric                | Standard            | Tool          |
| --------------------- | ------------------- | ------------- |
| Code Coverage         | ≥ 80%               | pytest-cov    |
| Cyclomatic Complexity | ≤ 15                | Radon, Flake8 |
| Code Duplication      | None                | Pylint        |
| Security Issues       | No blocker/critical | Bandit        |
| Code Quality Score    | ≥ 8.0/10            | Pylint        |

### Quick Commands

```bash
# Run all quality checks
python check_quality.py

# Run quality checks with auto-fix
python check_quality.py --fix

# Or using Makefile
make quality           # Run all checks
make quality-fix       # Run with auto-fix
make test              # Run tests only
make coverage          # Run tests with coverage report
```

### Individual Quality Checks

#### 1. Code Coverage

```bash
# Run tests with coverage report
pytest --cov=flaskr --cov-report=term-missing --cov-report=html

# View HTML coverage report
# Open htmlcov/index.html in browser

# Or using Makefile
make coverage
```

Coverage reports show:

- Line coverage percentage
- Missing lines for each file
- Branch coverage
- HTML report with detailed line-by-line coverage

#### 2. Code Formatting

```bash
# Check formatting
black --check --diff flaskr tests *.py

# Auto-fix formatting
black flaskr tests *.py

# Or using Makefile
make format
```

#### 3. Linting & Style

```bash
# Run Flake8 (style + complexity)
flake8 flaskr tests --config=.flake8

# Run Pylint (code quality)
pylint flaskr --rcfile=.pylintrc

# Or using Makefile
make lint
```

#### 4. Cyclomatic Complexity

```bash
# Check complexity with Radon
radon cc flaskr -a -nb           # Cyclomatic complexity
radon mi flaskr -nb              # Maintainability index

# Or using Makefile
make complexity
```

Complexity grades:

- **A**: 1-5 (simple)
- **B**: 6-10 (well-structured)
- **C**: 11-15 (acceptable) ⚠️
- **D+**: 16+ (needs refactoring) ❌

#### 5. Security Scanning

```bash
# Run Bandit security scanner
bandit -r flaskr -ll -f screen

# Or using Makefile
make security
```

#### 6. Code Duplication

```bash
# Check for duplicate code
pylint flaskr --disable=all --enable=duplicate-code

# Or using Makefile
make duplication
```

### Configuration Files

- [.pylintrc](.pylintrc) - Pylint configuration
- [.flake8](.flake8) - Flake8 configuration
- [pyproject.toml](pyproject.toml) - Black, coverage, pytest configuration
- [pytest.ini](pytest.ini) - Pytest settings

### CI/CD Integration

Quality checks run automatically on:

- Every push to `main` or `develop` branches
- Every pull request
- Manual workflow dispatch

See [.github/workflows/quality.yml](.github/workflows/quality.yml) for CI configuration.

### Pre-commit Hooks (Optional)

To run quality checks before each commit:

```bash
# Install pre-commit hooks (if using pre-commit)
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

### Development Workflow

1. **Before committing**:

   ```bash
   # Format code
   make format

   # Run quality checks
   make quality
   ```

2. **Fix issues**:

   ```bash
   # Auto-fix formatting
   make quality-fix

   # Manually fix linting/complexity issues
   # Check output of: make lint
   ```

3. **Verify coverage**:

   ```bash
   make coverage
   # Ensure coverage ≥ 80%
   ```

4. **Commit and push**:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   git push
   ```

### Quality Gate

The following checks must pass before merging:

✅ All tests pass  
✅ Code coverage ≥ 80%  
✅ No functions with complexity > 15  
✅ No code duplication  
✅ No blocker/critical security issues  
✅ Pylint score ≥ 8.0/10  
✅ Code formatted with Black

## �🐳 Docker Deployment

```bash
docker build -t mindsync-api .
docker run -p 5000:5000 --env-file .env mindsync-api
```

## 📝 Important Notes

- **JWT Authentication**: Some endpoints require a valid JWT token in the `auth_token` httpOnly cookie:
  - `/streak`, `/history`, `/chart/weekly`, `/daily-suggestion`, `/weekly-critical-factors` require authentication
  - The token is obtained from the authentication service
  - Guest users can still use `/predict` without authentication (identified via IP hash)
- **Storage backend (Database atau Valkey/Redis)**: Minimal **salah satu** harus tersedia agar flow `/predict` → `/result` bisa bekerja penuh (status bisa berubah menjadi `ready`).
  - Jika `DB_DISABLED=True` **dan** Valkey/Redis tidak tersedia, endpoint `/predict` tetap akan mengembalikan `202` dengan `prediction_id`, tetapi `/result` untuk ID tersebut tidak akan pernah selesai (status tidak akan menjadi `ready`).
- **Database**: Bisa dibuat optional untuk development **jika** Valkey/Redis aktif; app akan menampilkan warning tetapi prediksi tetap bisa diproses dan dilacak melalui cache.
- **Valkey/Redis**: Optional **jika** database aktif; tanpa cache performa bisa lebih lambat, namun prediksi dan tracking status tetap berjalan melalui database.
- **Gemini API**: Wajib untuk fitur AI advice; tanpa `GEMINI_API_KEY` endpoint terkait AI akan gagal (sebaiknya ditangani dengan error yang jelas di API).

## 🔧 Development Guide

### Adding New Routes

1. Buat blueprint baru di `flaskr/`
2. Register blueprint di `flaskr/__init__.py`

### Adding Tests

1. Tambahkan test file di `tests/`
2. Follow naming convention `test_*.py`

## 📦 Main Dependencies

- Flask 3.1.2 - Web framework
- pandas 2.2.3 - Data processing
- numpy 1.26.4 - Numerical computing
- scikit-learn 1.5.2 - Machine learning
- google-genai 1.57.0 - AI integration
- Flask-SQLAlchemy 3.1.1 - Database ORM
- valkey 6.1.1 - Caching
- psycopg2-binary 2.9.11 - PostgreSQL driver
- gunicorn 23.0.0 - Production WSGI server

See [requirements.txt](requirements.txt) for complete list.

## 🚀 Production Deployment

### Gunicorn Configuration

The application includes a production-ready Gunicorn configuration in `gunicorn_config.py`:

- **Workers**: Auto-calculated based on CPU cores (2 × cores + 1)
- **Timeout**: 120 seconds (suitable for ML inference)
- **Binding**: 0.0.0.0:5000 (configurable via PORT env var)
- **Logging**: JSON-formatted logs to stdout/stderr
- **Health checks**: Built-in monitoring hooks
- **Graceful restarts**: Zero-downtime deployments

### Environment Variables

```bash
# Server Configuration
PORT=5000                    # Server port (default: 5000)
LOG_LEVEL=info               # Logging level

# Gunicorn Configuration (Memory-Optimized for ML Models)
GUNICORN_WORKERS=2           # Number of worker processes (each loads full ML model)
GUNICORN_THREADS=2           # Threads per worker (shared memory)
GUNICORN_WORKER_CLASS=gthread # Use threaded workers for efficiency
# See MEMORY-OPTIMIZATION.md for tuning guidance

# Application Configuration
GEMINI_API_KEY=xxx           # Required for AI features
DATABASE_URL=postgresql://   # Optional database
VALKEY_URL=redis://          # Optional cache
WANDB_API_KEY=xxx            # For artifact downloads
```

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Create Pull Request

## 📄 License

This project is part of MindSync application.

---

**Need Help?** Check the issues or create a new one!

## 🐛 Troubleshooting

### Module Import Errors

Ensure you're using the virtual environment's Python:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Linux/Mac
source .venv/bin/activate
pip install -r requirements.txt
```

### API Key Missing

Check that `.env` file exists and contains valid `GEMINI_API_KEY` and `WANDB_API_KEY`.

### Port Already in Use

Change the port using environment variable:

```bash
# Windows
set PORT=5001
gunicorn --config gunicorn_config.py wsgi:app

# Linux/Mac
PORT=5001 gunicorn --config gunicorn_config.py wsgi:app
```

### Gunicorn Worker Timeout

If predictions take too long, increase the timeout:

```bash
gunicorn --config gunicorn_config.py --timeout 300 wsgi:app
```
