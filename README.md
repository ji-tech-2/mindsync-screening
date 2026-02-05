# MindSync - Mental Health Prediction API

Machine Learning API untuk prediksi kesehatan mental menggunakan Flask, scikit-learn, dan Google Gemini AI.

## 📁 Project Structure (Updated)

```
mindsync-model-flask/
├── flaskr/                    # Main application package
│   ├── __init__.py           # Application factory
│   ├── db.py                 # Database models (PostgreSQL)
│   ├── model.py              # ML model & preprocessing
│   ├── cache.py              # Valkey/Redis caching
│   ├── ai.py                 # Gemini AI integration
│   ├── predict.py            # Prediction routes
│   ├── templates/            # HTML templates (future)
│   └── static/               # Static files (future)
├── artifacts/                 # ML model artifacts
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── model_coefficients.csv
│   ├── feature_importance.csv
│   └── healthy_cluster_avg.csv
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_core_endpoints.py      # Health, predict, result, advice tests
│   └── test_weekly_daily_endpoints.py  # Weekly/daily suggestion tests
├── notebook/                  # Jupyter notebooks
│   └── final_FINAL.ipynb
├── .env                       # Environment variables
├── .dockerignore              # Docker ignore rules
├── Dockerfile                 # Docker configuration
├── wsgi.py                    # Application entry point
├── requirements.txt           # Python dependencies
└── README.md
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
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql://user:pass@host:port/dbname  # Optional for dev
VALKEY_URL=redis://localhost:6379  # Optional
```

### 3. Install & Run

**Option A: Using PowerShell Script (Recommended)**
```powershell
.\run.ps1
```

**Option B: Manual Installation**
```bash
# Install in development mode
pip install -e .


# Run application
python wsgi.py
```

**Option C: Flask Development Server**
```bash
set FLASK_APP=wsgi.py
set FLASK_DEBUG=True
flask run
```

## 📡 API Endpoints

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
```
GET /streak/<user_id>

Response: 200 OK
{
  "status": "success",
  "data": {
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "daily": {
      "current": 5,
      "last_date": "2026-02-02"
    },
    "weekly": {
      "current": 2,
      "last_date": "2026-02-02"
    }
  }
}
```

### Get User History
```
GET /history/<user_id>

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
```
GET /chart/weekly?user_id=<user_id>

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

Gunakan script `test_api_manual.py` untuk menguji API secara manual:

```bash
# Pastikan server sudah berjalan di terminal lain
python wsgi.py

# Di terminal baru, jalankan test script
python test_api_manual.py
```

Script ini akan:
1. Mengirim prediction request ke `/predict`
2. Polling `/result/<prediction_id>` sampai selesai
3. Menampilkan hasil prediksi dan AI advice

### Testing dengan cURL

```bash
# Health check
curl http://localhost:5000/

# Submit prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"screen_time_hours": 8, "work_screen_hours": 6, "leisure_screen_hours": 2, "sleep_hours": 7, "sleep_quality_1_5": 3, "stress_level_0_10": 5, "productivity_0_100": 70, "exercise_minutes_per_week": 150, "social_hours_per_week": 10}'

# Check result (ganti <prediction_id> dengan ID dari response sebelumnya)
curl http://localhost:5000/result/<prediction_id>
```

> **Note:** Automated pytest tests belum diimplementasikan. Untuk kontribusi test suite, silakan buat `tests/` directory dan tambahkan pytest ke requirements.

## 🐳 Docker Deployment

```bash
docker build -t mindsync-api .
docker run -p 5000:5000 --env-file .env mindsync-api
```

## 📝 Important Notes

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

### Migration dari app.py lama
File `app.py` lama masih ada untuk referensi. Struktur baru menggunakan:
- `wsgi.py` sebagai entry point
- `flaskr/` package untuk semua logic

## 📦 Main Dependencies

- Flask 3.1.2 - Web framework
- pandas 2.2.3 - Data processing
- numpy 1.26.4 - Numerical computing  
- scikit-learn 1.5.2 - Machine learning
- google-genai 1.57.0 - AI integration
- Flask-SQLAlchemy 3.1.1 - Database ORM
- valkey 6.1.1 - Caching
- psycopg2-binary 2.9.11 - PostgreSQL driver

See [pyproject.toml](pyproject.toml) for complete list.

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
- google-genai 1.57.0
- python-dotenv 1.2.1

See `requirements.txt` for complete list.

## Troubleshooting

### Module Import Errors
Ensure you're using the virtual environment's Python:
```bash
& ".venv/Scripts/python.exe" app.py
```

### API Key Missing
Check that `.env` file exists and contains valid `GEMINI_API_KEY`.

### Port Already in Use
If port 5000 is busy, modify the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```
