# MindSync Model Flask API

A Flask-based REST API for mental health risk prediction using machine learning.

## Prerequisites

- Python 3.9 or higher
- Virtual environment (recommended)
- Gemini API key (for AI features)

## Setup Instructions

### 1. Environment Setup

The project uses a virtual environment located in `.venv/`. This has already been configured.

### 2. Install Dependencies

All required packages are listed in `requirements.txt` and have been installed:

```bash
# Activate virtual environment (if not already activated)
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install packages (already done)
pip install -r requirements.txt
```

### 3. Configure Environment Variables

A `.env` file has been created in the project root. **IMPORTANT**: You need to add your Gemini API key:

1. Open `.env` file
2. Replace `your_api_key_here` with your actual Gemini API key
3. Get your API key from: https://ai.google.dev/

```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 4. Run the Application

Use the virtual environment's Python interpreter:

```bash
# Windows
.venv\Scripts\python.exe app.py

# Or using PowerShell
& ".venv/Scripts/python.exe" app.py
```

The server will start on:
- Local: http://127.0.0.1:5000
- Network: http://0.0.0.0:5000

## Project Structure

```
mindsync-model-flask/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
├── Dockerfile                  # Docker configuration
├── artifacts/                  # Model files
│   ├── model.pkl              # Trained ML model
│   ├── preprocessor.pkl       # Data preprocessor
│   ├── model_coefficients.csv # Model coefficients
│   ├── healthy_cluster_avg.csv# Cluster analysis data
│   └── feature_importance.csv # Feature importance scores
└── notebook/
    └── final_FINAL.ipynb      # Training notebook

```

## API Endpoints

### Health Check
```bash
GET /
```

### Predict Mental Health Risk
```bash
POST /predict
Content-Type: application/json

{
  "age": 25,
  "gender": "Male",
  "occupation": "Student",
  // ... other features
}
```

## Testing

Test files are available:
- `test_api.py` - API endpoint tests
- `test_app_preprocessor.py` - Preprocessor tests
- `test_preprocessor_pkl.py` - Pickle file tests

## Development

The application runs in debug mode by default. Changes to the code will automatically reload the server.

## Key Features

- Mental health risk prediction using scikit-learn
- Data preprocessing pipeline with StandardScaler, OneHotEncoder
- Integration with Google Gemini AI
- RESTful API with JSON responses
- Model coefficients and feature importance analysis

## Dependencies

Main packages:
- Flask 3.1.2
- pandas 2.2.3
- numpy 1.26.4
- scikit-learn 1.5.2
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
