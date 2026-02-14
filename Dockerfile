# 1. Base Image
FROM python:3.11.9-slim AS base

# 2. Optimization
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Work Directory
WORKDIR /app

# 4. Install Dependencies
# Install dependencies required for building some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Code and Artifacts
COPY artifacts/ ./artifacts/
COPY wsgi.py .
COPY gunicorn_config.py .
COPY custom_ridge.py .
COPY flaskr/ ./flaskr/

# Test Stage
FROM base AS test
COPY tests/ ./tests/
COPY pytest.ini .
COPY requirements-dev.txt .

# Install test dependencies from requirements-dev.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# Run linting
RUN flake8 flaskr/ tests/ --max-line-length=88
RUN black --check flaskr/ tests/

# Run tests
RUN pytest tests/ -v --tb=short

# Final Stage
FROM base

# 6. Security: Run as non-root user
RUN useradd -m -u 1000 modeluser && \
    chown -R modeluser:modeluser /app
USER modeluser

# 7. Expose Port 5000 (Matches your app)
EXPOSE 5000

# 8. Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/', timeout=5)"

# 9. Run with Gunicorn
CMD ["gunicorn", "--config", "gunicorn_config.py", "wsgi:app"]