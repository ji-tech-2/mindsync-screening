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
COPY pyproject.toml .
COPY .flake8 .
COPY .pylintrc .
COPY .coveragerc .
COPY requirements-dev.txt .

# Install test dependencies from requirements-dev.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# Run code quality checks
RUN echo "=== Code Formatting Check ===" && \
    black --check --diff flaskr/ tests/ *.py

RUN echo "=== Flake8 Style & Complexity Check ===" && \
    flake8 flaskr/ tests/ --config=.flake8 --statistics

RUN echo "=== Pylint Code Quality Check ===" && \
    pylint flaskr --rcfile=.pylintrc --fail-under=8.0

RUN echo "=== Cyclomatic Complexity Check ===" && \
    radon cc flaskr -a -nb --total-average

RUN echo "=== Security Scan ===" && \
    bandit -r flaskr -ll -f screen

RUN echo "=== Code Duplication Check ===" && \
    (pylint flaskr --disable=all --enable=duplicate-code --rcfile=.pylintrc || true)

# Run tests with coverage (80% minimum)
RUN echo "=== Running Tests with Coverage ===" && \
    pytest tests/ -v --tb=short --cov=flaskr --cov-report=term-missing --cov-fail-under=80

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