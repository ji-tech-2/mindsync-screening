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
COPY flaskr/ ./flaskr/

# Test Stage
FROM base AS test
COPY tests/ ./tests/
# TODO: Add and run tests

# Final Stage
FROM base

# 6. Security: Run as non-root user
RUN useradd -m modeluser
USER modeluser

# 7. Expose Port 5000 (Matches your app.py)
EXPOSE 5000

# 8. Run Command
# We force Gunicorn to bind to 5000 to match your expectations
CMD ["python", "wsgi.py"]