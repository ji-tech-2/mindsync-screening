# 1. Base Image
FROM python:3.11.9-slim

# 2. Optimization
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Work Directory
WORKDIR /app

# 4. Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Code and Artifacts
COPY artifacts/ ./artifacts/
COPY app.py .

# 6. Security: Run as non-root user
RUN useradd -m modeluser
USER modeluser

# 7. Expose Port 5000 (Matches your app.py)
EXPOSE 5000

# 8. Run Command
# We force Gunicorn to bind to 5000 to match your expectations
CMD ["python", "app.py"]