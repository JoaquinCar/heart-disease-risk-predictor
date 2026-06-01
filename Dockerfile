FROM python:3.11-slim

WORKDIR /app

# Copy requirements first — Docker caches this layer separately.
# If only source code changes, pip install is skipped on rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the code needed to run the API (not notebooks, data, venv)
COPY api/ ./api/
COPY src/ ./src/

# MLflow tracking URI points to the mounted mlruns/ volume (see docker-compose.yml).
# This decouples the container from local filesystem paths.
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
