# docker/Dockerfile.ml
# ML Model Serving - Simplified for Task 5
# No alert system - just predictions

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ml_server.py .
COPY predictor.py .
COPY auth_system.py .
COPY logging_config.py .
COPY privacy_utils.py .
COPY privacy_config.yaml .
COPY prometheus_metrics.py .

# Copy src directory (contains feature_engineering)
COPY src/ ./src/

# Create directories
RUN mkdir -p logs models data config

# Copy model metadata (small file, can be in GitHub)
COPY models/xgboost_model_metadata.json ./models/

# Model file will be mounted as volume or generated at runtime
# Not copied during build to avoid large file issues

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Set environment variables
ENV MAX_WORKERS=4
ENV BATCH_SIZE=32
ENV PYTHONUNBUFFERED=1

# Run application
CMD ["uvicorn", "ml_server:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]