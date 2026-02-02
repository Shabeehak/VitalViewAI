# docker/Dockerfile.ml
# ML Model Serving Container for VitalViewAI
# Handles XGBoost predictions with high performance

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ml_server.py .
COPY predictor.py .
COPY privacy_utils.py .
COPY logging_config.py .
COPY privacy_config.yaml .
COPY config/config.yaml config/
COPY src/ src/

# Copy trained models
COPY models/ models/

# Create necessary directories
RUN mkdir -p /app/logs

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Run the application
CMD ["python", "-u", "ml_server.py"]