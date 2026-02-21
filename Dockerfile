# Unified Dockerfile - Runs all services in one container
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY streaming_api_server.py .
COPY ml_server.py .
COPY predictor.py .
COPY streamlit_dashboard.py .
COPY auth_system.py .
COPY logging_config.py .
COPY privacy_utils.py .
COPY privacy_config.yaml .
COPY prometheus_metrics.py .

# Copy source directory
COPY src/ ./src/

# Copy model metadata (model will be downloaded at runtime)
COPY models/xgboost_model_metadata.json ./models/

# Create necessary directories
RUN mkdir -p logs models data config

# Create startup scripts for each service
RUN echo '#!/bin/bash\n\
exec uvicorn streaming_api_server:app --host 127.0.0.1 --port 8000 --log-level info\n\
' > /app/start-api.sh && chmod +x /app/start-api.sh

RUN echo '#!/bin/bash\n\
exec uvicorn ml_server:app --host 127.0.0.1 --port 8001 --log-level info\n\
' > /app/start-ml.sh && chmod +x /app/start-ml.sh

RUN echo '#!/bin/bash\n\
exec streamlit run streamlit_dashboard.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true --browser.gatherUsageStats=false\n\
' > /app/start-dashboard.sh && chmod +x /app/start-dashboard.sh

# Create supervisor configuration
RUN echo '[supervisord]\n\
nodaemon=true\n\
logfile=/app/logs/supervisord.log\n\
pidfile=/app/logs/supervisord.pid\n\
loglevel=info\n\
\n\
[program:api-server]\n\
command=/app/start-api.sh\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/app/logs/api-server.err.log\n\
stdout_logfile=/app/logs/api-server.out.log\n\
priority=10\n\
\n\
[program:ml-server]\n\
command=/app/start-ml.sh\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/app/logs/ml-server.err.log\n\
stdout_logfile=/app/logs/ml-server.out.log\n\
priority=20\n\
\n\
[program:dashboard]\n\
command=/app/start-dashboard.sh\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/app/logs/dashboard.err.log\n\
stdout_logfile=/app/logs/dashboard.out.log\n\
priority=30\n\
' > /etc/supervisor/conf.d/supervisord.conf

# Set environment variables for internal communication
ENV API_BASE=http://127.0.0.1:8000
ENV ML_API_BASE=http://127.0.0.1:8001

# Expose ONLY port 8501 (dashboard)
EXPOSE 8501

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start all services with supervisor
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]