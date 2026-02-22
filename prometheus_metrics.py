# prometheus_metrics.py
"""
Prometheus Metrics for VitalViewAI
Exposes metrics for Grafana dashboards
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time

# ============================================================================
# Metrics Definitions
# ============================================================================

# API Metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status_code']
)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

api_errors_total = Counter(
    'api_errors_total',
    'Total API errors',
    ['method', 'endpoint', 'error_type']
)

# ML Metrics
predictions_total = Counter(
    'predictions_total',
    'Total predictions made',
    ['risk_level']
)

model_inference_time_seconds = Histogram(
    'model_inference_time_seconds',
    'Model inference time in seconds',
    buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
)

# Patient Metrics
active_patients_total = Gauge(
    'active_patients_total',
    'Number of active patients'
)

alerts_triggered_total = Counter(
    'alerts_triggered_total',
    'Total alerts triggered',
    ['severity']
)

# System Metrics
service_uptime_seconds = Gauge(
    'service_uptime_seconds',
    'Service uptime in seconds'
)

# ============================================================================
# Metrics Helpers
# ============================================================================

class MetricsTracker:
    """Helper class to track metrics"""
    
    def __init__(self):
        self.start_time = time.time()
    
    def track_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Track API request metrics"""
        api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        api_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def track_error(self, method: str, endpoint: str, error_type: str):
        """Track API errors"""
        api_errors_total.labels(
            method=method,
            endpoint=endpoint,
            error_type=error_type
        ).inc()
    
    def track_prediction(self, risk_level: str, inference_time: float):
        """Track ML prediction"""
        predictions_total.labels(risk_level=risk_level).inc()
        model_inference_time_seconds.observe(inference_time)
    
    def track_alert(self, severity: str):
        """Track alert triggered"""
        alerts_triggered_total.labels(severity=severity).inc()
    
    def update_active_patients(self, count: int):
        """Update active patient count"""
        active_patients_total.set(count)
    
    def update_uptime(self):
        """Update service uptime"""
        uptime = time.time() - self.start_time
        service_uptime_seconds.set(uptime)

# Global tracker instance
metrics_tracker = MetricsTracker()

# ============================================================================
# FastAPI Endpoint
# ============================================================================

async def metrics_endpoint():
    """Expose metrics for Prometheus scraping"""
    metrics_tracker.update_uptime()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ============================================================================
# Usage Example
# ============================================================================

"""
# In streaming_api_server.py or ml_server.py:

from prometheus_metrics import metrics_tracker, metrics_endpoint

@app.get("/metrics")
async def metrics():
    return await metrics_endpoint()

@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        metrics_tracker.track_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration=duration
        )
        
        return response
    except Exception as e:
        metrics_tracker.track_error(
            method=request.method,
            endpoint=request.url.path,
            error_type=type(e).__name__
        )
        raise

# In predictor.py predict():
start = time.time()
result = model.predict(X)
inference_time = time.time() - start

metrics_tracker.track_prediction(
    risk_level=result['risk_level'],
    inference_time=inference_time
)

# Update patient count
metrics_tracker.update_active_patients(len(active_devices))
"""