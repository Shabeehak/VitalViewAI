# ml_server.py
"""
Production ML Model Serving API with Comprehensive Logging
High-performance prediction endpoint for real-time inference

Enhanced Features:
- Structured logging with correlation IDs
- Performance metrics tracking
- Comprehensive audit trail
- Error tracking and alerting
- Request/response logging

Usage:
    python ml_server.py
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import pandas as pd
import numpy as np
import time
import uuid
from datetime import datetime
import sys
sys.path.append('src')

from predictor import HealthPredictor
from privacy_utils import PrivacyManager

from logging_config import (
    setup_logging, get_logger,
    PerformanceLogger, AuditLogger, ErrorLogger
)

# Initialize logging
setup_logging(log_level="INFO", enable_json=True)
logger = get_logger(__name__)
perf_logger = PerformanceLogger()
audit_logger = AuditLogger()
error_logger = ErrorLogger()

# Initialize FastAPI
app = FastAPI(
    title="VitalViewAI ML Server",
    description="Production ML model serving with comprehensive logging",
    version="1.0.0"
)

# Global instances
predictor = None
privacy_manager = None

# Performance metrics
metrics = {
    'total_predictions': 0,
    'total_alerts': 0,
    'avg_inference_time_ms': 0.0,
    'uptime_seconds': 0,
    'start_time': None,
    'errors': 0
}

# ==================== MIDDLEWARE ====================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with correlation ID"""
    correlation_id = str(uuid.uuid4())
    request.state.correlation_id = correlation_id
    
    start_time = time.time()
    
    logger.info(
        f"Incoming ML server request",
        extra={
            'correlation_id': correlation_id,
            'method': request.method,
            'path': request.url.path,
            'client_ip': request.client.host if request.client else 'unknown'
        }
    )
    
    try:
        response = await call_next(request)
        response_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"ML server request completed",
            extra={
                'correlation_id': correlation_id,
                'method': request.method,
                'path': request.url.path,
                'status_code': response.status_code,
                'response_time_ms': response_time_ms
            }
        )
        
        # Log performance
        perf_logger.log_api_request(
            endpoint=request.url.path,
            method=request.method,
            response_time_ms=response_time_ms,
            status_code=response.status_code
        )
        
        return response
        
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        
        logger.error(
            f"ML server request failed: {str(e)}",
            exc_info=True,
            extra={
                'correlation_id': correlation_id,
                'method': request.method,
                'path': request.url.path,
                'response_time_ms': response_time_ms
            }
        )
        
        metrics['errors'] += 1
        raise

# ==================== REQUEST/RESPONSE MODELS ====================

class PredictionRequest(BaseModel):
    """Single prediction request"""
    patient_id: str
    vitals: List[Dict[str, Any]]
    user_id: Optional[str] = "anonymous"
    user_role: Optional[str] = "clinician"

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    requests: List[PredictionRequest]

class PredictionResponse(BaseModel):
    """Prediction response"""
    patient_id: str
    risk_score: float
    risk_level: str
    alert: bool
    interpretation: str
    timestamp: str
    inference_time_ms: float
    correlation_id: str

# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Initialize model and services on startup"""
    global predictor, privacy_manager, metrics
    
    logger.info("="*70)
    logger.info("🏥 Starting VitalViewAI ML Server")
    logger.info("="*70)
    
    try:
        # Load model
        logger.info("Loading ML model...")
        predictor = HealthPredictor(model_type="xgboost")
        logger.info(
            "Model loaded successfully",
            extra={
                'model_type': 'xgboost',
                'performance': predictor.metadata['performance']
            }
        )
        
        # Initialize privacy manager
        logger.info("Initializing privacy manager...")
        privacy_manager = PrivacyManager()
        logger.info("Privacy manager initialized")
        
        # Set start time
        metrics['start_time'] = datetime.now()
        
        logger.info("="*70)
        logger.info("✅ ML Server ready to accept requests")
        logger.info("="*70)
        
        # Log audit
        audit_logger.log_access(
            user_id='system',
            action='ml_server_startup',
            resource='ml_server',
            success=True,
            details={'model_type': 'xgboost'}
        )
        
        print("\n" + "="*70)
        print(" "*15 + "🏥 VITALVIEWAI ML SERVER")
        print("="*70)
        print("\n✅ Server started with enhanced logging")
        print("\n📊 Logging Features:")
        print("   ✓ Structured JSON logging")
        print("   ✓ Request correlation IDs")
        print("   ✓ Performance metrics tracking")
        print("   ✓ Comprehensive audit trail")
        print("   ✓ Error tracking")
        print("\n📡 Endpoints: http://localhost:8001/docs")
        print("📝 Logs: logs/ directory")
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to start ML server: {str(e)}", exc_info=True)
        error_logger.log_error(e, context={'operation': 'startup'})
        
        audit_logger.log_access(
            user_id='system',
            action='ml_server_startup',
            resource='ml_server',
            success=False,
            details={'error': str(e)}
        )
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    uptime = (datetime.now() - metrics['start_time']).total_seconds() if metrics['start_time'] else 0
    
    logger.info(
        "Shutting down ML server",
        extra={
            'total_predictions': metrics['total_predictions'],
            'total_alerts': metrics['total_alerts'],
            'uptime_seconds': uptime,
            'avg_inference_time_ms': metrics['avg_inference_time_ms']
        }
    )
    
    # Log audit
    audit_logger.log_access(
        user_id='system',
        action='ml_server_shutdown',
        resource='ml_server',
        success=True,
        details={
            'total_predictions': metrics['total_predictions'],
            'uptime_seconds': uptime
        }
    )
    
    print("\n🛑 ML server shut down gracefully")

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint"""
    logger.debug("Root endpoint accessed")
    
    return {
        "service": "VitalViewAI ML Server",
        "version": "1.0.0",
        "status": "running",
        "model_type": "xgboost",
        "endpoints": {
            "predict": "/predict",
            "batch": "/predict/batch",
            "health": "/health",
            "metrics": "/metrics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed status"""
    logger.debug("Health check requested")
    
    try:
        if predictor is None:
            logger.error("Health check failed: Model not loaded")
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        uptime = (datetime.now() - metrics['start_time']).total_seconds() if metrics['start_time'] else 0
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "model_loaded": True,
            "total_predictions": metrics['total_predictions'],
            "total_alerts": metrics['total_alerts'],
            "avg_inference_time_ms": metrics['avg_inference_time_ms'],
            "error_count": metrics['errors']
        }
        
        logger.info("Health check completed", extra=health_status)
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    logger.debug("Metrics endpoint accessed")
    
    if metrics['start_time']:
        metrics['uptime_seconds'] = (datetime.now() - metrics['start_time']).total_seconds()
    
    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "total_predictions": metrics['total_predictions'],
        "total_alerts": metrics['total_alerts'],
        "avg_inference_time_ms": metrics['avg_inference_time_ms'],
        "uptime_seconds": metrics['uptime_seconds'],
        "alert_rate": metrics['total_alerts'] / max(metrics['total_predictions'], 1),
        "error_count": metrics['errors'],
        "error_rate": metrics['errors'] / max(metrics['total_predictions'], 1)
    }
    
    logger.info("Metrics retrieved", extra=metrics_data)
    
    return metrics_data

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, http_request: Request):
    """Make single prediction with comprehensive logging"""
    correlation_id = getattr(http_request.state, 'correlation_id', str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(
        f"Prediction request received",
        extra={
            'correlation_id': correlation_id,
            'patient_id': request.patient_id,
            'user_id': request.user_id,
            'user_role': request.user_role,
            'vitals_count': len(request.vitals)
        }
    )
    
    try:
        # Validate access permissions
        if not privacy_manager.check_permission(request.user_role, 'view_predictions'):
            logger.warning(
                f"Permission denied",
                extra={
                    'correlation_id': correlation_id,
                    'user_id': request.user_id,
                    'user_role': request.user_role
                }
            )
            
            audit_logger.log_access(
                user_id=request.user_id,
                action='predict',
                resource=request.patient_id,
                success=False,
                details={'reason': 'permission_denied'}
            )
            
            raise HTTPException(status_code=403, detail="Permission denied")
        
        # Convert to DataFrame
        df = pd.DataFrame(request.vitals)
        
        logger.debug(
            f"Data prepared for prediction",
            extra={
                'correlation_id': correlation_id,
                'rows': len(df),
                'columns': len(df.columns)
            }
        )
        
        # Validate data integrity
        if not privacy_manager.validate_data_integrity(df.iloc[0].to_dict()):
            logger.warning(
                f"Invalid vital signs data",
                extra={'correlation_id': correlation_id}
            )
            raise HTTPException(status_code=400, detail="Invalid vital signs data")
        
        # Make prediction
        result = predictor.predict(
            df,
            patient_id=request.patient_id,
            user_id=request.user_id,
            correlation_id=correlation_id
        )
        
        # Calculate inference time
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Update metrics
        metrics['total_predictions'] += 1
        if result['alert']:
            metrics['total_alerts'] += 1
        
        # Update average inference time
        metrics['avg_inference_time_ms'] = (
            (metrics['avg_inference_time_ms'] * (metrics['total_predictions'] - 1) + inference_time_ms)
            / metrics['total_predictions']
        )
        
        # Log performance
        perf_logger.log_prediction(
            patient_id=request.patient_id,
            inference_time_ms=inference_time_ms,
            risk_score=result['risk_score'],
            alert=result['alert']
        )
        
        # Log audit
        audit_logger.log_access(
            user_id=request.user_id,
            action='predict',
            resource=request.patient_id,
            success=True,
            details={
                'risk_score': result['risk_score'],
                'risk_level': result['risk_level'],
                'alert': result['alert']
            }
        )
        
        logger.info(
            f"Prediction completed",
            extra={
                'correlation_id': correlation_id,
                'patient_id': request.patient_id,
                'risk_score': result['risk_score'],
                'risk_level': result['risk_level'],
                'alert': result['alert'],
                'inference_time_ms': inference_time_ms
            }
        )
        
        # Log alert if triggered
        if result['alert']:
            logger.warning(
                f"ALERT: High risk prediction",
                extra={
                    'correlation_id': correlation_id,
                    'patient_id': request.patient_id,
                    'risk_score': result['risk_score'],
                    'risk_level': result['risk_level']
                }
            )
        
        return PredictionResponse(
            patient_id=request.patient_id,
            risk_score=result['risk_score'],
            risk_level=result['risk_level'],
            alert=result['alert'],
            interpretation=result['interpretation'],
            timestamp=result['timestamp'],
            inference_time_ms=inference_time_ms,
            correlation_id=correlation_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        inference_time_ms = (time.time() - start_time) * 1000
        
        logger.error(
            f"Prediction failed: {str(e)}",
            exc_info=True,
            extra={
                'correlation_id': correlation_id,
                'patient_id': request.patient_id,
                'user_id': request.user_id,
                'elapsed_time_ms': inference_time_ms
            }
        )
        
        error_logger.log_error(e, context={
            'operation': 'predict',
            'correlation_id': correlation_id,
            'patient_id': request.patient_id
        })
        
        # Log failed audit
        audit_logger.log_access(
            user_id=request.user_id,
            action='predict',
            resource=request.patient_id,
            success=False,
            details={'error': str(e)}
        )
        
        metrics['errors'] += 1
        
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest, http_request: Request):
    """Make batch predictions with logging"""
    correlation_id = getattr(http_request.state, 'correlation_id', str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(
        f"Batch prediction request received",
        extra={
            'correlation_id': correlation_id,
            'batch_size': len(request.requests)
        }
    )
    
    try:
        results = []
        
        for patient_request in request.requests:
            try:
                result = await predict(patient_request, http_request)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Batch prediction item failed: {str(e)}",
                    extra={
                        'correlation_id': correlation_id,
                        'patient_id': patient_request.patient_id
                    }
                )
        
        batch_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Batch prediction completed",
            extra={
                'correlation_id': correlation_id,
                'batch_size': len(request.requests),
                'successful': len(results),
                'failed': len(request.requests) - len(results),
                'total_time_ms': batch_time,
                'avg_time_per_patient_ms': batch_time / len(results) if results else 0
            }
        )
        
        return {
            "results": results,
            "total_patients": len(request.requests),
            "successful": len(results),
            "failed": len(request.requests) - len(results),
            "total_time_ms": batch_time,
            "avg_time_per_patient_ms": batch_time / len(results) if results else 0,
            "correlation_id": correlation_id
        }
        
    except Exception as e:
        logger.error(
            f"Batch prediction failed: {str(e)}",
            exc_info=True,
            extra={'correlation_id': correlation_id}
        )
        
        error_logger.log_error(e, context={
            'operation': 'predict_batch',
            'correlation_id': correlation_id
        })
        
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/alert")
async def trigger_alert(
    patient_id: str,
    risk_score: float,
    user_id: str = "system",
    http_request: Request = None
):
    """Trigger clinical alert with logging"""
    correlation_id = getattr(http_request.state, 'correlation_id', str(uuid.uuid4())) if http_request else str(uuid.uuid4())
    
    logger.warning(
        f"ALERT TRIGGERED",
        extra={
            'correlation_id': correlation_id,
            'patient_id': patient_id,
            'risk_score': risk_score,
            'notified_user': user_id
        }
    )
    
    try:
        # Log audit
        audit_logger.log_access(
            user_id='system',
            action='trigger_alert',
            resource=patient_id,
            success=True,
            details={
                'risk_score': risk_score,
                'notified_user': user_id,
                'correlation_id': correlation_id
            }
        )
        
        # In production: Send notification (email, SMS, pager)
        
        return {
            "status": "alert_triggered",
            "patient_id": patient_id,
            "risk_score": risk_score,
            "timestamp": datetime.now().isoformat(),
            "notified": [user_id],
            "correlation_id": correlation_id
        }
        
    except Exception as e:
        logger.error(
            f"Alert trigger failed: {str(e)}",
            exc_info=True,
            extra={'correlation_id': correlation_id}
        )
        
        error_logger.log_error(e, context={
            'operation': 'trigger_alert',
            'patient_id': patient_id
        })
        
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting ML server")
    
    print("🚀 Starting VitalViewAI ML Server with Enhanced Logging...")
    print("📝 Press Ctrl+C to stop")
    
    # Run server
    uvicorn.run(
        "ml_server:app",
        port=8001,
        workers=1,  # Use 1 for development, increase for production
        log_level="info"
    )