# streaming_api_server.py
"""
Health Monitoring Streaming API with Comprehensive Logging
REST API with WebSocket support that mimics real wearable APIs

Enhanced Features:
- Structured JSON logging
- Request/response logging with correlation IDs
- Performance metrics tracking
- Audit trail for all operations
- Error tracking and alerting
"""

from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import asyncio
import sys
import os
import uuid
import time

# Add src to path
sys.path.append('src')
from data.wearable_simulator import WearableDevice

# Import logging system
from logging_config import (
    setup_logging, get_logger, 
    PerformanceLogger, AuditLogger
)

# Initialize logging
setup_logging(log_level="INFO", enable_json=True)
logger = get_logger(__name__)
perf_logger = PerformanceLogger()
audit_logger = AuditLogger()

# Initialize FastAPI
app = FastAPI(
    title="Health Monitoring Streaming API",
    description="Real-time wearable data API with comprehensive logging",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active patient devices
active_devices = {}

# Performance metrics
api_metrics = {
    'total_requests': 0,
    'total_errors': 0,
    'avg_response_time_ms': 0.0
}
# ==================== MIDDLEWARE ====================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with correlation ID"""
    # Generate correlation ID
    correlation_id = str(uuid.uuid4())
    request.state.correlation_id = correlation_id
    
    start_time = time.time()
    
    logger.info(
        f"Incoming request",
        extra={
            'correlation_id': correlation_id,
            'method': request.method,
            'path': request.url.path,
            'client_ip': request.client.host if request.client else 'unknown'
        }
    )
    
    try:
        response = await call_next(request)
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Log response
        logger.info(
            f"Request completed",
            extra={
                'correlation_id': correlation_id,
                'method': request.method,
                'path': request.url.path,
                'status_code': response.status_code,
                'response_time_ms': response_time_ms
            }
        )
        
        # Log performance metrics
        perf_logger.log_api_request(
            endpoint=request.url.path,
            method=request.method,
            response_time_ms=response_time_ms,
            status_code=response.status_code
        )
        
        # Update metrics
        api_metrics['total_requests'] += 1
        api_metrics['avg_response_time_ms'] = (
            (api_metrics['avg_response_time_ms'] * (api_metrics['total_requests'] - 1) + response_time_ms)
            / api_metrics['total_requests']
        )
        
        return response
        
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        
        logger.error(
            f"Request failed: {str(e)}",
            exc_info=True,
            extra={
                'correlation_id': correlation_id,
                'method': request.method,
                'path': request.url.path,
                'response_time_ms': response_time_ms
            }
        )
        
        api_metrics['total_errors'] += 1
        raise

# ==================== REQUEST/RESPONSE MODELS ====================

class PatientCreate(BaseModel):
    """Request to start monitoring a patient"""
    patient_id: str
    sampling_interval_seconds: int = 300
    health_profile: Optional[str] = None

class EventTrigger(BaseModel):
    """Request to trigger deterioration event"""
    patient_id: str
    event_type: str = "hypertensive_crisis"

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """API information endpoint"""
    logger.debug("Root endpoint accessed")
    
    return {
        "name": "Health Monitoring Streaming API",
        "version": "1.0.0",
        "status": "operational",
        "description": "Real-time wearable health data API with comprehensive logging",
        "endpoints": {
            "health": "GET /health",
            "create_patient": "POST /patients",
            "get_current": "GET /patients/{patient_id}/current",
            "get_history": "GET /patients/{patient_id}/history",
            "stream_websocket": "WS /stream/{patient_id}",
            "trigger_event": "POST /patients/{patient_id}/trigger-event",
            "metrics": "GET /api/metrics"
        },
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed status"""
    logger.debug("Health check requested")
    
    uptime_seconds = (datetime.now() - app.state.start_time).total_seconds() if hasattr(app.state, 'start_time') else 0
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_patients": len(active_devices),
        "uptime_seconds": uptime_seconds,
        "total_requests": api_metrics['total_requests'],
        "total_errors": api_metrics['total_errors'],
        "error_rate": api_metrics['total_errors'] / max(api_metrics['total_requests'], 1)
    }
    
    logger.info("Health check completed", extra=health_status)
    
    return health_status

@app.get("/api/metrics")
async def get_api_metrics():
    """Get API performance metrics"""
    logger.debug("Metrics endpoint accessed")
    
    return {
        "timestamp": datetime.now().isoformat(),
        "metrics": api_metrics,
        "active_devices": len(active_devices)
    }

@app.post("/patients")
async def create_patient_monitoring(patient: PatientCreate, request: Request):
    """Start monitoring for a patient"""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    logger.info(
        f"Creating patient monitoring",
        extra={
            'correlation_id': correlation_id,
            'patient_id': patient.patient_id,
            'sampling_interval': patient.sampling_interval_seconds,
            'health_profile': patient.health_profile
        }
    )
    
    if patient.patient_id in active_devices:
        logger.warning(
            f"Patient already being monitored",
            extra={
                'correlation_id': correlation_id,
                'patient_id': patient.patient_id
            }
        )
        raise HTTPException(
            status_code=400,
            detail=f"Patient {patient.patient_id} is already being monitored"
        )
    
    try:
        # Create device
        device = WearableDevice(
            patient_id=patient.patient_id,
            health_profile=patient.health_profile  # Pass health profile
            )
        active_devices[patient.patient_id] = {
            'device': device,
            'sampling_interval': patient.sampling_interval_seconds,
            'created_at': datetime.now(),
            'health_profile': patient.health_profile
        }
        
        # Log audit event
        audit_logger.log_access(
            user_id='system',
            action='create_patient_monitoring',
            resource=patient.patient_id,
            success=True,
            details={
                'sampling_interval': patient.sampling_interval_seconds,
                'baseline_vitals': device.baseline,
                'health_profile': device.health_profile
            }
        )
        
        logger.info(
            f"Patient monitoring created successfully",
            extra={
                'correlation_id': correlation_id,
                'patient_id': patient.patient_id,
                'health_profile': device.health_profile,
                'baseline_hr': device.baseline['heart_rate'],
                'baseline_bp': device.baseline['bp_systolic']
            }
        )
        
        return {
            "status": "success",
            "message": f"Started monitoring patient {patient.patient_id}",
            "patient_id": patient.patient_id,
            "sampling_interval": patient.sampling_interval_seconds,
            "health_profile": device.health_profile,  
            "baseline_vitals": device.baseline,
            "correlation_id": correlation_id
        }
        
    except Exception as e:
        logger.error(
            f"Failed to create patient monitoring: {str(e)}",
            exc_info=True,
            extra={
                'correlation_id': correlation_id,
                'patient_id': patient.patient_id
            }
        )
        
        audit_logger.log_access(
            user_id='system',
            action='create_patient_monitoring',
            resource=patient.patient_id,
            success=False,
            details={'error': str(e)}
        )
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients/{patient_id}/current")
async def get_current_reading(patient_id: str, request: Request):
    """Get current vital signs reading"""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    logger.debug(
        f"Fetching current reading",
        extra={
            'correlation_id': correlation_id,
            'patient_id': patient_id
        }
    )
    
    if patient_id not in active_devices:
        logger.warning(
            f"Patient not found",
            extra={
                'correlation_id': correlation_id,
                'patient_id': patient_id
            }
        )
        raise HTTPException(status_code=404, detail="Patient not found")
    
    try:
        device = active_devices[patient_id]['device']
        reading = device.generate_reading()
        
        logger.info(
            f"Current reading retrieved",
            extra={
                'correlation_id': correlation_id,
                'patient_id': patient_id,
                'heart_rate': reading['heart_rate'],
                'bp_systolic': reading['bp_systolic'],
                'is_deteriorating': device.is_deteriorating
            }
        )
        
        # Log audit
        audit_logger.log_access(
            user_id='api_user',
            action='read_current_vitals',
            resource=patient_id,
            success=True
        )
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": reading,
            "correlation_id": correlation_id
        }
        
    except Exception as e:
        logger.error(
            f"Failed to get current reading: {str(e)}",
            exc_info=True,
            extra={
                'correlation_id': correlation_id,
                'patient_id': patient_id
            }
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients/{patient_id}/history")
async def get_historical_data(
    patient_id: str,
    request: Request,
    hours: int = 24,
    interval_minutes: int = 5
):
    """Get historical vital signs data"""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    logger.info(
        f"Fetching historical data",
        extra={
            'correlation_id': correlation_id,
            'patient_id': patient_id,
            'hours': hours,
            'interval_minutes': interval_minutes
        }
    )
    
    if patient_id not in active_devices:
        logger.warning(f"Patient not found: {patient_id}")
        raise HTTPException(status_code=404, detail="Patient not found")
    
    try:
        start_time = time.time()
        device = active_devices[patient_id]['device']
        
        # Generate historical readings
        end_time = datetime.now()
        start_time_dt = end_time - timedelta(hours=hours)
        
        readings = []
        current_time = start_time_dt
        
        while current_time <= end_time:
            reading = device.generate_reading(current_time)
            readings.append(reading)
            current_time += timedelta(minutes=interval_minutes)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Historical data retrieved",
            extra={
                'correlation_id': correlation_id,
                'patient_id': patient_id,
                'records_count': len(readings),
                'processing_time_ms': elapsed_ms
            }
        )
        
        # Log audit
        audit_logger.log_access(
            user_id='api_user',
            action='read_historical_data',
            resource=patient_id,
            success=True,
            details={
                'hours': hours,
                'records_count': len(readings)
            }
        )
        
        return {
            "status": "success",
            "patient_id": patient_id,
            "start_time": start_time_dt.isoformat(),
            "end_time": end_time.isoformat(),
            "count": len(readings),
            "interval_minutes": interval_minutes,
            "data": readings,
            "correlation_id": correlation_id
        }
        
    except Exception as e:
        logger.error(
            f"Failed to get historical data: {str(e)}",
            exc_info=True,
            extra={'correlation_id': correlation_id, 'patient_id': patient_id}
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/patients/{patient_id}/trigger-event")
async def trigger_deterioration_event(
    patient_id: str,
    event: EventTrigger,
    request: Request
):
    """Trigger a health deterioration event"""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    logger.warning(
        f"Triggering deterioration event",
        extra={
            'correlation_id': correlation_id,
            'patient_id': patient_id,
            'event_type': event.event_type
        }
    )
    
    if patient_id not in active_devices:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    try:
        device = active_devices[patient_id]['device']
        device.trigger_deterioration(event.event_type)
        
        logger.warning(
            f"Deterioration event triggered",
            extra={
                'correlation_id': correlation_id,
                'patient_id': patient_id,
                'event_type': event.event_type
            }
        )
        
        # Log security event
        audit_logger.log_security_incident(
            incident_type='test_deterioration_triggered',
            severity='medium',
            description=f"Test deterioration event triggered for patient {patient_id}",
            affected_resources=[patient_id]
        )
        
        return {
            "status": "success",
            "message": f"Triggered {event.event_type} for patient {patient_id}",
            "patient_id": patient_id,
            "event_type": event.event_type,
            "correlation_id": correlation_id
        }
        
    except Exception as e:
        logger.error(
            f"Failed to trigger event: {str(e)}",
            exc_info=True,
            extra={'correlation_id': correlation_id}
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/patients/{patient_id}/resolve-event")
async def resolve_deterioration(patient_id: str, request: Request):
    """Resolve deterioration event"""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    logger.info(
        f"Resolving deterioration",
        extra={'correlation_id': correlation_id, 'patient_id': patient_id}
    )
    
    if patient_id not in active_devices:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    device = active_devices[patient_id]['device']
    device.resolve_deterioration()
    
    logger.info(f"Deterioration resolved for patient {patient_id}")
    
    return {
        "status": "success",
        "message": f"Resolved deterioration for patient {patient_id}",
        "correlation_id": correlation_id
    }

@app.websocket("/stream/{patient_id}")
async def websocket_stream(websocket: WebSocket, patient_id: str):
    """WebSocket endpoint for real-time streaming"""
    session_id = str(uuid.uuid4())
    
    logger.info(
        f"WebSocket connection requested",
        extra={'session_id': session_id, 'patient_id': patient_id}
    )
    
    await websocket.accept()
    
    # Create device if doesn't exist
    if patient_id not in active_devices:
        device = WearableDevice(patient_id=patient_id)
        active_devices[patient_id] = {
            'device': device,
            'sampling_interval': 60,
            'created_at': datetime.now()
        }
        logger.info(f"Created device for WebSocket: {patient_id}")
    
    device_info = active_devices[patient_id]
    device = device_info['device']
    interval = device_info['sampling_interval']
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "patient_id": patient_id,
            "session_id": session_id,
            "message": f"Streaming data every {interval} seconds"
        })
        
        logger.info(
            f"WebSocket connected",
            extra={'session_id': session_id, 'patient_id': patient_id}
        )
        
        # Stream data continuously
        while True:
            reading = device.generate_reading()
            
            await websocket.send_json({
                "type": "reading",
                "data": reading,
                "session_id": session_id
            })
            
            logger.debug(
                f"WebSocket data sent",
                extra={
                    'session_id': session_id,
                    'patient_id': patient_id,
                    'heart_rate': reading['heart_rate']
                }
            )
            
            await asyncio.sleep(interval)
            
    except WebSocketDisconnect:
        logger.info(
            f"WebSocket disconnected",
            extra={'session_id': session_id, 'patient_id': patient_id}
        )
    except Exception as e:
        logger.error(
            f"WebSocket error: {str(e)}",
            exc_info=True,
            extra={'session_id': session_id, 'patient_id': patient_id}
        )
        await websocket.close()

@app.get("/patients")
async def list_patients():
    """List all active patient monitors"""
    logger.debug(f"Listing patients, count: {len(active_devices)}")
    
    patients = []
    for patient_id, info in active_devices.items():
        device = info['device']
        patients.append({
            "patient_id": patient_id,
            "created_at": info['created_at'].isoformat(),
            "sampling_interval": info['sampling_interval'],
            "is_deteriorating": device.is_deteriorating,
            "deterioration_type": device.deterioration_type
        })
    
    return {
        "status": "success",
        "count": len(patients),
        "patients": patients
    }

@app.delete("/patients/{patient_id}")
async def stop_monitoring(patient_id: str, request: Request):
    """Stop monitoring a patient"""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    logger.info(
        f"Stopping monitoring",
        extra={'correlation_id': correlation_id, 'patient_id': patient_id}
    )
    
    if patient_id not in active_devices:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    del active_devices[patient_id]
    
    audit_logger.log_access(
        user_id='system',
        action='stop_monitoring',
        resource=patient_id,
        success=True
    )
    
    logger.info(f"Monitoring stopped for patient {patient_id}")
    
    return {
        "status": "success",
        "message": f"Stopped monitoring patient {patient_id}",
        "correlation_id": correlation_id
    }

# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    app.state.start_time = datetime.now()
    
    logger.info("="*70)
    logger.info("🏥 Health Monitoring Streaming API Starting")
    logger.info("="*70)
    logger.info("API Server initialized successfully")
    logger.info(f"Logging enabled: JSON format")
    logger.info(f"Log files: logs/application.log, logs/errors.log, logs/audit.log")
    logger.info("="*70)
    
    print("\n" + "="*70)
    print(" "*15 + "🏥 HEALTH MONITORING STREAMING API")
    print("="*70)
    print("\n✅ API Server Started with Enhanced Logging")
    print("\n📊 Logging Features:")
    print("   ✓ Structured JSON logging")
    print("   ✓ Request correlation IDs")
    print("   ✓ Performance metrics tracking")
    print("   ✓ Comprehensive audit trail")
    print("   ✓ Error tracking and alerting")
    print("\n📡 Endpoints: http://localhost:8000/docs")
    print("📝 Logs: logs/ directory")
    print("\n" + "="*70 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API server")
    logger.info(f"Total requests served: {api_metrics['total_requests']}")
    logger.info(f"Average response time: {api_metrics['avg_response_time_ms']:.2f}ms")
    
    active_devices.clear()
    
    print("\n🛑 API server shut down gracefully")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Health Monitoring API with Enhanced Logging...")
    print("📝 Press Ctrl+C to stop")
    
    uvicorn.run(
        app,
        port=8000,
        log_level="info"
    )