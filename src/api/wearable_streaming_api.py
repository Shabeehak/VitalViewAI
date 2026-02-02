#src/api/wearable_streaming_api.py
"""
Wearable Device Streaming API
REST API that mimics real wearable device APIs (like Fitbit, Apple HealthKit)

This demonstrates:
- API design for health data
- WebSocket streaming
- Real-time data handling
- Production-ready endpoints

Interview Value:
"I built a REST API that simulates real wearable device streams, with both
HTTP endpoints for batch data and WebSocket connections for real-time streaming"
"""

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import json
import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.realtime_health_simulator import WearableDeviceSimulator, RealTimeDataStream

# Initialize FastAPI app
app = FastAPI(
    title="Wearable Health Monitoring API",
    description="Real-time health data streaming API for ML monitoring system",
    version="1.0.0"
)

# Enable CORS (for web frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active patient devices
active_devices = {}

# ==================== DATA MODELS ====================

class PatientCreate(BaseModel):
    """Request model for creating new patient monitoring"""
    patient_id: str
    sampling_rate_seconds: int = 300  # Default: 5 minutes

class VitalReading(BaseModel):
    """Single vital signs reading"""
    patient_id: str
    timestamp: str
    heart_rate: float
    bp_systolic: float
    bp_diastolic: float
    spo2: float
    respiratory_rate: float
    temperature: float
    activity_state: str

class HistoricalDataRequest(BaseModel):
    """Request for historical data"""
    patient_id: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    hours_back: int = 24

class DeteriorationTrigger(BaseModel):
    """Trigger deterioration event (for testing)"""
    patient_id: str
    deterioration_type: str = "hypertensive_crisis"

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """API info endpoint"""
    return {
        "name": "Wearable Health Monitoring API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health_check": "/health",
            "create_patient": "POST /patients",
            "get_current_reading": "GET /patients/{patient_id}/current",
            "get_historical": "POST /patients/{patient_id}/historical",
            "stream_realtime": "WS /stream/{patient_id}",
            "trigger_event": "POST /patients/{patient_id}/trigger-deterioration"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_patients": len(active_devices)
    }

@app.post("/patients")
async def create_patient_monitoring(patient: PatientCreate):
    """
    Start monitoring for a new patient
    
    This creates a virtual wearable device that generates data
    """
    if patient.patient_id in active_devices:
        raise HTTPException(status_code=400, detail="Patient already being monitored")
    
    # Create device simulator
    device = WearableDeviceSimulator(
        patient_id=patient.patient_id,
        sampling_rate_seconds=patient.sampling_rate_seconds
    )
    
    active_devices[patient.patient_id] = device
    
    return {
        "status": "success",
        "message": f"Started monitoring for patient {patient.patient_id}",
        "patient_id": patient.patient_id,
        "sampling_rate": patient.sampling_rate_seconds,
        "baseline_vitals": device.baseline
    }

@app.get("/patients/{patient_id}/current")
async def get_current_reading(patient_id: str):
    """
    Get current vital signs reading
    
    This mimics: GET /v1/user/-/activities/heart/date/today/1d.json (Fitbit)
    """
    if patient_id not in active_devices:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    device = active_devices[patient_id]
    reading = device.generate_reading()
    
    return {
        "status": "success",
        "data": reading
    }

@app.post("/patients/{patient_id}/historical")
async def get_historical_data(patient_id: str, request: HistoricalDataRequest):
    """
    Get historical vital signs data
    
    This mimics batch data retrieval from wearable APIs
    """
    if patient_id not in active_devices:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    device = active_devices[patient_id]
    
    # Generate historical data
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=request.hours_back)
    
    # Generate readings every sampling_rate
    readings = []
    current_time = start_time
    
    # Reset device state for historical generation to ensure variation
    device.last_values = device.baseline.copy()
    
    while current_time <= end_time:
        reading = device.generate_reading(current_time)
        readings.append(reading)
        current_time += timedelta(seconds=device.sampling_rate)
    
    return {
        "status": "success",
        "patient_id": patient_id,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "count": len(readings),
        "data": readings
    }

@app.post("/patients/{patient_id}/trigger-deterioration")
async def trigger_deterioration(patient_id: str, trigger: DeteriorationTrigger):
    """
    Trigger a health deterioration event (for testing ML alerts)
    
    Types: hypertensive_crisis, hypoxia, sepsis, cardiac
    """
    if patient_id not in active_devices:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    device = active_devices[patient_id]
    device.trigger_deterioration(trigger.deterioration_type)
    
    return {
        "status": "success",
        "message": f"Triggered {trigger.deterioration_type} for patient {patient_id}",
        "patient_id": patient_id,
        "deterioration_type": trigger.deterioration_type
    }

@app.post("/patients/{patient_id}/stop-deterioration")
async def stop_deterioration(patient_id: str):
    """Stop deterioration event (patient treated)"""
    if patient_id not in active_devices:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    device = active_devices[patient_id]
    device.stop_deterioration()
    
    return {
        "status": "success",
        "message": f"Stopped deterioration for patient {patient_id}"
    }

@app.websocket("/stream/{patient_id}")
async def websocket_stream(websocket: WebSocket, patient_id: str):
    """
    WebSocket endpoint for real-time data streaming
    
    This mimics real-time streaming from wearable devices
    
    Interview Talking Point:
    "I implemented WebSocket streaming to handle real-time health data,
    allowing the ML pipeline to receive and process data continuously"
    
    Usage (JavaScript):
        const ws = new WebSocket('ws://localhost:8000/stream/patient_001');
        ws.onmessage = (event) => {
            const reading = JSON.parse(event.data);
            console.log('New reading:', reading);
        };
    """
    await websocket.accept()
    
    # Create or get device
    if patient_id not in active_devices:
        device = WearableDeviceSimulator(patient_id=patient_id, sampling_rate_seconds=60)
        active_devices[patient_id] = device
    else:
        device = active_devices[patient_id]
    
    try:
        # Send initial message
        await websocket.send_json({
            "type": "connection",
            "message": f"Connected to patient {patient_id} stream",
            "patient_id": patient_id
        })
        
        # Stream data continuously
        while True:
            # Generate reading
            reading = device.generate_reading()
            
            # Send to client
            await websocket.send_json({
                "type": "reading",
                "data": reading
            })
            
            # Wait for next sampling interval
            await asyncio.sleep(device.sampling_rate)
            
    except Exception as e:
        print(f"WebSocket error for patient {patient_id}: {e}")
    finally:
        await websocket.close()

@app.get("/patients")
async def list_patients():
    """List all active patient monitors"""
    patients = []
    for patient_id, device in active_devices.items():
        patients.append({
            "patient_id": patient_id,
            "sampling_rate": device.sampling_rate,
            "is_deteriorating": device.is_deteriorating,
            "current_activity": device.activity_state
        })
    
    return {
        "status": "success",
        "count": len(patients),
        "patients": patients
    }

@app.delete("/patients/{patient_id}")
async def stop_monitoring(patient_id: str):
    """Stop monitoring a patient"""
    if patient_id not in active_devices:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    del active_devices[patient_id]
    
    return {
        "status": "success",
        "message": f"Stopped monitoring for patient {patient_id}"
    }

# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("\n" + "="*60)
    print("🏥 WEARABLE HEALTH MONITORING API")
    print("="*60)
    print("\n✅ API started successfully")
    print("\n📡 Endpoints:")
    print("   GET  /           - API info")
    print("   GET  /health     - Health check")
    print("   POST /patients   - Create patient monitoring")
    print("   GET  /patients/{id}/current - Get current reading")
    print("   POST /patients/{id}/historical - Get historical data")
    print("   WS   /stream/{id} - Real-time WebSocket stream")
    print("\n🌐 Documentation: http://localhost:8000/docs")
    print("\n" + "="*60 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n🛑 Shutting down API...")
    active_devices.clear()

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Wearable Health Monitoring API...")
    print("Access at: http://localhost:8000")
    print("Docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        port=8000,
        log_level="info"
    )