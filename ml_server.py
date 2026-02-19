"""
Production ML Model Serving API with Comprehensive Logging and Authentication
High-performance prediction endpoint for real-time inference

Enhanced Features:
- JWT-based authentication and authorization
- Role-based access control (RBAC)
- Structured logging with correlation IDs
- Performance metrics tracking
- Comprehensive audit trail
- Error tracking and alerting
- Request/response logging

Usage:
    python ml_server.py
"""

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from auth_system import auth_manager, AuthenticationManager
from privacy_utils import privacy_manager
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

# OAuth2 scheme for token-based auth
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Initialize FastAPI
app = FastAPI(
    title="VitalViewAI ML Server",
    description="Production ML model serving with authentication and comprehensive logging",
    version="1.0.0"
)

# Global instances
predictor = None

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
    """Single prediction request (user info comes from auth token)"""
    patient_id: str
    vitals: List[Dict[str, Any]]

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

# ==================== AUTHENTICATION MODELS ====================

class Token(BaseModel):
    """Token response model"""
    access_token: str
    token_type: str
    user: dict

class UserCreate(BaseModel):
    """User registration model"""
    username: str
    password: str
    role: str
    email: str
    full_name: str

# ==================== AUTHENTICATION DEPENDENCIES ====================

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Dependency to get current authenticated user from token
    
    Usage:
        @app.get("/protected")
        async def protected_route(user = Depends(get_current_user)):
            return {"message": f"Hello {user['username']}"}
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = auth_manager.verify_token(token)
    
    if payload is None:
        raise credentials_exception
    
    username = payload.get("sub")
    role = payload.get("role")
    
    if username is None or role is None:
        raise credentials_exception
    
    return {
        "username": username,
        "role": role
    }

async def require_permission(permission: str):
    """
    Dependency factory to check specific permission
    
    Usage:
        @app.get("/admin-only")
        async def admin_route(
            user = Depends(get_current_user),
            _ = Depends(require_permission("modify_patient_records"))
        ):
            return {"message": "Admin access granted"}
    """
    async def permission_checker(user: dict = Depends(get_current_user)):
        if not privacy_manager.check_permission(user['role'], permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        return user
    
    return permission_checker


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
        print("\n✅ Server started with authentication and logging")
        print("\n🔐 Authentication Features:")
        print("   ✓ JWT token-based authentication")
        print("   ✓ Role-based access control (RBAC)")
        print("   ✓ User registration and login")
        print("\n📊 Logging Features:")
        print("   ✓ Structured JSON logging")
        print("   ✓ Request correlation IDs")
        print("   ✓ Performance metrics tracking")
        print("   ✓ Comprehensive audit trail")
        print("   ✓ Error tracking")
        print("\n📡 Endpoints: http://localhost:8001/docs")
        print("📝 Logs: logs/ directory")
        print("\n🔑 First time? Register at: POST /register")
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

# ==================== PUBLIC ENDPOINTS (No Auth Required) ====================

@app.get("/", tags=["Public"])
async def root():
    """Root endpoint - no authentication required"""
    logger.debug("Root endpoint accessed")
    
    return {
        "service": "VitalViewAI ML Server",
        "version": "1.0.0",
        "status": "running",
        "model_type": "xgboost",
        "authentication": "enabled",
        "endpoints": {
            "register": "POST /register",
            "login": "POST /login",
            "predict": "POST /predict (requires auth)",
            "batch": "POST /predict/batch (requires auth)",
            "health": "GET /health",
            "metrics": "GET /metrics",
            "docs": "GET /docs"
        }
    }

@app.get("/health", tags=["Public"])
async def health_check():
    """Health check endpoint with detailed status - no authentication required"""
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
            "authentication": "enabled",
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

# ==================== AUTHENTICATION ENDPOINTS ====================

@app.post("/register", response_model=Token, tags=["Authentication"])
async def register(user: UserCreate):
    """
    Register new user
    
    Example:
        POST /register
        {
            "username": "dr_jones",
            "password": "secure123",
            "role": "clinician",
            "email": "jones@hospital.com",
            "full_name": "Dr. Sarah Jones"
        }
    
    Valid roles: clinician, nurse, researcher, admin
    """
    logger.info(f"Registration attempt for username: {user.username}")
    
    # Validate role
    valid_roles = ["clinician", "nurse", "researcher", "admin"]
    if user.role not in valid_roles:
        logger.warning(f"Invalid role registration attempt: {user.role}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role. Must be one of: {valid_roles}"
        )
    
    # Create user
    success = auth_manager.register_user(
        username=user.username,
        password=user.password,
        role=user.role,
        email=user.email,
        full_name=user.full_name
    )
    
    if not success:
        logger.warning(f"Registration failed - username already exists: {user.username}")
        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )
    
    logger.info(f"User registered successfully: {user.username} (role: {user.role})")
    
    # Log audit
    audit_logger.log_access(
        user_id=user.username,
        action='register',
        resource='auth_system',
        success=True,
        details={'role': user.role, 'email': user.email}
    )
    
    # Auto-login after registration
    result = auth_manager.login(user.username, user.password)
    
    return Token(
        access_token=result['access_token'],
        token_type=result['token_type'],
        user=result['user']
    )

@app.post("/login", response_model=Token, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint
    
    Usage:
        POST /login
        Content-Type: application/x-www-form-urlencoded
        
        username=dr_smith&password=doctor123
    
    Returns:
        {
            "access_token": "eyJ...",
            "token_type": "bearer",
            "user": {
                "username": "dr_smith",
                "role": "clinician",
                ...
            }
        }
    """
    logger.info(f"Login attempt for username: {form_data.username}")
    
    result = auth_manager.login(form_data.username, form_data.password)
    
    if not result:
        logger.warning(f"Failed login attempt for username: {form_data.username}")
        
        # Log failed audit
        audit_logger.log_access(
            user_id=form_data.username,
            action='login',
            resource='auth_system',
            success=False,
            details={'reason': 'invalid_credentials'}
        )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    logger.info(f"Successful login for: {form_data.username} (role: {result['user']['role']})")
    
    # Log successful audit
    audit_logger.log_access(
        user_id=form_data.username,
        action='login',
        resource='auth_system',
        success=True,
        details={'role': result['user']['role']}
    )
    
    return Token(
        access_token=result['access_token'],
        token_type=result['token_type'],
        user=result['user']
    )

@app.get("/me", tags=["Authentication"])
async def get_current_user_info(user: dict = Depends(get_current_user)):
    """
    Get current user info
    
    Requires: Valid JWT token
    
    Usage:
        GET /me
        Authorization: Bearer <token>
    """
    logger.info(f"User info requested for: {user['username']}")
    
    return {
        "username": user['username'],
        "role": user['role'],
        "permissions": privacy_manager.get_user_permissions(user['role'])
    }

# ==================== PROTECTED ENDPOINTS (Auth Required) ====================

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics(user: dict = Depends(get_current_user)):
    """
    Get performance metrics
    
    Requires: Authentication
    """
    logger.debug(f"Metrics endpoint accessed by: {user['username']}")
    
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
    
    logger.info(f"Metrics retrieved by {user['username']}", extra=metrics_data)
    
    return metrics_data

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    request: PredictionRequest,
    http_request: Request,
    user: dict = Depends(get_current_user)
):
    """
    Make single prediction with comprehensive logging
    
    Requires: Authentication + 'view_predictions' permission
    
    Usage:
        POST /predict
        Authorization: Bearer <token>
        
        {
            "patient_id": "patient_001",
            "vitals": [
                {
                    "heart_rate": 75,
                    "blood_pressure_systolic": 120,
                    "blood_pressure_diastolic": 80,
                    "temperature": 98.6,
                    "respiratory_rate": 16,
                    "oxygen_saturation": 98
                }
            ]
        }
    """
    correlation_id = getattr(http_request.state, 'correlation_id', str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(
        f"Prediction request received",
        extra={
            'correlation_id': correlation_id,
            'patient_id': request.patient_id,
            'user_id': user['username'],
            'user_role': user['role'],
            'vitals_count': len(request.vitals)
        }
    )
    
    try:
        # Validate access permissions
        if not privacy_manager.check_permission(user['role'], 'view_predictions'):
            logger.warning(
                f"Permission denied",
                extra={
                    'correlation_id': correlation_id,
                    'user_id': user['username'],
                    'user_role': user['role'],
                    'required_permission': 'view_predictions'
                }
            )
            
            audit_logger.log_access(
                user_id=user['username'],
                action='predict',
                resource=request.patient_id,
                success=False,
                details={'reason': 'permission_denied'}
            )
            
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to view predictions"
            )
        
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
            user_id=user['username'],  # Use authenticated username
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
            user_id=user['username'],
            action='predict',
            resource=request.patient_id,
            success=True,
            details={
                'risk_score': result['risk_score'],
                'risk_level': result['risk_level'],
                'alert': result['alert'],
                'role': user['role']
            }
        )
        
        logger.info(
            f"Prediction completed",
            extra={
                'correlation_id': correlation_id,
                'patient_id': request.patient_id,
                'user_id': user['username'],
                'risk_score': result['risk_score'],
                'risk_level': result['risk_level'],
                'alert': result['alert'],
                'inference_time_ms': inference_time_ms
            }
        )
        
        # Log alert if triggered + fire alert system
        if result['alert']:
            logger.warning(
                f"ALERT: High risk prediction",
                extra={
                    'correlation_id': correlation_id,
                    'patient_id': request.patient_id,
                    'user_id': user['username'],
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
                'user_id': user['username'],
                'elapsed_time_ms': inference_time_ms
            }
        )
        
        error_logger.log_error(e, context={
            'operation': 'predict',
            'correlation_id': correlation_id,
            'patient_id': request.patient_id,
            'user_id': user['username']
        })
        
        # Log failed audit
        audit_logger.log_access(
            user_id=user['username'],
            action='predict',
            resource=request.patient_id,
            success=False,
            details={'error': str(e)}
        )
        
        metrics['errors'] += 1
        
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    http_request: Request,
    user: dict = Depends(get_current_user)
):
    """
    Make batch predictions with logging
    
    Requires: Authentication + 'view_predictions' permission
    
    Usage:
        POST /predict/batch
        Authorization: Bearer <token>
        
        {
            "requests": [
                {
                    "patient_id": "patient_001",
                    "vitals": [...]
                },
                {
                    "patient_id": "patient_002",
                    "vitals": [...]
                }
            ]
        }
    """
    correlation_id = getattr(http_request.state, 'correlation_id', str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(
        f"Batch prediction request received",
        extra={
            'correlation_id': correlation_id,
            'batch_size': len(request.requests),
            'user_id': user['username'],
            'user_role': user['role']
        }
    )
    
    # Check permission
    if not privacy_manager.check_permission(user['role'], 'view_predictions'):
        logger.warning(f"Batch prediction permission denied for {user['username']}")
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to view predictions"
        )
    
    try:
        results = []
        
        for patient_request in request.requests:
            try:
                # Create a mock Request object for each prediction
                result = await predict(patient_request, http_request, user)
                results.append(result)
            except HTTPException as e:
                logger.error(
                    f"Batch prediction item failed: {e.detail}",
                    extra={
                        'correlation_id': correlation_id,
                        'patient_id': patient_request.patient_id,
                        'status_code': e.status_code
                    }
                )
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
                'user_id': user['username'],
                'batch_size': len(request.requests),
                'successful': len(results),
                'failed': len(request.requests) - len(results),
                'total_time_ms': batch_time,
                'avg_time_per_patient_ms': batch_time / len(results) if results else 0
            }
        )
        
        # Log audit
        audit_logger.log_access(
            user_id=user['username'],
            action='predict_batch',
            resource='batch_predictions',
            success=True,
            details={
                'batch_size': len(request.requests),
                'successful': len(results),
                'failed': len(request.requests) - len(results)
            }
        )
        
        return {
            "results": results,
            "total_patients": len(request.requests),
            "successful": len(results),
            "failed": len(request.requests) - len(results),
            "total_time_ms": batch_time,
            "avg_time_per_patient_ms": batch_time / len(results) if results else 0,
            "correlation_id": correlation_id,
            "processed_by": user['username']
        }
        
    except Exception as e:
        logger.error(
            f"Batch prediction failed: {str(e)}",
            exc_info=True,
            extra={
                'correlation_id': correlation_id,
                'user_id': user['username']
            }
        )
        
        error_logger.log_error(e, context={
            'operation': 'predict_batch',
            'correlation_id': correlation_id,
            'user_id': user['username']
        })
        
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/alert", tags=["Alerts"])
async def trigger_alert(
    patient_id: str,
    risk_score: float,
    http_request: Request,
    user: dict = Depends(get_current_user)
):
    """
    Trigger clinical alert with logging
    
    Requires: Authentication + 'trigger_alerts' permission
    """
    correlation_id = getattr(http_request.state, 'correlation_id', str(uuid.uuid4()))
    
    # Check permission
    if not privacy_manager.check_permission(user['role'], 'trigger_alerts'):
        logger.warning(f"Alert trigger permission denied for {user['username']}")
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to trigger alerts"
        )
    
    logger.warning(
        f"ALERT TRIGGERED",
        extra={
            'correlation_id': correlation_id,
            'patient_id': patient_id,
            'risk_score': risk_score,
            'triggered_by': user['username']
        }
    )
    
    try:
        # Log audit
        audit_logger.log_access(
            user_id=user['username'],
            action='trigger_alert',
            resource=patient_id,
            success=True,
            details={
                'risk_score': risk_score,
                'correlation_id': correlation_id
            }
        )
        
        # In production: Send notification (email, SMS, pager)
        
        return {
            "status": "alert_triggered",
            "patient_id": patient_id,
            "risk_score": risk_score,
            "timestamp": datetime.now().isoformat(),
            "triggered_by": user['username'],
            "notified": [user['username']],
            "correlation_id": correlation_id
        }
        
    except Exception as e:
        logger.error(
            f"Alert trigger failed: {str(e)}",
            exc_info=True,
            extra={
                'correlation_id': correlation_id,
                'user_id': user['username']
            }
        )
        
        error_logger.log_error(e, context={
            'operation': 'trigger_alert',
            'patient_id': patient_id,
            'user_id': user['username']
        })
        
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ADMIN ENDPOINTS ====================

# @app.post("/admin/reset-metrics", tags=["Admin"])
# async def reset_metrics(user: dict = Depends(require_permission("admin_access"))):
#     """
#     Reset performance metrics (Admin only)
    
#     Requires: Admin role
#     """
#     logger.info(f"Metrics reset by admin: {user['username']}")
    
#     global metrics
#     old_metrics = metrics.copy()
    
#     metrics['total_predictions'] = 0
#     metrics['total_alerts'] = 0
#     metrics['avg_inference_time_ms'] = 0.0
#     metrics['errors'] = 0
    
#     # Log audit
#     audit_logger.log_access(
#         user_id=user['username'],
#         action='reset_metrics',
#         resource='metrics',
#         success=True,
#         details={'old_metrics': old_metrics}
#     )
    
#     return {
#         "message": "Metrics reset successfully",
#         "reset_by": user['username'],
#         "timestamp": datetime.now().isoformat()
#     }

# ==================== MAIN ====================

if __name__ == "__main__":
    logger.info("Starting ML server with authentication")
    
    print("🚀 Starting VitalViewAI ML Server with Authentication...")
    print("📝 Press Ctrl+C to stop")
    
    # Run server
    uvicorn.run(
        "ml_server:app",
        host="0.0.0.0",
        port=8001,
        workers=1,  # Use 1 for development, increase for production
        log_level="info"
    )