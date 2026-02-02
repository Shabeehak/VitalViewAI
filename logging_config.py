#logging_config.py
"""
Comprehensive Logging Configuration
Implements structured logging for production monitoring

Features:
- Structured JSON logging
- Log rotation
- Different log levels for different components
- Performance logging
- Error tracking
- Audit trail

Usage:
    from logging_config import setup_logging, get_logger
    
    setup_logging()
    logger = get_logger(__name__)
    logger.info("Application started", extra={'user_id': '123'})
"""

import logging
import logging.handlers
import json
import os
from datetime import datetime
from typing import Dict, Any
import traceback
import sys


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        # Add any additional attributes
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'lineno', 'module', 'msecs', 'message',
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                          'extra_data']:
                if not key.startswith('_'):
                    log_data[key] = value
        
        return json.dumps(log_data)


class ContextFilter(logging.Filter):
    """
    Add contextual information to log records
    """
    
    def __init__(self, context: Dict[str, Any] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to record"""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_console: bool = True,
    enable_file: bool = True,
    enable_json: bool = True
) -> None:
    """
    Setup comprehensive logging system
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_json: Use JSON format for file logs
    """
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console Handler (human-readable)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File Handler - General logs
    if enable_file:
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, 'application.log'),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=10
        )
        file_handler.setLevel(logging.DEBUG)
        
        if enable_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        root_logger.addHandler(file_handler)
    
    # Error Handler - Separate file for errors
    error_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'errors.log'),
        maxBytes=10 * 1024 * 1024,
        backupCount=10
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter() if enable_json else 
                               logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(error_handler)
    
    # Performance Handler - For performance metrics
    perf_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'performance.log'),
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(JSONFormatter() if enable_json else
                             logging.Formatter('%(asctime)s - %(message)s'))
    perf_handler.addFilter(logging.Filter('performance'))
    root_logger.addHandler(perf_handler)
    
    # Audit Handler - For security/compliance
    audit_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'audit.log'),
        maxBytes=50 * 1024 * 1024,  # 50 MB (audit logs are important)
        backupCount=20
    )
    audit_handler.setLevel(logging.INFO)
    audit_handler.setFormatter(JSONFormatter())
    audit_handler.addFilter(logging.Filter('audit'))
    root_logger.addHandler(audit_handler)
    
    # Log startup
    root_logger.info("Logging system initialized", extra={
        'log_level': log_level,
        'log_dir': log_dir,
        'handlers': len(root_logger.handlers)
    })


def get_logger(name: str, context: Dict[str, Any] = None) -> logging.Logger:
    """
    Get logger with optional context
    
    Args:
        name: Logger name (usually __name__)
        context: Additional context to include in all logs
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    if context:
        logger.addFilter(ContextFilter(context))
    
    return logger


class PerformanceLogger:
    """
    Log performance metrics
    """
    
    def __init__(self):
        self.logger = logging.getLogger('performance')
    
    def log_prediction(
        self,
        patient_id: str,
        inference_time_ms: float,
        risk_score: float,
        alert: bool
    ) -> None:
        """Log prediction performance"""
        self.logger.info('prediction', extra={
            'patient_id': patient_id,
            'inference_time_ms': inference_time_ms,
            'risk_score': risk_score,
            'alert': alert,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def log_api_request(
        self,
        endpoint: str,
        method: str,
        response_time_ms: float,
        status_code: int
    ) -> None:
        """Log API request performance"""
        self.logger.info('api_request', extra={
            'endpoint': endpoint,
            'method': method,
            'response_time_ms': response_time_ms,
            'status_code': status_code,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def log_model_training(
        self,
        model_type: str,
        training_time_seconds: float,
        samples: int,
        performance_metrics: Dict[str, float]
    ) -> None:
        """Log model training performance"""
        self.logger.info('model_training', extra={
            'model_type': model_type,
            'training_time_seconds': training_time_seconds,
            'samples': samples,
            'metrics': performance_metrics,
            'timestamp': datetime.utcnow().isoformat()
        })


class AuditLogger:
    """
    Log security and compliance events
    """
    
    def __init__(self):
        self.logger = logging.getLogger('audit')
    
    def log_access(
        self,
        user_id: str,
        action: str,
        resource: str,
        success: bool,
        ip_address: str = None,
        details: Dict[str, Any] = None
    ) -> None:
        """Log data access event"""
        self.logger.info('access', extra={
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'success': success,
            'ip_address': ip_address,
            'details': details or {},
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def log_authentication(
        self,
        user_id: str,
        success: bool,
        method: str,
        ip_address: str = None
    ) -> None:
        """Log authentication attempt"""
        self.logger.info('authentication', extra={
            'user_id': user_id,
            'success': success,
            'method': method,
            'ip_address': ip_address,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def log_data_export(
        self,
        user_id: str,
        data_type: str,
        record_count: int,
        approved_by: str = None
    ) -> None:
        """Log data export event"""
        self.logger.warning('data_export', extra={
            'user_id': user_id,
            'data_type': data_type,
            'record_count': record_count,
            'approved_by': approved_by,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def log_security_incident(
        self,
        incident_type: str,
        severity: str,
        description: str,
        affected_resources: list = None
    ) -> None:
        """Log security incident"""
        self.logger.critical('security_incident', extra={
            'incident_type': incident_type,
            'severity': severity,
            'description': description,
            'affected_resources': affected_resources or [],
            'timestamp': datetime.utcnow().isoformat()
        })


class ErrorLogger:
    """
    Enhanced error logging with context
    """
    
    def __init__(self):
        self.logger = logging.getLogger('errors')
    
    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        user_id: str = None
    ) -> None:
        """Log error with full context"""
        self.logger.error(
            f"{type(error).__name__}: {str(error)}",
            exc_info=True,
            extra={
                'error_type': type(error).__name__,
                'context': context or {},
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def log_validation_error(
        self,
        field: str,
        value: Any,
        expected: str,
        user_id: str = None
    ) -> None:
        """Log validation error"""
        self.logger.warning('validation_error', extra={
            'field': field,
            'value': value,
            'expected': expected,
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat()
        })


# Convenience functions
def log_prediction(patient_id: str, inference_time_ms: float, risk_score: float, alert: bool):
    """Quick function to log prediction"""
    perf_logger = PerformanceLogger()
    perf_logger.log_prediction(patient_id, inference_time_ms, risk_score, alert)


def log_access(user_id: str, action: str, resource: str, success: bool = True):
    """Quick function to log access"""
    audit_logger = AuditLogger()
    audit_logger.log_access(user_id, action, resource, success)


def log_error(error: Exception, context: Dict[str, Any] = None):
    """Quick function to log error"""
    error_logger = ErrorLogger()
    error_logger.log_error(error, context)


# Demo
def demo_logging():
    """Demonstrate logging capabilities"""
    print("\n" + "="*70)
    print(" "*20 + "LOGGING DEMO")
    print("="*70 + "\n")
    
    # Setup logging
    setup_logging(log_level="DEBUG", enable_json=True)
    
    # Get loggers
    app_logger = get_logger(__name__)
    perf_logger = PerformanceLogger()
    audit_logger = AuditLogger()
    error_logger = ErrorLogger()
    
    # 1. Basic logging
    print("1. Basic Logging:")
    app_logger.info("Application started successfully")
    app_logger.debug("Debug information")
    app_logger.warning("This is a warning")
    
    # 2. Performance logging
    print("\n2. Performance Logging:")
    perf_logger.log_prediction(
        patient_id='patient_001',
        inference_time_ms=45.3,
        risk_score=0.25,
        alert=False
    )
    
    # 3. Audit logging
    print("\n3. Audit Logging:")
    audit_logger.log_access(
        user_id='doctor_smith',
        action='view_patient_record',
        resource='patient_001',
        success=True,
        ip_address='192.168.1.100'
    )
    
    # 4. Error logging
    print("\n4. Error Logging:")
    try:
        raise ValueError("Example error for demonstration")
    except Exception as e:
        error_logger.log_error(e, context={'operation': 'demo'})
    
    print("\n✅ Logs written to logs/ directory")
    print("   - application.log (all logs)")
    print("   - errors.log (errors only)")
    print("   - performance.log (performance metrics)")
    print("   - audit.log (security/compliance)")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    demo_logging()