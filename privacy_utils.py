# privacy_utils.py
"""
Privacy and Security Utilities with Comprehensive Logging
Implements encryption, hashing, and de-identification

Enhanced Features:
- Structured logging for all privacy operations
- Audit trail for security events
- Performance tracking
- Error handling with context

Task 3 Deliverable: Privacy and Compliance Implementation
"""

import hashlib
import hmac
import os
import json
import yaml
from datetime import datetime
from typing import Dict, Any, List, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import numpy as np

from logging_config import (
    setup_logging, get_logger,
    AuditLogger, ErrorLogger
)

# Initialize logging
setup_logging(log_level="INFO", enable_json=True)
logger = get_logger(__name__)
audit_logger = AuditLogger()
error_logger = ErrorLogger()


class PrivacyManager:
    """
    Manages privacy and security operations with comprehensive logging
    """
    
    def __init__(self, config_path: str = "privacy_config.yaml"):
        """Initialize privacy manager with logging"""
        logger.info(
            "Initializing PrivacyManager",
            extra={'config_path': config_path}
        )
        
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Get encryption key
            self.encryption_key = self._get_or_create_encryption_key()
            self.cipher_suite = Fernet(self.encryption_key)
            
            # Get patient ID salt
            self.patient_id_salt = self._get_or_create_salt()
            
            logger.info(
                "PrivacyManager initialized successfully",
                extra={
                    'encryption_enabled': self.config['encryption']['at_rest']['enabled'],
                    'rbac_enabled': self.config['access_control']['rbac']['enabled'],
                    'audit_enabled': self.config['audit_logging']['enabled']
                }
            )
            
            # Audit log
            audit_logger.log_access(
                user_id='system',
                action='privacy_manager_init',
                resource='privacy_manager',
                success=True
            )
            
        except Exception as e:
            logger.error(
                f"Failed to initialize PrivacyManager: {str(e)}",
                exc_info=True
            )
            error_logger.log_error(e, context={
                'operation': 'init',
                'config_path': config_path
            })
            raise
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get encryption key from environment or generate new one"""
        logger.debug("Loading encryption key")
        
        key = os.environ.get('ENCRYPTION_KEY')
        
        if key:
            logger.info("Encryption key loaded from environment")
            return base64.b64decode(key)
        else:
            # Generate new key
            key = Fernet.generate_key()
            key_b64 = base64.b64encode(key).decode()
            
            logger.warning(
                "Generated new encryption key",
                extra={'key_preview': key_b64[:20] + '...'}
            )
            
            print(f"⚠️ Set ENCRYPTION_KEY environment variable to: {key_b64}")
            
            return key
    
    def _get_or_create_salt(self) -> bytes:
        """Get salt from environment or generate new one"""
        logger.debug("Loading patient ID salt")
        
        salt = os.environ.get('PATIENT_ID_SALT')
        
        if salt:
            logger.info("Salt loaded from environment")
            return salt.encode()
        else:
            # Generate new salt
            salt_bytes = os.urandom(32)
            salt_hex = salt_bytes.hex()
            
            logger.warning(
                "Generated new salt",
                extra={'salt_preview': salt_hex[:20] + '...'}
            )
            
            print(f"⚠️ Set PATIENT_ID_SALT environment variable to: {salt_hex}")
            
            return salt_bytes
    
    # =========================================================================
    # PATIENT ID HASHING
    # =========================================================================
    
    def hash_patient_id(self, patient_id: str) -> str:
        """Hash patient ID using SHA-256 with salt and logging"""
        if not self.config['de_identification']['patient_id_hashing']['enabled']:
            logger.debug("Patient ID hashing disabled, returning original")
            return patient_id
        
        try:
            algorithm = self.config['de_identification']['patient_id_hashing']['algorithm']
            
            if algorithm == 'SHA-256':
                hasher = hashlib.sha256()
            elif algorithm == 'SHA-512':
                hasher = hashlib.sha512()
            else:
                hasher = hashlib.sha256()
            
            # Hash with salt
            hasher.update(self.patient_id_salt)
            hasher.update(patient_id.encode())
            
            hashed = hasher.hexdigest()
            
            logger.debug(
                "Patient ID hashed",
                extra={
                    'algorithm': algorithm,
                    'original_preview': patient_id[:8] + '...',
                    'hashed_preview': hashed[:16] + '...'
                }
            )
            
            return hashed
            
        except Exception as e:
            logger.error(
                f"Patient ID hashing failed: {str(e)}",
                exc_info=True
            )
            error_logger.log_error(e, context={'operation': 'hash_patient_id'})
            raise
    
    def hash_patient_id_hmac(self, patient_id: str) -> str:
        """Hash patient ID using HMAC-SHA256 with logging"""
        logger.debug("Hashing patient ID with HMAC")
        
        try:
            mac = hmac.new(
                self.patient_id_salt,
                patient_id.encode(),
                hashlib.sha256
            )
            
            hashed = mac.hexdigest()
            
            logger.debug(
                "Patient ID HMAC hashed",
                extra={'hashed_preview': hashed[:16] + '...'}
            )
            
            return hashed
            
        except Exception as e:
            logger.error(f"HMAC hashing failed: {str(e)}", exc_info=True)
            error_logger.log_error(e, context={'operation': 'hash_patient_id_hmac'})
            raise
    
    # =========================================================================
    # DATA ENCRYPTION
    # =========================================================================
    
    def encrypt_data(self, data: Any) -> str:
        """Encrypt sensitive data with logging"""
        if not self.config['encryption']['at_rest']['enabled']:
            logger.debug("Encryption disabled, returning plaintext JSON")
            return json.dumps(data)
        
        try:
            # Serialize to JSON
            json_data = json.dumps(data)
            data_size = len(json_data)
            
            # Encrypt
            encrypted = self.cipher_suite.encrypt(json_data.encode())
            encrypted_b64 = base64.b64encode(encrypted).decode()
            
            logger.debug(
                "Data encrypted",
                extra={
                    'original_size_bytes': data_size,
                    'encrypted_size_bytes': len(encrypted_b64)
                }
            )
            
            return encrypted_b64
            
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}", exc_info=True)
            error_logger.log_error(e, context={'operation': 'encrypt_data'})
            raise
    
    def decrypt_data(self, encrypted_data: str) -> Any:
        """Decrypt encrypted data with logging"""
        if not self.config['encryption']['at_rest']['enabled']:
            logger.debug("Encryption disabled, parsing plaintext JSON")
            return json.loads(encrypted_data)
        
        try:
            # Decode base64
            encrypted_bytes = base64.b64decode(encrypted_data)
            
            # Decrypt
            decrypted = self.cipher_suite.decrypt(encrypted_bytes)
            
            # Parse JSON
            data = json.loads(decrypted.decode())
            
            logger.debug("Data decrypted successfully")
            
            return data
            
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}", exc_info=True)
            error_logger.log_error(e, context={'operation': 'decrypt_data'})
            raise
    
    def encrypt_field(self, value: str) -> str:
        """Encrypt a single field value with logging"""
        try:
            encrypted = self.cipher_suite.encrypt(value.encode())
            encrypted_b64 = base64.b64encode(encrypted).decode()
            
            logger.debug("Field encrypted")
            
            return encrypted_b64
            
        except Exception as e:
            logger.error(f"Field encryption failed: {str(e)}", exc_info=True)
            error_logger.log_error(e, context={'operation': 'encrypt_field'})
            raise
    
    def decrypt_field(self, encrypted_value: str) -> str:
        """Decrypt a single field value with logging"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_value)
            decrypted = self.cipher_suite.decrypt(encrypted_bytes)
            
            logger.debug("Field decrypted")
            
            return decrypted.decode()
            
        except Exception as e:
            logger.error(f"Field decryption failed: {str(e)}", exc_info=True)
            error_logger.log_error(e, context={'operation': 'decrypt_field'})
            raise
    
    # =========================================================================
    # PII MASKING
    # =========================================================================
    
    def mask_pii(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask personally identifiable information with logging"""
        if not self.config['de_identification']['pii_masking']['enabled']:
            logger.debug("PII masking disabled")
            return data
        
        try:
            masked_data = data.copy()
            fields_to_mask = self.config['de_identification']['pii_masking']['fields_to_mask']
            method = self.config['de_identification']['pii_masking']['masking_method']
            
            masked_count = 0
            
            for field in fields_to_mask:
                if field in masked_data:
                    if method == 'hash':
                        masked_data[field] = hashlib.sha256(
                            str(masked_data[field]).encode()
                        ).hexdigest()[:16] + "..."
                    elif method == 'redact':
                        masked_data[field] = "[REDACTED]"
                    elif method == 'pseudonymize':
                        masked_data[field] = f"PATIENT_{hashlib.sha256(str(masked_data[field]).encode()).hexdigest()[:8]}"
                    
                    masked_count += 1
            
            logger.info(
                "PII masked",
                extra={
                    'fields_masked': masked_count,
                    'masking_method': method
                }
            )
            
            # Audit log
            audit_logger.log_access(
                user_id='system',
                action='mask_pii',
                resource='patient_data',
                success=True,
                details={'fields_masked': masked_count}
            )
            
            return masked_data
            
        except Exception as e:
            logger.error(f"PII masking failed: {str(e)}", exc_info=True)
            error_logger.log_error(e, context={'operation': 'mask_pii'})
            raise
    
    # =========================================================================
    # DATA ANONYMIZATION
    # =========================================================================
    
    def anonymize_for_research(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize data for research purposes with logging"""
        if not self.config['de_identification']['anonymization']['enabled']:
            logger.debug("Anonymization disabled")
            return data
        
        try:
            anonymized = data.copy()
            removed_identifiers = []
            
            # Remove direct identifiers
            identifiers = ['patient_id', 'patient_name', 'email', 'phone', 'address', 'ssn']
            for identifier in identifiers:
                if identifier in anonymized:
                    del anonymized[identifier]
                    removed_identifiers.append(identifier)
            
            # Generalize age
            if 'age' in anonymized and self.config['de_identification']['anonymization']['age_generalization']['enabled']:
                age = anonymized['age']
                bins = self.config['de_identification']['anonymization']['age_generalization']['bins']
                labels = self.config['de_identification']['anonymization']['age_generalization']['labels']
                
                for i, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
                    if lower <= age < upper:
                        anonymized['age_range'] = labels[i]
                        del anonymized['age']
                        break
            
            # Add noise to continuous variables
            if 'heart_rate' in anonymized:
                anonymized['heart_rate'] += np.random.normal(0, 1)
            
            logger.info(
                "Data anonymized for research",
                extra={
                    'identifiers_removed': len(removed_identifiers),
                    'removed_fields': removed_identifiers
                }
            )
            
            # Audit log
            audit_logger.log_data_export(
                user_id='system',
                data_type='anonymized_research_data',
                record_count=1,
                approved_by='privacy_manager'
            )
            
            return anonymized
            
        except Exception as e:
            logger.error(f"Anonymization failed: {str(e)}", exc_info=True)
            error_logger.log_error(e, context={'operation': 'anonymize_for_research'})
            raise
    
    # =========================================================================
    # AUDIT LOGGING
    # =========================================================================
    
    def log_access(
        self, 
        user_id: str, 
        action: str, 
        resource: str,
        success: bool = True,
        details: Optional[Dict] = None
    ) -> None:
        """Log data access for audit trail with enhanced logging"""
        if not self.config['audit_logging']['enabled']:
            return
        
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'action': action,
                'resource': resource,
                'success': success,
                'details': details or {}
            }
            
            logger.info(
                f"Access logged",
                extra={
                    'user_id': user_id,
                    'action': action,
                    'resource': resource,
                    'success': success
                }
            )
            
            # Log to audit trail via audit logger
            audit_logger.log_access(
                user_id=user_id,
                action=action,
                resource=resource,
                success=success,
                details=details
            )
            
            # Save to audit log file
            os.makedirs('logs', exist_ok=True)
            with open('logs/audit_trail.jsonl', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Audit logging failed: {str(e)}", exc_info=True)
            # Don't raise - audit logging failure shouldn't break functionality
    
    # =========================================================================
    # ACCESS CONTROL
    # =========================================================================
    
    def check_permission(self, user_role: str, permission: str) -> bool:
        """Check if user role has permission with logging"""
        if not self.config['access_control']['rbac']['enabled']:
            logger.debug("RBAC disabled, allowing all permissions")
            return True
        
        try:
            roles = self.config['access_control']['rbac']['roles']
            
            if user_role not in roles:
                logger.warning(
                    f"Permission check failed: unknown role",
                    extra={
                        'user_role': user_role,
                        'permission': permission
                    }
                )
                return False
            
            has_permission = permission in roles[user_role]['permissions']
            
            if not has_permission:
                logger.warning(
                    f"Permission denied",
                    extra={
                        'user_role': user_role,
                        'permission': permission,
                        'allowed_permissions': roles[user_role]['permissions']
                    }
                )
            else:
                logger.debug(
                    f"Permission granted",
                    extra={
                        'user_role': user_role,
                        'permission': permission
                    }
                )
            
            return has_permission
            
        except Exception as e:
            logger.error(f"Permission check failed: {str(e)}", exc_info=True)
            error_logger.log_error(e, context={
                'operation': 'check_permission',
                'user_role': user_role,
                'permission': permission
            })
            return False  # Fail closed
    
    def validate_data_access(
        self,
        user_role: str,
        data_type: str,
        patient_id: Optional[str] = None
    ) -> bool:
        """Validate if user can access specific data with logging"""
        if not self.config['access_control']['rbac']['enabled']:
            return True
        
        try:
            roles = self.config['access_control']['rbac']['roles']
            
            if user_role not in roles:
                logger.warning(
                    f"Data access validation failed: unknown role",
                    extra={'user_role': user_role}
                )
                return False
            
            data_access = roles[user_role]['data_access']
            
            # Check data access level
            allowed = False
            if data_access == 'all':
                allowed = True
            elif data_access == 'anonymized_only' and data_type == 'anonymized':
                allowed = True
            elif data_access == 'assigned_patients' and patient_id:
                allowed = True  # In production, check assignment
            
            logger.info(
                f"Data access validation",
                extra={
                    'user_role': user_role,
                    'data_type': data_type,
                    'patient_id': patient_id,
                    'allowed': allowed
                }
            )
            
            return allowed
            
        except Exception as e:
            logger.error(f"Data access validation failed: {str(e)}", exc_info=True)
            error_logger.log_error(e, context={'operation': 'validate_data_access'})
            return False
    
    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate secure random token with logging"""
        try:
            token = base64.b64encode(os.urandom(length)).decode()
            
            logger.debug(
                "Secure token generated",
                extra={'token_length': len(token)}
            )
            
            return token
            
        except Exception as e:
            logger.error(f"Token generation failed: {str(e)}", exc_info=True)
            error_logger.log_error(e, context={'operation': 'generate_secure_token'})
            raise
    
    def validate_data_integrity(self, data: Dict[str, Any]) -> bool:
        """Validate data integrity with logging"""
        if not self.config['data_validation']['enabled']:
            return True
        
        try:
            # Check required fields
            required_fields = ['heart_rate', 'bp_systolic', 'bp_diastolic', 'spo2']
            
            for field in required_fields:
                if field not in data:
                    logger.warning(
                        f"Data validation failed: missing field",
                        extra={'missing_field': field}
                    )
                    return False
            
            # Range validation
            ranges = {
                'heart_rate': (30, 200),
                'bp_systolic': (70, 250),
                'bp_diastolic': (40, 150),
                'spo2': (70, 100),
                'respiratory_rate': (8, 40),
                'temperature': (35.0, 42.0)
            }
            
            for field, (min_val, max_val) in ranges.items():
                if field in data:
                    value = data[field]
                    if not (min_val <= value <= max_val):
                        logger.warning(
                            f"Data validation failed: value out of range",
                            extra={
                                'field': field,
                                'value': value,
                                'min': min_val,
                                'max': max_val
                            }
                        )
                        return False
            
            logger.debug("Data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}", exc_info=True)
            error_logger.log_error(e, context={'operation': 'validate_data_integrity'})
            return False


# Demo and testing
def demo_privacy_features():
    """Demonstrate privacy features with logging"""
    print("\n" + "="*70)
    print(" "*20 + "PRIVACY MANAGER DEMO")
    print("="*70)
    
    # Initialize
    pm = PrivacyManager()
    
    # 1. Patient ID Hashing
    print("\n1️⃣ Patient ID Hashing:")
    patient_id = "patient_12345"
    hashed_id = pm.hash_patient_id(patient_id)
    print(f"   Original: {patient_id}")
    print(f"   Hashed:   {hashed_id[:32]}...")
    
    # 2. Data Encryption
    print("\n2️⃣ Data Encryption:")
    vitals = {
        'heart_rate': 75,
        'bp_systolic': 120,
        'timestamp': '2024-01-01 12:00:00'
    }
    encrypted = pm.encrypt_data(vitals)
    print(f"   Original: {vitals}")
    print(f"   Encrypted: {encrypted[:50]}...")
    decrypted = pm.decrypt_data(encrypted)
    print(f"   Decrypted: {decrypted}")
    print(f"   Match: {vitals == decrypted}")
    
    # 3. PII Masking
    print("\n3️⃣ PII Masking:")
    patient_data = {
        'patient_name': 'John Doe',
        'email': 'john.doe@example.com',
        'heart_rate': 75,
        'age': 45
    }
    masked = pm.mask_pii(patient_data)
    print(f"   Original: {patient_data}")
    print(f"   Masked:   {masked}")
    
    # 4. Access Control
    print("\n4️⃣ Access Control:")
    print(f"   Clinician can read patient data: {pm.check_permission('clinician', 'read_patient_data')}")
    print(f"   Nurse can manage users: {pm.check_permission('nurse', 'manage_users')}")
    print(f"   Researcher can read anonymized: {pm.check_permission('researcher', 'read_anonymized_data')}")
    
    # 5. Audit Logging
    print("\n5️⃣ Audit Logging:")
    pm.log_access(
        user_id='doctor_smith',
        action='read_patient_data',
        resource='patient_12345',
        success=True,
        details={'vitals_accessed': ['heart_rate', 'bp']}
    )
    print("   ✅ Access logged to audit trail")
    
    print("\n" + "="*70)
    print(" "*20 + "✅ DEMO COMPLETE")
    print("="*70)
    print("\n📝 Check logs/ directory for detailed logs")


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run demo
    demo_privacy_features()