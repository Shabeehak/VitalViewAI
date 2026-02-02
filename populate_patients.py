# populate_patients.py
"""
Enhanced Patient Population System with ML Integration
Creates demo patients with varied health profiles and initial predictions

Features:
- Creates patients in streaming API
- Triggers initial ML predictions
- Stores patient metadata for dashboard
- Implements health profiles (stable, warning, critical)
- Comprehensive logging and audit trail
- Privacy-compliant patient ID generation

Usage:
    python populate_patients.py --count 10 --profile-mix balanced
    python populate_patients.py --count 5 --all-stable
"""

import requests
import random
import time
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys

sys.path.append('.')

from logging_config import setup_logging, get_logger, AuditLogger
from privacy_utils import PrivacyManager

# Initialize logging
setup_logging(log_level="INFO", enable_json=True)
logger = get_logger(__name__)
audit_logger = AuditLogger()

# API Configuration
API_BASE = "http://localhost:8000"
ML_API_BASE = "http://localhost:8001"

# Health Profile Definitions
HEALTH_PROFILES = {
    'stable': {
        'weight': 0.5,  # 50% of patients
        'description': 'Normal vitals, low risk',
        'sampling_interval': 300,  # 5 minutes
        'trigger_event': None,
        'wait_for_prediction': True
    },
    'warning': {
        'weight': 0.3,  # 30% of patients
        'description': 'Elevated vitals, medium risk',
        'sampling_interval': 180,  # 3 minutes
        'trigger_event': None,  # Will show variation naturally
        'wait_for_prediction': True
    },
    'critical': {
        'weight': 0.2,  # 20% of patients
        'description': 'Deteriorating vitals, high risk',
        'sampling_interval': 60,  # 1 minute
        'trigger_event': 'hypertensive_crisis',  # Trigger event after creation
        'wait_for_prediction': True
    }
}


class PatientPopulator:
    """Manages patient creation with ML integration"""
    
    def __init__(self):
        """Initialize populator with privacy manager"""
        logger.info("Initializing PatientPopulator")
        
        try:
            self.privacy_manager = PrivacyManager()
            logger.info("Privacy manager loaded successfully")
        except Exception as e:
            logger.warning(f"Privacy manager not available: {e}")
            self.privacy_manager = None
        
        self.created_patients = []
        self.failed_patients = []
        
    def generate_patient_id(self, profile: str) -> str:
        """Generate unique patient ID with profile prefix"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = random.randint(1000, 9999)
        patient_id = f"patient_{profile}_{timestamp}_{random_suffix}"
        
        logger.debug(
            f"Generated patient ID",
            extra={'patient_id': patient_id, 'profile': profile}
        )
        
        return patient_id
    
    def check_api_health(self) -> Tuple[bool, bool]:
        """Check if APIs are running"""
        streaming_api_ok = False
        ml_api_ok = False
        
        try:
            response = requests.get(f"{API_BASE}/health", timeout=3)
            streaming_api_ok = response.status_code == 200
            logger.info(f"Streaming API: {'✅ Online' if streaming_api_ok else '❌ Offline'}")
        except Exception as e:
            logger.error(f"Streaming API check failed: {e}")
        
        try:
            response = requests.get(f"{ML_API_BASE}/health", timeout=3)
            ml_api_ok = response.status_code == 200
            logger.info(f"ML API: {'✅ Online' if ml_api_ok else '❌ Offline'}")
        except Exception as e:
            logger.error(f"ML API check failed: {e}")
        
        return streaming_api_ok, ml_api_ok
    
    def create_patient_in_api(
        self, 
        patient_id: str, 
        profile: str,
        sampling_interval: int
    ) -> bool:
        """Create patient in streaming API"""
        logger.info(
            f"Creating patient in streaming API",
            extra={
                'patient_id': patient_id,
                'profile': profile,
                'sampling_interval': sampling_interval
            }
        )
        
        # Map our profiles to WearableDevice health profiles
        profile_mapping = {
            'stable': 'healthy',
            'warning': 'at_risk',
            'critical': 'deteriorating'
        }
        health_profile = profile_mapping.get(profile, 'healthy')
        
        try:
            payload = {
                "patient_id": patient_id,
                "sampling_interval_seconds": sampling_interval,
                "health_profile": health_profile  # NEW: Send health profile
            }
            
            response = requests.post(
                f"{API_BASE}/patients",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(
                    f"Patient created successfully",
                    extra={
                        'patient_id': patient_id,
                        'response': response.json()
                    }
                )
                
                # Audit log
                audit_logger.log_access(
                    user_id='populate_patients',
                    action='create_patient',
                    resource=patient_id,
                    success=True,
                    details={'profile': profile}
                )
                
                return True
            else:
                logger.error(
                    f"Failed to create patient",
                    extra={
                        'patient_id': patient_id,
                        'status_code': response.status_code,
                        'response': response.text
                    }
                )
                return False
                
        except Exception as e:
            logger.error(
                f"Patient creation failed: {str(e)}",
                exc_info=True,
                extra={'patient_id': patient_id}
            )
            return False
    
    def wait_for_data_accumulation(
        self, 
        patient_id: str, 
        min_samples: int = 30,
        max_wait_seconds: int = 180
    ) -> bool:
        """Wait for patient to accumulate enough data for prediction"""
        logger.info(
            f"Waiting for data accumulation",
            extra={
                'patient_id': patient_id,
                'min_samples': min_samples,
                'max_wait_seconds': max_wait_seconds
            }
        )
        
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait_seconds:
            try:
                # Check data count
                response = requests.get(
                    f"{API_BASE}/patients/{patient_id}/history",
                    params={'hours': 1, 'interval_minutes': 1},
                    timeout=5
                )
                
                if response.status_code == 200:
                    data_count = response.json()['count']
                    
                    if data_count >= min_samples:
                        logger.info(
                            f"Sufficient data accumulated",
                            extra={
                                'patient_id': patient_id,
                                'sample_count': data_count,
                                'elapsed_seconds': time.time() - start_time
                            }
                        )
                        return True
                    
                    # Log progress
                    if data_count % 10 == 0:
                        logger.debug(
                            f"Data accumulation progress",
                            extra={
                                'patient_id': patient_id,
                                'current_samples': data_count,
                                'target_samples': min_samples
                            }
                        )
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error checking data: {e}")
                time.sleep(5)
        
        logger.warning(
            f"Timeout waiting for data",
            extra={'patient_id': patient_id, 'elapsed_seconds': max_wait_seconds}
        )
        return False
    
    def trigger_event_for_patient(
        self, 
        patient_id: str, 
        event_type: str
    ) -> bool:
        """Trigger deterioration event for critical patients"""
        logger.warning(
            f"Triggering deterioration event",
            extra={
                'patient_id': patient_id,
                'event_type': event_type
            }
        )
        
        try:
            response = requests.post(
                f"{API_BASE}/patients/{patient_id}/trigger-event",
                json={
                    "patient_id": patient_id,
                    "event_type": event_type
                },
                timeout=5
            )
            
            if response.status_code == 200:
                logger.info(
                    f"Event triggered successfully",
                    extra={'patient_id': patient_id, 'event_type': event_type}
                )
                
                # Audit log
                audit_logger.log_access(
                    user_id='populate_patients',
                    action='trigger_event',
                    resource=patient_id,
                    success=True,
                    details={'event_type': event_type}
                )
                
                return True
            else:
                logger.error(
                    f"Failed to trigger event",
                    extra={'patient_id': patient_id, 'status_code': response.status_code}
                )
                return False
                
        except Exception as e:
            logger.error(f"Event trigger failed: {str(e)}", exc_info=True)
            return False
    
    def get_initial_prediction(self, patient_id: str) -> Optional[Dict]:
        """Get initial ML prediction for patient"""
        logger.info(
            f"Requesting initial prediction",
            extra={'patient_id': patient_id}
        )
        
        try:
            # Get recent vitals
            history_response = requests.get(
                f"{API_BASE}/patients/{patient_id}/history",
                params={'hours': 2, 'interval_minutes': 5},
                timeout=10
            )
            
            if history_response.status_code != 200:
                logger.error(f"Failed to get history for prediction")
                return None
            
            vitals_data = history_response.json()['data']
            
            if len(vitals_data) < 30:
                logger.warning(
                    f"Insufficient data for prediction",
                    extra={'patient_id': patient_id, 'samples': len(vitals_data)}
                )
                return None
            
            # Request prediction
            prediction_payload = {
                "patient_id": patient_id,
                "vitals": vitals_data,
                "user_id": "populate_patients",
                "user_role": "system"
            }
            
            prediction_response = requests.post(
                f"{ML_API_BASE}/predict",
                json=prediction_payload,
                timeout=15
            )
            
            if prediction_response.status_code == 200:
                prediction = prediction_response.json()
                
                logger.info(
                    f"Prediction obtained",
                    extra={
                        'patient_id': patient_id,
                        'risk_score': prediction['risk_score'],
                        'risk_level': prediction['risk_level'],
                        'alert': prediction['alert']
                    }
                )
                
                return prediction
            else:
                logger.error(
                    f"Prediction request failed",
                    extra={
                        'patient_id': patient_id,
                        'status_code': prediction_response.status_code
                    }
                )
                return None
                
        except Exception as e:
            logger.error(
                f"Prediction failed: {str(e)}",
                exc_info=True,
                extra={'patient_id': patient_id}
            )
            return None
    
    def create_patient_with_profile(
        self, 
        profile: str,
        wait_for_prediction: bool = True
    ) -> Optional[Dict]:
        """Create a patient with specified health profile"""
        profile_config = HEALTH_PROFILES[profile]
        patient_id = self.generate_patient_id(profile)
        
        print(f"\n{'='*70}")
        print(f"Creating Patient: {patient_id}")
        print(f"Profile: {profile.upper()} - {profile_config['description']}")
        print(f"{'='*70}")
        
        # Step 1: Create in streaming API
        print("⏳ Step 1/4: Creating patient in streaming API...")
        if not self.create_patient_in_api(
            patient_id, 
            profile, 
            profile_config['sampling_interval']
        ):
            print("❌ Failed to create patient in API")
            self.failed_patients.append({
                'patient_id': patient_id,
                'profile': profile,
                'reason': 'API creation failed'
            })
            return None
        
        print(f"✅ Patient created in streaming API")
        
        # Step 2: Wait for data accumulation
        if wait_for_prediction:
            print("⏳ Step 2/4: Waiting for data accumulation (30+ samples)...")
            if not self.wait_for_data_accumulation(patient_id, min_samples=30):
                print("⚠️ Warning: Insufficient data for prediction")
        else:
            print("⏭️ Step 2/4: Skipped (not waiting for data)")
            time.sleep(5)  # Brief wait for initial data
        
        # Step 3: Trigger event if critical profile
        if profile_config['trigger_event']:
            print(f"⏳ Step 3/4: Triggering {profile_config['trigger_event']} event...")
            self.trigger_event_for_patient(patient_id, profile_config['trigger_event'])
            print(f"⚠️ Event triggered: {profile_config['trigger_event']}")
            time.sleep(10)  # Wait for event to affect vitals
        else:
            print("⏭️ Step 3/4: No event trigger (stable/warning profile)")
        
        # Step 4: Get initial prediction
        prediction = None
        if wait_for_prediction:
            print("⏳ Step 4/4: Getting initial ML prediction...")
            prediction = self.get_initial_prediction(patient_id)
            
            if prediction:
                print(f"✅ Prediction obtained:")
                print(f"   Risk Score: {prediction['risk_score']:.1%}")
                print(f"   Risk Level: {prediction['risk_level']}")
                print(f"   Alert: {'⚠️ YES' if prediction['alert'] else '✓ NO'}")
            else:
                print("⚠️ Warning: Could not get initial prediction")
        else:
            print("⏭️ Step 4/4: Skipped (not requesting prediction)")
        
        # Store patient info
        patient_info = {
            'patient_id': patient_id,
            'profile': profile,
            'created_at': datetime.now().isoformat(),
            'sampling_interval': profile_config['sampling_interval'],
            'prediction': prediction,
            'event_triggered': profile_config['trigger_event']
        }
        
        self.created_patients.append(patient_info)
        
        print(f"\n✅ Patient {patient_id} fully initialized!")
        print(f"{'='*70}\n")
        
        return patient_info
    
    def populate_patients(
        self, 
        count: int = 6,
        profile_distribution: Optional[Dict[str, float]] = None,
        wait_for_predictions: bool = True
    ) -> Dict:
        """
        Populate multiple patients with health profiles
        
        Args:
            count: Number of patients to create
            profile_distribution: Custom profile weights (optional)
            wait_for_predictions: Whether to wait for initial predictions
        
        Returns:
            Summary dictionary with results
        """
        logger.info(
            f"Starting patient population",
            extra={
                'count': count,
                'wait_for_predictions': wait_for_predictions
            }
        )
        
        print("\n" + "="*70)
        print(" "*15 + "🏥 PATIENT POPULATION SYSTEM")
        print("="*70)
        
        # Check API health
        print("\n🔍 Checking API availability...")
        streaming_ok, ml_ok = self.check_api_health()
        
        if not streaming_ok:
            print("\n❌ ERROR: Streaming API is not available!")
            print("   Please start: python streaming_api_server.py")
            return {'success': False, 'error': 'Streaming API offline'}
        
        if not ml_ok and wait_for_predictions:
            print("\n⚠️ WARNING: ML API is not available!")
            print("   Predictions will be skipped.")
            print("   To enable predictions, start: python ml_server.py")
            wait_for_predictions = False
        
        # Determine profile distribution
        if profile_distribution:
            profiles = list(profile_distribution.keys())
            weights = list(profile_distribution.values())
        else:
            profiles = list(HEALTH_PROFILES.keys())
            weights = [HEALTH_PROFILES[p]['weight'] for p in profiles]
        
        # Select profiles for patients
        selected_profiles = random.choices(profiles, weights=weights, k=count)
        
        print(f"\n📊 Creating {count} patients with distribution:")
        for profile in set(selected_profiles):
            profile_count = selected_profiles.count(profile)
            print(f"   {profile.upper()}: {profile_count} patients")
        
        print("\n" + "="*70)
        
        # Create patients
        start_time = time.time()
        
        for i, profile in enumerate(selected_profiles, 1):
            print(f"\n[{i}/{count}] Creating patient...")
            
            patient_info = self.create_patient_with_profile(
                profile=profile,
                wait_for_prediction=wait_for_predictions
            )
            
            if patient_info:
                # Brief delay between patients
                if i < count:
                    time.sleep(2)
        
        elapsed_time = time.time() - start_time
        
        # Generate summary
        summary = self.generate_summary(elapsed_time)
        
        # Save results
        self.save_results()
        
        return summary
    
    def generate_summary(self, elapsed_time: float) -> Dict:
        """Generate summary of population operation"""
        print("\n" + "="*70)
        print(" "*20 + "📊 POPULATION SUMMARY")
        print("="*70)
        
        total = len(self.created_patients) + len(self.failed_patients)
        success_rate = len(self.created_patients) / total * 100 if total > 0 else 0
        
        print(f"\n✅ Successfully created: {len(self.created_patients)}/{total}")
        print(f"❌ Failed: {len(self.failed_patients)}")
        print(f"📊 Success rate: {success_rate:.1f}%")
        print(f"⏱️ Total time: {elapsed_time:.1f} seconds")
        
        if self.created_patients:
            print(f"\n📋 Created Patients:")
            
            # Group by profile
            by_profile = {}
            for patient in self.created_patients:
                profile = patient['profile']
                if profile not in by_profile:
                    by_profile[profile] = []
                by_profile[profile].append(patient)
            
            for profile, patients in by_profile.items():
                print(f"\n   {profile.upper()} ({len(patients)} patients):")
                for patient in patients:
                    pred = patient.get('prediction')
                    if pred:
                        print(f"   • {patient['patient_id']}")
                        print(f"     Risk: {pred['risk_level']} ({pred['risk_score']:.1%})")
                    else:
                        print(f"   • {patient['patient_id']} (no prediction yet)")
        
        if self.failed_patients:
            print(f"\n❌ Failed Patients:")
            for failed in self.failed_patients:
                print(f"   • {failed['patient_id']}: {failed['reason']}")
        
        print("\n💡 Next Steps:")
        print("   1. Start dashboard: streamlit run streamlit_dashboard.py")
        print("   2. Monitor patients in real-time")
        print("   3. Check predictions and alerts")
        
        print("\n" + "="*70 + "\n")
        
        summary = {
            'success': True,
            'total_created': len(self.created_patients),
            'total_failed': len(self.failed_patients),
            'success_rate': success_rate,
            'elapsed_time': elapsed_time,
            'patients': self.created_patients,
            'failed': self.failed_patients
        }
        
        logger.info(
            "Patient population completed",
            extra=summary
        )
        
        return summary
    
    def save_results(self):
        """Save population results to file"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'created_patients': self.created_patients,
            'failed_patients': self.failed_patients
        }
        
        filename = f"logs/patient_population_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {filename}")
            print(f"📝 Results saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Populate patients with health profiles and ML predictions'
    )
    
    parser.add_argument(
        '--count',
        type=int,
        default=6,
        help='Number of patients to create (default: 6)'
    )
    
    parser.add_argument(
        '--all-stable',
        action='store_true',
        help='Create all patients with stable profile'
    )
    
    parser.add_argument(
        '--all-critical',
        action='store_true',
        help='Create all patients with critical profile'
    )
    
    parser.add_argument(
        '--profile-mix',
        choices=['balanced', 'mostly-stable', 'high-risk'],
        default='balanced',
        help='Profile distribution preset'
    )
    
    parser.add_argument(
        '--no-predictions',
        action='store_true',
        help='Skip waiting for predictions (faster but less complete)'
    )
    
    args = parser.parse_args()
    
    # Determine profile distribution
    profile_distribution = None
    
    if args.all_stable:
        profile_distribution = {'stable': 1.0}
    elif args.all_critical:
        profile_distribution = {'critical': 1.0}
    elif args.profile_mix == 'mostly-stable':
        profile_distribution = {'stable': 0.7, 'warning': 0.2, 'critical': 0.1}
    elif args.profile_mix == 'high-risk':
        profile_distribution = {'stable': 0.2, 'warning': 0.3, 'critical': 0.5}
    
    # Create populator and run
    populator = PatientPopulator()
    
    populator.populate_patients(
        count=args.count,
        profile_distribution=profile_distribution,
        wait_for_predictions=not args.no_predictions
    )


if __name__ == "__main__":
    main()


# # populate_patients.py
# """
# Create demo patients with RANDOM IDs and mixed health profiles.
# Each patient is assigned a health profile:
# - stable: mostly normal vitals
# """

# import requests
# import random
# from datetime import datetime

# API_BASE = "http://localhost:8000"

# HEALTH_PROFILES = [
#     ("stable", 0.5),
#     ("warning", 0.3),
#     ("critical", 0.2)
# ]

# def generate_patient_id():
#     """Generate unique patient ID"""
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     rand = random.randint(1000, 9999)
#     return f"patient_{ts}_{rand}"

# def pick_health_profile():
#     """Randomly pick health profile"""
#     profiles, weights = zip(*HEALTH_PROFILES)
#     return random.choices(profiles, weights=weights, k=1)[0]

# def create_demo_patients(count: int = 6):
#     print("\n🏥 Creating demo patients...\n")

#     for _ in range(count):
#         patient_id = generate_patient_id()
#         profile = pick_health_profile()

#         payload = {
#             "patient_id": patient_id,
#             "sampling_interval_seconds": random.choice([60, 300, 600]),
#             "metadata": {
#                 "demo_profile": profile,
#                 "created_for": "frontend_demo"
#             }
#         }

#         try:
#             r = requests.post(f"{API_BASE}/patients", json=payload, timeout=5)

#             if r.status_code == 200:
#                 print(f"✅ Created {patient_id} | Profile: {profile.upper()}")
#             else:
#                 print(f"⚠️ Failed to create {patient_id} | Status: {r.status_code}")

#         except Exception as e:
#             print(f"❌ Error creating patient {patient_id}: {e}")

#     print("\n🎉 Demo patients created successfully!\n")

# if __name__ == "__main__":
#     create_demo_patients()
