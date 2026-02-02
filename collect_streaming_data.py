"""
Streaming Data Collector
Collects data from the streaming API and saves for ML training

Save as: collect_streaming_data.py (in root folder)
Run: python collect_streaming_data.py
"""

import requests
import pandas as pd
from datetime import datetime
import time
import sys

class APIDataCollector:
    """
    Collects health data from streaming API
    
    Methods:
    1. create_patient() - Register patient for monitoring
    2. get_current() - Get single reading
    3. get_history() - Get batch historical data
    4. collect_continuous() - Collect data over time period
    """
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.session = requests.Session()
    
    def check_api_health(self) -> bool:
        """Check if API is running"""
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def create_patient(self, patient_id: str, sampling_interval: int = 300):
        """
        Start monitoring for a patient
        
        Args:
            patient_id: Patient identifier
            sampling_interval: Seconds between readings
        """
        endpoint = f"{self.api_url}/patients"
        
        payload = {
            "patient_id": patient_id,
            "sampling_interval_seconds": sampling_interval
        }
        
        response = self.session.post(endpoint, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error: {response.json()}")
            return None
    
    def get_current_reading(self, patient_id: str):
        """Get current vital signs"""
        endpoint = f"{self.api_url}/patients/{patient_id}/current"
        
        response = self.session.get(endpoint)
        
        if response.status_code == 200:
            return response.json()['data']
        else:
            print(f"❌ Error: {response.status_code}")
            return None
    
    def get_historical_data(self, patient_id: str, hours: int = 24, interval_minutes: int = 5):
        """
        Get historical data
        
        Args:
            patient_id: Patient identifier
            hours: Hours of history to retrieve
            interval_minutes: Minutes between readings
        """
        endpoint = f"{self.api_url}/patients/{patient_id}/history"
        
        params = {
            "hours": hours,
            "interval_minutes": interval_minutes
        }
        
        response = self.session.get(endpoint, params=params)
        
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data['data'])
        else:
            print(f"❌ Error: {response.status_code}")
            return pd.DataFrame()
    
    def trigger_event(self, patient_id: str, event_type: str = "hypertensive_crisis"):
        """Trigger deterioration event for testing"""
        endpoint = f"{self.api_url}/patients/{patient_id}/trigger-event"
        
        payload = {
            "patient_id": patient_id,
            "event_type": event_type
        }
        
        response = self.session.post(endpoint, json=payload)
        return response.json()
    
    def collect_continuous(
        self, 
        patient_id: str, 
        duration_minutes: int = 10,
        poll_interval_seconds: int = 30,
        output_path: str = "data/processed/streaming_data.csv"
    ):
        """
        Collect data continuously for a duration
        
        Args:
            patient_id: Patient to monitor
            duration_minutes: How long to collect
            poll_interval_seconds: Seconds between API polls
            output_path: Where to save CSV
        """
        print("\n" + "="*70)
        print("📡 CONTINUOUS DATA COLLECTION")
        print("="*70)
        print(f"\n⚙️  Configuration:")
        print(f"   Patient: {patient_id}")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Poll interval: {poll_interval_seconds} seconds")
        print(f"   Output: {output_path}")
        
        collected_data = []
        start_time = datetime.now()
        
        print(f"\n🔄 Starting collection...")
        print(f"   Press Ctrl+C to stop early\n")
        
        try:
            while True:
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                
                if elapsed >= duration_minutes:
                    break
                
                # Get reading
                reading = self.get_current_reading(patient_id)
                
                if reading:
                    collected_data.append(reading)
                    
                    # Progress display
                    hr = reading['heart_rate']
                    bp = f"{reading['bp_systolic']}/{reading['bp_diastolic']}"
                    spo2 = reading['spo2']
                    
                    progress = (elapsed / duration_minutes) * 100
                    print(f"   [{progress:5.1f}%] HR:{hr:5.1f} BP:{bp} SpO2:{spo2:4.1f}% ✓")
                
                # Wait before next poll
                time.sleep(poll_interval_seconds)
        
        except KeyboardInterrupt:
            print(f"\n⏹️  Collection stopped by user")
        
        # Save data
        if collected_data:
            df = pd.DataFrame(collected_data)
            df.to_csv(output_path, index=False)
            
            print(f"\n" + "="*70)
            print("✅ COLLECTION COMPLETE")
            print("="*70)
            print(f"\n📊 Results:")
            print(f"   Readings collected: {len(df)}")
            print(f"   Duration: {elapsed:.1f} minutes")
            print(f"   File: {output_path}")
            
            print(f"\n📈 Data Summary:")
            print(df[['heart_rate', 'bp_systolic', 'spo2', 'temperature']].describe().round(1))
            
            return df
        else:
            print(f"\n❌ No data collected")
            return pd.DataFrame()


# ==================== DEMO SCENARIOS ====================

def demo_basic_usage(collector):
    """Demo: Basic API usage"""
    print("\n" + "="*70)
    print("DEMO 1: Basic API Usage")
    print("="*70)
    
    patient_id = "demo_patient_001"
    
    # Create patient
    print(f"\n1️⃣  Creating patient monitoring...")
    result = collector.create_patient(patient_id)
    if result:
        print(f"   ✅ {result['message']}")
    
    # Get current reading
    print(f"\n2️⃣  Getting current reading...")
    reading = collector.get_current_reading(patient_id)
    if reading:
        print(f"   ✅ Current vitals:")
        print(f"      HR: {reading['heart_rate']}")
        print(f"      BP: {reading['bp_systolic']}/{reading['bp_diastolic']}")
        print(f"      SpO2: {reading['spo2']}%")
    
    # Get historical data
    print(f"\n3️⃣  Getting 24h historical data...")
    df = collector.get_historical_data(patient_id, hours=24, interval_minutes=5)
    if not df.empty:
        print(f"   ✅ Retrieved {len(df)} readings")
        print(f"\n   Sample data:")
        print(df.head(3).to_string(index=False))
    
    return df

def demo_deterioration_testing(collector):
    """Demo: Test deterioration event detection"""
    print("\n" + "="*70)
    print("DEMO 2: Deterioration Event Testing")
    print("="*70)
    
    patient_id = "test_patient_002"
    
    # Create patient
    print(f"\n1️⃣  Creating patient...")
    collector.create_patient(patient_id)
    
    # Collect normal readings
    print(f"\n2️⃣  Collecting 5 normal readings...")
    normal_readings = []
    for i in range(5):
        reading = collector.get_current_reading(patient_id)
        normal_readings.append(reading)
        print(f"   Reading {i+1}: HR={reading['heart_rate']:.1f} BP={reading['bp_systolic']:.0f}")
        time.sleep(1)
    
    # Trigger event
    print(f"\n3️⃣  ⚠️  Triggering hypertensive crisis...")
    collector.trigger_event(patient_id, "hypertensive_crisis")
    time.sleep(2)  # Let severity build
    
    # Collect deteriorating readings
    print(f"\n4️⃣  Collecting 5 deteriorating readings...")
    deteriorating_readings = []
    for i in range(5):
        reading = collector.get_current_reading(patient_id)
        deteriorating_readings.append(reading)
        print(f"   Reading {i+1}: HR={reading['heart_rate']:.1f} BP={reading['bp_systolic']:.0f} ⚠️")
        time.sleep(1)
    
    # Analysis
    normal_df = pd.DataFrame(normal_readings)
    deteriorating_df = pd.DataFrame(deteriorating_readings)
    
    print(f"\n5️⃣  Analysis:")
    print(f"   Normal BP (avg): {normal_df['bp_systolic'].mean():.1f} mmHg")
    print(f"   Deteriorating BP (avg): {deteriorating_df['bp_systolic'].mean():.1f} mmHg")
    print(f"   Change: +{deteriorating_df['bp_systolic'].mean() - normal_df['bp_systolic'].mean():.1f} mmHg")
    
    return normal_df, deteriorating_df

def demo_continuous_collection(collector):
    """Demo: Collect data over time"""
    print("\n" + "="*70)
    print("DEMO 3: Continuous Data Collection")
    print("="*70)
    
    patient_id = "continuous_patient_003"
    
    # Create patient
    print(f"\n1️⃣  Creating patient...")
    collector.create_patient(patient_id)
    
    # Collect data
    print(f"\n2️⃣  Starting continuous collection...")
    df = collector.collect_continuous(
        patient_id=patient_id,
        duration_minutes=5,  # 5 minutes
        poll_interval_seconds=15,  # Every 15 seconds
        output_path="data/processed/streaming_demo.csv"
    )
    
    return df


# ==================== MAIN ====================

def main():
    """Main execution"""
    print("\n" + "="*70)
    print(" "*15 + "🏥 STREAMING DATA COLLECTOR")
    print("="*70)
    
    # Initialize collector
    collector = APIDataCollector()
    
    # Check if API is running
    print(f"\n🔍 Checking API connection...")
    if not collector.check_api_health():
        print(f"\n❌ API is not running!")
        print(f"\n📝 Start the API first:")
        print(f"   python streaming_api_server.py")
        print(f"\n   Then run this script again.")
        sys.exit(1)
    
    print(f"   ✅ API is running at {collector.api_url}")
    
    # Menu
    print(f"\n📋 Choose a demo:")
    print(f"   1. Basic usage (create patient, get data)")
    print(f"   2. Deterioration testing (trigger event)")
    print(f"   3. Continuous collection (5 min)")
    print(f"   4. Full dataset generation (30 min)")
    
    choice = input(f"\n   Enter choice (1-4): ").strip()
    
    if choice == '1':
        demo_basic_usage(collector)
    
    elif choice == '2':
        demo_deterioration_testing(collector)
    
    elif choice == '3':
        demo_continuous_collection(collector)
    
    elif choice == '4':
        patient_id = "ml_training_patient"
        collector.create_patient(patient_id)
        
        print(f"\n⏱️  This will collect 30 minutes of data...")
        print(f"   That's ~60 readings (one every 30 seconds)")
        confirm = input(f"\n   Continue? (y/n): ").strip().lower()
        
        if confirm == 'y':
            df = collector.collect_continuous(
                patient_id=patient_id,
                duration_minutes=30,
                poll_interval_seconds=30,
                output_path="data/processed/ml_training_data.csv"
            )
            
            print(f"\n🎯 Next steps:")
            print(f"   1. Generate lab data: python quick_generate_data.py")
            print(f"   2. Or use this data directly for initial testing")
    
    else:
        print(f"Invalid choice")
    
    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()