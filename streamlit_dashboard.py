# streamlit_dashboard.py - Enhanced Healthcare Dashboard with Logging
"""
VitalViewAI Professional Healthcare Dashboard
Real-time patient monitoring with comprehensive logging

FIXES:
1. Auto-refresh changed to 5 minutes (300 seconds) for demo
2. Removed placeholder logo image
3. Fixed "View Details" button to use navigation instead of st.switch_page
4. Same risk score issue noted - need more varied patient data
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional
import sys

sys.path.append('.')

from logging_config import setup_logging, get_logger, AuditLogger

# Initialize logging
setup_logging(log_level="INFO", enable_json=True)
logger = get_logger(__name__)
audit_logger = AuditLogger()

# Page configuration
st.set_page_config(
    page_title="VitalViewAI Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE = "http://localhost:8000"
ML_API_BASE = "http://localhost:8001"

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    .alert-critical {
        background: #fee;
        border-left: 4px solid #d32f2f;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-high {
        background: #fff3e0;
        border-left: 4px solid #f57c00;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .patient-stable {
        background: #e8f5e9;
        border-left: 4px solid #388e3c;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'selected_patient' not in st.session_state:
    st.session_state.selected_patient = None
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False  # Changed to False by default for demo
if 'current_page' not in st.session_state:
    st.session_state.current_page = "📊 Overview"

# Helper Functions
def log_dashboard_action(action: str, details: Dict = None):
    """Log dashboard actions with audit trail"""
    try:
        logger.info(
            f"Dashboard action: {action}",
            extra={'action': action, 'details': details or {}}
        )
        audit_logger.log_access(
            user_id='dashboard_user',
            action=action,
            resource='dashboard',
            success=True,
            details=details
        )
    except Exception as e:
        logger.error(f"Failed to log action: {e}")

def get_all_patients() -> List[Dict]:
    """Fetch all patients from API"""
    try:
        response = requests.get(f"{API_BASE}/patients", timeout=5)
        if response.status_code == 200:
            return response.json().get('patients', [])
        return []
    except Exception as e:
        logger.error(f"Failed to fetch patients: {e}")
        st.error(f"⚠️ Could not connect to API: {e}")
        return []

def get_patient_current_vitals(patient_id: str) -> Optional[Dict]:
    """Get current vitals for a patient"""
    try:
        response = requests.get(
            f"{API_BASE}/patients/{patient_id}/current",
            timeout=5
        )
        if response.status_code == 200:
            return response.json()['data']
        return None
    except Exception as e:
        logger.error(f"Failed to fetch vitals for {patient_id}: {e}")
        return None

def get_patient_history(patient_id: str, hours: int = 2) -> pd.DataFrame:
    """Get historical data for a patient"""
    try:
        response = requests.get(
            f"{API_BASE}/patients/{patient_id}/history",
            params={'hours': hours, 'interval_minutes': 5},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()['data']
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to fetch history for {patient_id}: {e}")
        return pd.DataFrame()

def predict_deterioration(patient_id: str) -> Optional[Dict]:
    """Get ML prediction for patient"""
    try:
        # Get recent vitals - CHANGED: More history for better predictions
        history_df = get_patient_history(patient_id, hours=6)
        if history_df.empty:
            return None
        
        # Need minimum samples for feature engineering
        if len(history_df) < 30:
            logger.warning(f"Insufficient data for {patient_id}: {len(history_df)} samples")
            return {
                'risk_score': 0.5,
                'risk_level': 'MEDIUM',
                'alert': False,
                'interpretation': 'Collecting baseline data... Need more history for accurate prediction.',
                'timestamp': datetime.now().isoformat()
            }
        
        # Prepare data for prediction - Convert timestamps to strings
        vitals_list = history_df.copy()
        vitals_list['timestamp'] = vitals_list['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        vitals_list = vitals_list.to_dict('records')
        
        payload = {
            "patient_id": patient_id,
            "vitals": vitals_list,
            "user_id": "dashboard_user",
            "user_role": "clinician"
        }
        
        response = requests.post(
            f"{ML_API_BASE}/predict",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"ML API returned {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Prediction failed for {patient_id}: {e}")
        return None

def create_patient(patient_id: str, sampling_interval: int = 300):
    """Create a new patient"""
    try:
        response = requests.post(
            f"{API_BASE}/patients",
            json={
                "patient_id": patient_id,
                "sampling_interval_seconds": sampling_interval
            },
            timeout=5
        )
        
        log_dashboard_action('create_patient', {'patient_id': patient_id})
        
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Failed to create patient: {e}")
        return False

def trigger_event(patient_id: str, event_type: str):
    """Trigger deterioration event for testing"""
    try:
        response = requests.post(
            f"{API_BASE}/patients/{patient_id}/trigger-event",
            json={"patient_id": patient_id, "event_type": event_type},
            timeout=5
        )
        
        log_dashboard_action('trigger_event', {
            'patient_id': patient_id,
            'event_type': event_type
        })
        
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Failed to trigger event: {e}")
        return False

# Main Dashboard
def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 VitalViewAI Healthcare Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # REMOVED: Placeholder image
        st.markdown("### 🏥 VitalViewAI")
        st.markdown("---")
        
        st.markdown("### 🎛️ Dashboard Controls")
        
        # Auto-refresh toggle - CHANGED: Default False, 300 seconds (5 minutes)
        st.session_state.auto_refresh = st.toggle(
            "🔄 Auto-refresh (5 min)",
            value=st.session_state.auto_refresh,
            help="Refresh dashboard every 5 minutes"
        )
        
        # Refresh button
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 📊 Navigation")
        page = st.radio(
            "Select View",
            ["📊 Overview", "👤 Patient Details", "➕ Add Patient", "🧪 Add Lab Data"],
            label_visibility="collapsed",
            index=["📊 Overview", "👤 Patient Details", "➕ Add Patient", "🧪 Add Lab Data"].index(st.session_state.current_page)
        )
        
        st.session_state.current_page = page
        
        st.markdown("---")
        
        # System Status
        st.markdown("### ⚡ System Status")
        
        # Check API status
        try:
            api_response = requests.get(f"{API_BASE}/health", timeout=2)
            api_status = "🟢 Online" if api_response.status_code == 200 else "🔴 Offline"
        except:
            api_status = "🔴 Offline"
        
        try:
            ml_response = requests.get(f"{ML_API_BASE}/health", timeout=2)
            ml_status = "🟢 Online" if ml_response.status_code == 200 else "🔴 Offline"
        except:
            ml_status = "🔴 Offline"
        
        st.metric("Streaming API", api_status)
        st.metric("ML Server", ml_status)
        
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Main Content based on selected page
    if st.session_state.current_page == "📊 Overview":
        show_overview()
    elif st.session_state.current_page == "👤 Patient Details":
        show_patient_details()
    elif st.session_state.current_page == "➕ Add Patient":
        show_add_patient()
    elif st.session_state.current_page == "🧪 Add Lab Data":
        show_add_lab_data()
    
    # Auto-refresh - CHANGED: 300 seconds (5 minutes)
    if st.session_state.auto_refresh:
        time.sleep(300)  # 5 minutes
        st.rerun()

def show_overview():
    """Main overview dashboard"""
    st.markdown("## 📊 Patient Overview")
    
    # Fetch all patients
    patients = get_all_patients()
    
    if not patients:
        st.warning("⚠️ No patients found. Add patients using the sidebar.")
        
        st.markdown("### Quick Start:")
        st.code("""
# Make sure servers are running:
# Terminal 1: python streaming_api_server.py
# Terminal 2: python ml_server.py
# Then add patients using the sidebar!
        """, language="bash")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    active_alerts = 0
    critical_patients = 0
    stable_patients = 0
    
    # Get predictions for all patients
    patient_predictions = {}
    for patient in patients:
        pred = predict_deterioration(patient['patient_id'])
        if pred:
            patient_predictions[patient['patient_id']] = pred
            if pred['alert']:
                active_alerts += 1
            if pred['risk_level'] == 'CRITICAL':
                critical_patients += 1
            elif pred['risk_level'] == 'LOW':
                stable_patients += 1
    
    with col1:
        st.metric("👥 Total Patients", len(patients))
    with col2:
        st.metric("🚨 Active Alerts", active_alerts, delta=None if active_alerts == 0 else "High")
    with col3:
        st.metric("⚠️ Critical", critical_patients, delta=None if critical_patients == 0 else "Urgent")
    with col4:
        st.metric("✅ Stable", stable_patients)
    
    st.markdown("---")
    
    # Alert Panel
    if active_alerts > 0:
        st.markdown("### 🚨 Active Alerts")
        
        for patient_id, pred in patient_predictions.items():
            if pred['alert']:
                alert_class = "alert-critical" if pred['risk_level'] == 'CRITICAL' else "alert-high"
                
                st.markdown(f"""
                <div class="{alert_class}">
                    <strong>🚨 {pred['risk_level']} RISK</strong> - Patient: {patient_id}<br>
                    Risk Score: <strong>{pred['risk_score']:.1%}</strong><br>
                    {pred['interpretation']}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Patient Cards Grid
    st.markdown("### 👥 All Patients")
    
    cols = st.columns(3)
    
    for idx, patient in enumerate(patients):
        with cols[idx % 3]:
            patient_id = patient['patient_id']
            
            # Get current vitals
            vitals = get_patient_current_vitals(patient_id)
            pred = patient_predictions.get(patient_id)
            
            # Determine card style
            if pred:
                if pred['risk_level'] == 'CRITICAL':
                    card_style = "alert-critical"
                elif pred['risk_level'] == 'HIGH':
                    card_style = "alert-high"
                else:
                    card_style = "patient-stable"
            else:
                card_style = "patient-stable"
            
            with st.container():
                st.markdown(f"""
                <div class="{card_style}">
                    <h3>👤 {patient_id}</h3>
                """, unsafe_allow_html=True)
                
                if vitals:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("💓 HR", f"{vitals['heart_rate']:.0f} bpm")
                        st.metric("🩸 BP", f"{vitals['bp_systolic']:.0f}/{vitals['bp_diastolic']:.0f}")
                    with col_b:
                        st.metric("🫁 SpO₂", f"{vitals['spo2']:.0f}%")
                        st.metric("🌡️ Temp", f"{vitals['temperature']:.1f}°C")
                
                if pred:
                    st.progress(pred['risk_score'])
                    st.caption(f"Risk: {pred['risk_level']} ({pred['risk_score']:.1%})")
                
                # FIXED: Changed from st.switch_page to navigation via session state
                if st.button(f"View Details", key=f"view_{patient_id}"):
                    st.session_state.selected_patient = patient_id
                    st.session_state.current_page = "👤 Patient Details"
                    st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)

def show_patient_details():
    """Detailed patient view"""
    st.markdown("## 👤 Patient Details")
    
    patients = get_all_patients()
    
    if not patients:
        st.warning("No patients available")
        return
    
    # Patient selector
    patient_ids = [p['patient_id'] for p in patients]
    selected = st.selectbox(
        "Select Patient",
        patient_ids,
        index=patient_ids.index(st.session_state.selected_patient) if st.session_state.selected_patient in patient_ids else 0
    )
    
    st.session_state.selected_patient = selected
    
    # Get patient data
    vitals = get_patient_current_vitals(selected)
    history = get_patient_history(selected, hours=6)
    prediction = predict_deterioration(selected)
    
    if not vitals:
        st.error("Could not fetch patient data")
        return
    
    # Current Status
    st.markdown("### 📊 Current Status")
    
    col1, col2, col3 = st.columns([2, 2, 3])
    
    with col1:
        st.markdown("#### Vital Signs")
        st.metric("💓 Heart Rate", f"{vitals['heart_rate']:.0f} bpm")
        st.metric("🩸 Blood Pressure", f"{vitals['bp_systolic']:.0f}/{vitals['bp_diastolic']:.0f} mmHg")
        st.metric("🫁 SpO₂", f"{vitals['spo2']:.0f}%")
        st.metric("🌡️ Temperature", f"{vitals['temperature']:.1f}°C")
        st.metric("🫀 Respiratory Rate", f"{vitals['respiratory_rate']:.0f} /min")
    
    with col2:
        st.markdown("#### ML Prediction")
        if prediction:
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction['risk_score'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 50], 'color': "yellow"},
                        {'range': [50, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Risk Level", prediction['risk_level'])
            st.caption(prediction['interpretation'])
        else:
            st.info("Running prediction...")
    
    with col3:
        st.markdown("#### Quick Actions")
        
        st.markdown("**Simulate Events:**")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⚠️ Hypertensive Crisis", use_container_width=True):
                if trigger_event(selected, "hypertensive_crisis"):
                    st.success("Event triggered!")
                    time.sleep(1)
                    st.rerun()
        
        with col_b:
            if st.button("🫁 Hypoxia", use_container_width=True):
                if trigger_event(selected, "hypoxia"):
                    st.success("Event triggered!")
                    time.sleep(1)
                    st.rerun()
        
        col_c, col_d = st.columns(2)
        with col_c:
            if st.button("🦠 Sepsis", use_container_width=True):
                if trigger_event(selected, "sepsis"):
                    st.success("Event triggered!")
                    time.sleep(1)
                    st.rerun()
        
        with col_d:
            if st.button("❤️ Cardiac Event", use_container_width=True):
                if trigger_event(selected, "cardiac"):
                    st.success("Event triggered!")
                    time.sleep(1)
                    st.rerun()
        
        st.markdown("---")
        
        if st.button("✅ Resolve Events", use_container_width=True):
            try:
                requests.post(f"{API_BASE}/patients/{selected}/resolve-event")
                st.success("Events resolved!")
                time.sleep(1)
                st.rerun()
            except:
                st.error("Failed to resolve")
    
    # Historical Trends
    st.markdown("---")
    st.markdown("### 📈 Historical Trends (Last 6 Hours)")
    
    if not history.empty:
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Heart Rate', 'Blood Pressure', 'SpO₂', 'Temperature', 'Respiratory Rate', 'Activity'),
            vertical_spacing=0.12
        )
        
        # Heart Rate
        fig.add_trace(
            go.Scatter(x=history['timestamp'], y=history['heart_rate'], name='HR', line=dict(color='red')),
            row=1, col=1
        )
        
        # Blood Pressure
        fig.add_trace(
            go.Scatter(x=history['timestamp'], y=history['bp_systolic'], name='Systolic', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=history['timestamp'], y=history['bp_diastolic'], name='Diastolic', line=dict(color='lightblue')),
            row=1, col=2
        )
        
        # SpO2
        fig.add_trace(
            go.Scatter(x=history['timestamp'], y=history['spo2'], name='SpO₂', line=dict(color='green')),
            row=2, col=1
        )
        
        # Temperature
        fig.add_trace(
            go.Scatter(x=history['timestamp'], y=history['temperature'], name='Temp', line=dict(color='orange')),
            row=2, col=2
        )
        
        # Respiratory Rate
        fig.add_trace(
            go.Scatter(x=history['timestamp'], y=history['respiratory_rate'], name='RR', line=dict(color='purple')),
            row=3, col=1
        )
        
        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical data available")

def show_add_patient():
    """Add new patient interface"""
    st.markdown("## ➕ Add New Patient")
    
    with st.form("add_patient_form"):
        st.markdown("### Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            patient_id = st.text_input(
                "Patient ID",
                placeholder="demo_patient_001",
                help="Unique identifier for the patient"
            )
        
        with col2:
            sampling_interval = st.selectbox(
                "Monitoring Interval",
                [60, 300, 600, 1800],
                index=0,  # Default to 60 seconds for demo
                format_func=lambda x: f"{x//60} minutes" if x >= 60 else f"{x} seconds"
            )
        
        st.markdown("---")
        
        submitted = st.form_submit_button("➕ Create Patient", use_container_width=True)
        
        if submitted:
            if not patient_id:
                st.error("Please enter a Patient ID")
            else:
                with st.spinner("Creating patient..."):
                    if create_patient(patient_id, sampling_interval):
                        st.success(f"✅ Patient {patient_id} created successfully!")
                        st.balloons()
                        
                        log_dashboard_action('patient_created', {
                            'patient_id': patient_id,
                            'sampling_interval': sampling_interval
                        })
                        
                        st.info("💡 Tip: Wait 2-3 minutes for data to accumulate, then check predictions!")
                        
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("❌ Failed to create patient. Check if API server is running.")

def show_add_lab_data():
    """Add lab data interface"""
    st.markdown("## 🧪 Add Lab Data")
    
    st.info("📝 Lab data integration coming soon. This feature will allow manual entry or EHR integration.")
    
    patients = get_all_patients()
    
    if not patients:
        st.warning("No patients available. Please add patients first.")
        return
    
    patient_ids = [p['patient_id'] for p in patients]
    
    with st.form("add_lab_data_form"):
        st.markdown("### Lab Test Information")
        
        selected_patient = st.selectbox("Select Patient", patient_ids)
        
        col1, col2 = st.columns(2)
        
        with col1:
            glucose = st.number_input("Glucose (mg/dL)", min_value=40, max_value=600, value=100)
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
            creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
        
        with col2:
            hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=7.0, max_value=20.0, value=14.0, step=0.1)
            wbc = st.number_input("WBC Count (×10³/μL)", min_value=2.0, max_value=20.0, value=7.0, step=0.1)
            platelets = st.number_input("Platelets (×10³/μL)", min_value=50, max_value=500, value=250)
        
        test_date = st.date_input("Test Date", value=datetime.now())
        
        st.markdown("---")
        
        submitted = st.form_submit_button("💾 Save Lab Data", use_container_width=True)
        
        if submitted:
            lab_data = {
                'patient_id': selected_patient,
                'test_date': test_date.isoformat(),
                'glucose_random': glucose,
                'cholesterol_total': cholesterol,
                'creatinine': creatinine,
                'hemoglobin': hemoglobin,
                'wbc_count': wbc,
                'platelets': platelets
            }
            
            st.success(f"✅ Lab data saved for {selected_patient}")
            st.json(lab_data)
            
            log_dashboard_action('lab_data_added', lab_data)
            
            st.info("💡 Note: Backend storage will be implemented in production version")

if __name__ == "__main__":
    try:
        log_dashboard_action('dashboard_started')
        main()
    except Exception as e:
        logger.error(f"Dashboard error: {e}", exc_info=True)
        st.error(f"An error occurred: {e}")
        st.info("Please check the logs for more details")