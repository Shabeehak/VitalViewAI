# streamlit_dashboard.py - Enhanced Healthcare Dashboard with Authentication & Logging
"""
VitalViewAI Professional Healthcare Dashboard
Real-time patient monitoring with comprehensive logging and authentication

Features:
- JWT-based authentication
- Role-based access control
- Auto-refresh (5 minutes)
- Real-time patient monitoring
- ML predictions
- Comprehensive logging
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
import os

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
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
ML_API_BASE = os.getenv("ML_API_BASE", "http://localhost:8001")

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
    .login-container {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'token' not in st.session_state:
    st.session_state.token = None
if 'selected_patient' not in st.session_state:
    st.session_state.selected_patient = None
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "📊 Overview"
if 'show_registration' not in st.session_state:
    st.session_state.show_registration = False

# ==================== AUTHENTICATION FUNCTIONS ====================

def show_registration_page():
    """Display user registration page"""
    st.markdown('<h1 class="main-header">🏥 VitalViewAI Registration</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown("### 📝 Create New Account")
        
        with st.form("register_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            password_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
            
            full_name = st.text_input("Full Name", placeholder="Dr. Jane Doe")
            email = st.text_input("Email", placeholder="jane.doe@hospital.com")
            
            role = st.selectbox(
                "Role",
                options=["clinician", "nurse", "researcher"],
                help="Admin accounts must be created by existing admins"
            )
            
            st.info("ℹ️ Registration may take 5-10 seconds to complete. Please be patient.")
            
            submit = st.form_submit_button("✅ Register", use_container_width=True)
            
            if submit:
                # Validation
                if not username or not password or not full_name or not email:
                    st.error("⚠️ Please fill in all fields")
                    return
                
                if password != password_confirm:
                    st.error("❌ Passwords do not match")
                    return
                
                if len(password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                    return
                
                # Call registration API
                try:
                    with st.spinner("Creating account..."):
                        response = requests.post(
                            f"{ML_API_BASE}/register",
                            json={
                                "username": username,
                                "password": password,
                                "role": role,
                                "email": email,
                                "full_name": full_name
                            },
                            timeout=10  # Increased timeout for registration
                        )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Auto-login after registration
                        st.session_state.logged_in = True
                        st.session_state.user = result['user']
                        st.session_state.token = result['access_token']
                        
                        # Log the registration
                        log_dashboard_action('user_registration', {
                            'username': result['user']['username'],
                            'role': result['user']['role']
                        })
                        
                        st.success(f"✅ Account created successfully! Welcome, {full_name}!")
                        st.balloons()
                        
                        # Redirect to dashboard
                        time.sleep(2)
                        st.rerun()
                    
                    elif response.status_code == 400:
                        st.error("❌ Username already exists. Please choose another.")
                        log_dashboard_action('registration_failed', {'username': username, 'reason': 'username_exists'})
                    else:
                        st.error(f"❌ Registration failed: {response.text}")
                        log_dashboard_action('registration_failed', {'username': username, 'status_code': response.status_code})
                
                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Cannot connect to ML server. Make sure it's running:")
                    st.code("python ml_server.py", language="bash")
                except Exception as e:
                    st.error(f"❌ Registration failed: {str(e)}")
                    logger.error(f"Registration error: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Link to login
        st.markdown("---")
        if st.button("Already have an account? Login", use_container_width=True):
            st.session_state.show_registration = False
            st.rerun()
        
        st.markdown("---")
        st.info("💡 Tip: If servers aren't running, start them with:\n```bash\npython streaming_api_server.py\npython ml_server.py\n```")


def show_login_page():
    """Display login page"""
    st.markdown('<h1 class="main-header">🏥 VitalViewAI Login</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown("### 🔐 Sign In")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="dr_smith")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            
            st.info("ℹ️ Login may take 5-10 seconds. Please be patient.")
            
            submit = st.form_submit_button("🔓 Login", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("⚠️ Please enter username and password")
                    return
                
                # Call login API
                try:
                    with st.spinner("Authenticating..."):
                        response = requests.post(
                            f"{ML_API_BASE}/login",
                            data={
                                "username": username,
                                "password": password
                            },
                            timeout=10  # Increased timeout for login
                        )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Store in session state
                        st.session_state.logged_in = True
                        st.session_state.user = result['user']
                        st.session_state.token = result['access_token']
                        
                        # Log the login
                        log_dashboard_action('user_login', {
                            'username': result['user']['username'],
                            'role': result['user']['role']
                        })
                        
                        st.success(f"✅ Welcome, {result['user']['full_name']}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                        log_dashboard_action('login_failed', {'username': username})
                
                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Cannot connect to ML server. Make sure it's running:")
                    st.code("python ml_server.py", language="bash")
                except Exception as e:
                    st.error(f"❌ Login failed: {str(e)}")
                    logger.error(f"Login error: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ADD REGISTRATION LINK
        st.markdown("---")
        if st.button("Don't have an account? Register", use_container_width=True):
            st.session_state.show_registration = True
            st.rerun()
        
        # Show demo credentials
        st.markdown("---")
        st.markdown("### 👥 Demo Accounts")
        
        demo_users = [
            ("admin", "admin123", "Administrator", "Full system access"),
            ("dr_smith", "doctor123", "Clinician", "View predictions & patient data"),
            ("nurse_alice", "nurse123", "Nurse", "View patient data"),
            ("researcher", "research123", "Researcher", "View anonymized data")
        ]
        
        st.markdown("""
        <style>
        .demo-account {
            background: #f0f2f6;
            padding: 0.5rem;
            border-radius: 5px;
            margin: 0.3rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        for username, password, role, desc in demo_users:
            st.markdown(f"""
            <div class="demo-account">
                <strong>{role}:</strong> <code>{username}</code> / <code>{password}</code><br>
                <small>{desc}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("💡 Tip: If servers aren't running, start them with:\n```bash\npython streaming_api_server.py\npython ml_server.py\n```")


def show_logout_button():
    """Show logout button and user info in sidebar"""
    if st.session_state.get('logged_in', False):
        with st.sidebar:
            st.markdown("---")
            
            # User info
            user = st.session_state.get('user', {})
            if user:
                st.markdown(f"### 👤 User Profile")
                st.markdown(f"**Name:** {user.get('full_name', 'User')}")
                st.markdown(f"**Username:** {user.get('username', 'Unknown')}")
                
                role = user.get('role', 'Unknown')
                # Add role badge with color
                role_colors = {
                    'admin': '🔴',
                    'clinician': '🟢',
                    'nurse': '🔵',
                    'researcher': '🟡',
                    'viewer': '⚪'
                }
                role_badge = role_colors.get(role, '⚪')
                st.markdown(f"**Role:** {role_badge} {role.title()}")
                
                # Show key permissions
                with st.expander("📋 View Permissions"):
                    role_permissions = {
                        "admin": ["✅ All permissions", "✅ Manage users", "✅ System settings"],
                        "clinician": ["✅ Read/Write patient data", "✅ View predictions", "✅ Trigger alerts"],
                        "nurse": ["✅ Read patient data", "✅ View predictions", "✅ Acknowledge alerts"],
                        "researcher": ["✅ Read anonymized data", "✅ Export data", "✅ View metrics"],
                        "viewer": ["✅ Read patient data only"]
                    }
                    
                    for perm in role_permissions.get(role, ["❌ No permissions"]):
                        st.markdown(perm)
            
            if st.button("🚪 Logout", use_container_width=True):
                # Log the logout
                log_dashboard_action('user_logout', {
                    'username': user.get('username', 'unknown') if user else 'unknown',
                    'role': user.get('role', 'unknown') if user else 'unknown'
                })
                
                # Clear session
                st.session_state.logged_in = False
                st.session_state.user = None
                st.session_state.token = None
                st.success("✅ Logged out successfully!")
                time.sleep(1)
                st.rerun()


def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get('logged_in', False)


def check_permission(permission: str) -> bool:
    """
    Check if current user has specific permission
    
    Args:
        permission: Permission to check (e.g., 'read_patient_data', 'modify_patient_records')
    
    Returns:
        bool: True if user has permission
    """
    if not check_authentication():
        return False
    
    user = st.session_state.get('user', {})
    role = user.get('role', 'viewer')
    
    # Define permissions for each role (matches privacy_config.yaml)
    role_permissions = {
        "admin": [
            "read_patient_data",
            "write_patient_data",
            "view_predictions",
            "trigger_alerts",
            "modify_patient_records",
            "manage_users",
            "export_data",
            "view_audit_logs"
        ],
        "clinician": [
            "read_patient_data",
            "write_patient_data",
            "view_predictions",
            "trigger_alerts",
            "modify_patient_records"
        ],
        "nurse": [
            "read_patient_data",
            "view_predictions",
            "trigger_alerts"
        ],
        "researcher": [
            "read_anonymized_data",
            "export_anonymized_data",
            "view_model_metrics"
        ],
        "viewer": [
            "read_patient_data"
        ]
    }
    
    allowed = permission in role_permissions.get(role, [])
    
    if not allowed:
        logger.warning(
            f"Permission denied",
            extra={
                'user': user.get('username'),
                'role': role,
                'permission': permission
            }
        )
    
    return allowed


def get_user_role() -> str:
    """Get current user's role"""
    if not check_authentication():
        return "viewer"
    
    user = st.session_state.get('user', {})
    return user.get('role', 'viewer')


def get_auth_headers():
    """Get authentication headers for API calls"""
    token = st.session_state.get('token', '')
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }


def make_authenticated_request(method: str, url: str, **kwargs):
    """
    Make authenticated API request
    
    Args:
        method: HTTP method (GET, POST, etc.)
        url: Full URL
        **kwargs: Additional arguments for requests
    
    Returns:
        Response object
    """
    headers = get_auth_headers()
    
    if 'headers' in kwargs:
        kwargs['headers'].update(headers)
    else:
        kwargs['headers'] = headers
    
    try:
        response = requests.request(method, url, **kwargs)
        
        # Check for auth errors
        if response.status_code == 401:
            st.error("🔒 Session expired. Please login again.")
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.token = None
            time.sleep(2)
            st.rerun()
        
        return response
    
    except Exception as e:
        logger.error(f"API request error: {e}")
        raise

# ==================== HELPER FUNCTIONS ====================

def log_dashboard_action(action: str, details: Dict = None):
    """Log dashboard actions with audit trail"""
    try:
        user = st.session_state.get('user', None)
        username = user.get('username', 'anonymous') if user else 'anonymous'
        
        logger.info(
            f"Dashboard action: {action}",
            extra={
                'action': action,
                'details': details or {},
                'user': username
            }
        )
        audit_logger.log_access(
            user_id=username,
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
    """Get ML prediction for patient (with authentication)"""
    try:
        # Get recent vitals
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
        
        # Prepare data for prediction
        vitals_list = history_df.copy()
        vitals_list['timestamp'] = vitals_list['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        vitals_list = vitals_list.to_dict('records')
        
        payload = {
            "patient_id": patient_id,
            "vitals": vitals_list
        }
        
        # Make authenticated request
        response = make_authenticated_request(
            'POST',
            f"{ML_API_BASE}/predict",
            json=payload,
            timeout=15  # Increased timeout for ML predictions
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            st.warning("⚠️ You don't have permission to view predictions")
            return None
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

# ==================== MAIN DASHBOARD ====================

def main():
    """Main dashboard with authentication"""
    
    # Check authentication
    if not check_authentication():
        # Show registration or login based on session state
        if st.session_state.show_registration:
            show_registration_page()
        else:
            show_login_page()
        return
    
    # User is logged in - show main dashboard
    # Header
    st.markdown('<h1 class="main-header">🏥 VitalViewAI Healthcare Dashboard</h1>', unsafe_allow_html=True)
    
    # Role Banner
    user = st.session_state.get('user', {})
    role = user.get('role', 'viewer')
    role_display = {
        'admin': ('🔴 Administrator', 'Full system access'),
        'clinician': ('🟢 Clinician', 'Patient care and monitoring'),
        'nurse': ('🔵 Nurse', 'Patient monitoring'),
        'researcher': ('🟡 Researcher', 'Anonymized data only'),
        'viewer': ('⚪ Viewer', 'Read-only access')
    }
    
    badge, description = role_display.get(role, ('⚪ Unknown', 'Limited access'))
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 0.5rem 1rem; 
                border-radius: 5px; 
                margin-bottom: 1rem;
                text-align: center;">
        <span style="color: white; font-weight: bold;">{badge}</span> 
        <span style="color: #e0e0e0;">| {description}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🏥 VitalViewAI")
        st.markdown("---")
        
        st.markdown("### 🎛️ Dashboard Controls")
        
        # Auto-refresh toggle
        st.session_state.auto_refresh = st.toggle(
            "🔄 Auto-refresh (5 min)",
            value=st.session_state.auto_refresh,
            help="Refresh dashboard every 5 minutes"
        )
        
        # Refresh button
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
        
        st.markdown("---")
        
        # Navigation - Role-based
        st.markdown("### 📊 Navigation")
        
        # Build available pages based on permissions
        available_pages = ["📊 Overview"]  # Everyone can see overview
        
        if check_permission("read_patient_data"):
            available_pages.append("👤 Patient Details")
        
        if check_permission("write_patient_data"):
            available_pages.append("➕ Add Patient")
        
        if check_permission("write_patient_data"):
            available_pages.append("🧪 Add Lab Data")
        
        # Ensure current page is available
        if st.session_state.current_page not in available_pages:
            st.session_state.current_page = "📊 Overview"
        
        page = st.radio(
            "Select View",
            available_pages,
            label_visibility="collapsed",
            index=available_pages.index(st.session_state.current_page) if st.session_state.current_page in available_pages else 0
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
            ml_response = make_authenticated_request('GET', f"{ML_API_BASE}/health", timeout=2)
            ml_status = "🟢 Online" if ml_response.status_code == 200 else "🔴 Offline"
        except:
            ml_status = "🔴 Offline"
        
        st.metric("Streaming API", api_status)
        st.metric("ML Server", ml_status)
        
        # Show logout button and user info
        show_logout_button()
        
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
    
    # Auto-refresh
    if st.session_state.auto_refresh:
        time.sleep(300)  # 5 minutes
        st.rerun()


def show_overview():
    """Main overview dashboard"""
    st.markdown("## 📊 Patient Overview")
    
    # Check if user is researcher (should see anonymized data only)
    user_role = get_user_role()
    
    if user_role == "researcher":
        st.warning("🔬 **Researcher Access Mode**")
        st.info("""
        As a researcher, you have access to anonymized patient data only.
        
        **Available Features:**
        - ✅ View aggregated metrics
        - ✅ Export anonymized datasets
        - ✅ View model performance metrics
        - ❌ Patient identifiable information hidden
        """)
    
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
                
                # Only show "View Details" button if user has permission
                if check_permission("read_patient_data"):
                    if st.button(f"View Details", key=f"view_{patient_id}"):
                        st.session_state.selected_patient = patient_id
                        st.session_state.current_page = "👤 Patient Details"
                        st.rerun()
                else:
                    st.caption("🔒 View access restricted")
                
                st.markdown("</div>", unsafe_allow_html=True)


def show_patient_details():
    """Detailed patient view"""
    st.markdown("## 👤 Patient Details")
    
    # Check permission
    if not check_permission("read_patient_data"):
        st.error("🔒 **Access Denied**")
        st.warning("You don't have permission to view detailed patient data.")
        st.info(f"""
        **Your role:** {get_user_role()}  
        **Required permission:** read_patient_data
        
        Researchers can only access anonymized data.
        Please contact your administrator if you need access.
        """)
        return
    
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
        
        # Check if user can trigger events
        can_trigger = check_permission("trigger_alerts")
        
        if can_trigger:
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
        else:
            st.warning("🔒 **Access Restricted**")
            st.info("You don't have permission to trigger events.")
            st.caption(f"Your role: **{get_user_role()}**")
            st.caption("Required permission: **trigger_alerts**")
    
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
    
    # Check permission
    if not check_permission("write_patient_data"):
        st.error("🔒 **Access Denied**")
        st.warning("You don't have permission to add patients.")
        st.info(f"""
        **Your role:** {get_user_role()}  
        **Required permission:** write_patient_data
        
        Please contact your administrator if you need access.
        """)
        
        # Show role permissions
        st.markdown("### Your Permissions:")
        user = st.session_state.get('user', {})
        role = user.get('role', 'viewer')
        
        role_permissions = {
            "admin": ["All permissions"],
            "clinician": ["Read patient data", "Write patient data", "View predictions", "Trigger alerts", "Modify records"],
            "nurse": ["Read patient data", "View predictions", "Trigger alerts"],
            "researcher": ["Read anonymized data", "Export anonymized data", "View model metrics"],
            "viewer": ["Read patient data"]
        }
        
        for perm in role_permissions.get(role, []):
            st.markdown(f"✅ {perm}")
        
        return
    
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
                index=0,
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
    
    # Check permission
    if not check_permission("write_patient_data"):
        st.error("🔒 **Access Denied**")
        st.warning("You don't have permission to add lab data.")
        st.info(f"""
        **Your role:** {get_user_role()}  
        **Required permission:** write_patient_data
        
        Please contact your administrator if you need access.
        """)
        return
    
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