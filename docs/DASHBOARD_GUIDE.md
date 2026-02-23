# 📊 VitalViewAI Dashboard Guide
## Complete Frontend User Guide

---

## 🎯 Overview

The VitalViewAI Dashboard is a Streamlit-based real-time monitoring interface for healthcare professionals to monitor patient vitals and receive deterioration alerts.

**Access**: http://localhost:8501 (after running `streamlit run streamlit_dashboard.py`)

---

## 🚀 Starting the Dashboard

### Prerequisites
Make sure backend servers are running:

```powershell
# Windows
.\start_system.ps1

# Linux/Mac  
./start_system.sh
```

This starts:
- **Streaming API** (port 8000) - Patient data collection
- **ML Server** (port 8001) - Predictions

### Launch Dashboard

```powershell
streamlit run streamlit_dashboard.py
```

Browser opens automatically at http://localhost:8501

---

## 🎨 Dashboard Layout

### **Sidebar (Left Panel)**

```
┌─────────────────────────┐
│  🏥 VitalViewAI         │
├─────────────────────────┤
│  Dashboard Controls     │
│  ☐ Auto-refresh (5min)  │
│  🔄 Refresh Now         │
├─────────────────────────┤
│  Navigation             │
│  • 📊 Overview          │
│  • 👤 Patient Details   │
│  • ➕ Add Patient       │
│  • 🧪 Add Lab Data      │
├─────────────────────────┤
│  System Status          │
│  • API: 🟢 Online       │
│  • ML:  🟢 Online       │
├─────────────────────────┤
│  Last updated: 14:32:15 │
└─────────────────────────┘
```

**Controls:**
- **Auto-refresh**: Toggle 5-minute automatic data refresh
- **Refresh Now**: Manual refresh button
- **Navigation**: Switch between different views
- **System Status**: Backend health indicators

---

## 📊 Page 1: Overview Dashboard

**Purpose**: Monitor all patients at a glance, see active alerts

### Summary Metrics (Top)

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ 👥 Total    │ 🚨 Active   │ ⚠️ Critical │ ✅ Stable   │
│ Patients    │ Alerts      │ Patients    │ Patients    │
│    15       │     3       │     2       │     10      │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

### Active Alerts Panel

Shows patients with HIGH or CRITICAL risk levels:

```
🚨 CRITICAL RISK - Patient: demo_patient_002
Risk Score: 87%
Patient at high risk of deterioration in next 48 hours.
Immediate clinical assessment recommended.
```

**Color coding:**
- 🔴 **Red background** = CRITICAL (risk > 70%)
- 🟡 **Orange background** = HIGH (risk 50-70%)
- 🟢 **Green background** = STABLE (risk < 50%)

### Patient Cards Grid

3-column grid showing all patients:

```
┌──────────────────────────┐
│ 👤 patient_001          │
├──────────────────────────┤
│ 💓 HR    72 bpm         │
│ 🩸 BP    120/80 mmHg    │
│ 🫁 SpO₂  98%            │
│ 🌡️ Temp  36.8°C         │
├──────────────────────────┤
│ ████████░░ 83%          │
│ Risk: HIGH              │
├──────────────────────────┤
│ [View Details]          │
└──────────────────────────┘
```

**Click "View Details"** → Navigates to Patient Details page

---

## 👤 Page 2: Patient Details

**Purpose**: Deep dive into individual patient monitoring

### Patient Selector

```
Select Patient: [Dropdown Menu ▼]
├── demo_patient_001
├── demo_patient_002
└── demo_patient_003
```

### Current Status (Left Column)

**Vital Signs Display:**
```
💓 Heart Rate:       75 bpm
🩸 Blood Pressure:   122/78 mmHg
🫁 SpO₂:             97%
🌡️ Temperature:      36.9°C
🫀 Respiratory Rate: 16 /min
```

**Updated every 5 seconds when auto-refresh is on**

### ML Prediction (Middle Column)

**Risk Gauge:**
- Circular gauge showing 0-100% risk score
- Color-coded by severity:
  - Green (0-30%): Low risk
  - Yellow (30-50%): Medium risk
  - Orange (50-70%): High risk
  - Red (70-100%): Critical risk

**Risk Metrics:**
```
Risk Level: HIGH
Risk Score: 73.2%

Interpretation:
Patient at elevated risk of deterioration.
Close monitoring required.
```

### Quick Actions (Right Column)

**Simulate Events (Testing Only):**

```
⚠️ Hypertensive Crisis    🫁 Hypoxia
🦠 Sepsis                 ❤️ Cardiac Event

✅ Resolve Events
```

**How Event Simulation Works:**

1. **Click Event Button** → API call to `/patients/{id}/trigger-event`
2. **Backend triggers deterioration** → Patient vitals change
3. **Wait 5-10 seconds** → Model detects abnormal vitals
4. **Risk score increases** → Alert triggered
5. **Click "Resolve Events"** → Vitals return to normal

**Example Workflow:**
```
1. Select patient_001
2. Click "Hypertensive Crisis"
3. Watch BP increase: 120/80 → 195/110
4. Risk score increases: 15% → 85%
5. Alert appears: "CRITICAL RISK"
6. Click "Resolve Events"
7. BP returns to normal: 195/110 → 122/78
8. Risk decreases: 85% → 18%
```

### Historical Trends (Bottom)

**6-Hour Trend Charts:**

```
┌─────────────────────────────────────────┐
│ Heart Rate Trend                        │
│ 90┤        ╭──╮                          │
│ 80┤     ╭──╯  ╰──╮                       │
│ 70┤  ╭──╯         ╰──╮                   │
│ 60┼──╯                ╰──────────        │
│   └─────────────────────────────────────│
│   12:00  14:00  16:00  18:00  20:00     │
└─────────────────────────────────────────┘
```

**Charts shown:**
- Heart Rate over time
- Blood Pressure (systolic/diastolic)
- SpO₂ levels
- Temperature
- Respiratory Rate
- Activity State

**Patterns to look for:**
- **Increasing trend** = Potential deterioration
- **High variability** = Unstable patient
- **Sudden spikes** = Events occurring
- **Flat line** = Stable vitals

---

## ➕ Page 3: Add Patient

**Purpose**: Create new patient monitoring sessions

### Patient Creation Form

```
┌──────────────────────────────────────┐
│ Patient Information                  │
├──────────────────────────────────────┤
│ Patient ID: [text input]             │
│ Example: demo_patient_001            │
├──────────────────────────────────────┤
│ Monitoring Interval:                 │
│ • ◉ 1 minute                         │
│ • ○ 5 minutes                        │
│ • ○ 10 minutes                       │
│ • ○ 30 minutes                       │
├──────────────────────────────────────┤
│ Health Profile (Optional):           │
│ [Dropdown ▼]                         │
│ • Healthy                            │
│ • Hypertensive                       │
│ • Diabetic                           │
│ • Cardiac                            │
│ • Respiratory                        │
├──────────────────────────────────────┤
│ [➕ Create Patient]                  │
└──────────────────────────────────────┘
```

### How to Add a Patient

**Step 1**: Enter unique Patient ID
```
Good: demo_patient_005, john_doe_001, patient_cardiac_003
Bad: test, 123 (too simple)
```

**Step 2**: Choose monitoring interval
- **1 minute**: Intensive monitoring (ICU patients)
- **5 minutes**: Standard monitoring (default)
- **10 minutes**: Less critical patients
- **30 minutes**: Routine monitoring

**Step 3**: Select health profile (optional)
- **Healthy**: Normal baseline vitals
- **Hypertensive**: Elevated BP baseline
- **Diabetic**: Glucose monitoring focus
- **Cardiac**: Heart-focused monitoring
- **Respiratory**: Lung function focus

**Step 4**: Click "Create Patient"

**What Happens:**
1. API creates patient record
2. Wearable simulator starts generating vitals
3. Data streams every [interval] seconds
4. Patient appears in Overview page
5. ML model starts monitoring

**Success Message:**
```
✅ Patient demo_patient_005 created successfully!

💡 Tip: Wait 2-3 minutes for data to accumulate,
   then check predictions!
```

---

## 🧪 Page 4: Add Lab Data

**Purpose**: Manually enter laboratory test results

### Lab Data Form

```
┌──────────────────────────────────────┐
│ Lab Test Information                 │
├──────────────────────────────────────┤
│ Select Patient: [Dropdown ▼]         │
├──────────────────────────────────────┤
│ Test Results:                        │
│                                      │
│ Glucose:        [100  ] mg/dL       │
│ Cholesterol:    [200  ] mg/dL       │
│ Creatinine:     [1.0  ] mg/dL       │
│ Hemoglobin:     [14.0 ] g/dL        │
│ WBC Count:      [7.0  ] ×10³/μL     │
│ Platelets:      [250  ] ×10³/μL     │
├──────────────────────────────────────┤
│ Test Date: [2026-01-26]             │
├──────────────────────────────────────┤
│ [💾 Save Lab Data]                  │
└──────────────────────────────────────┘
```

**Normal Ranges Reference:**
| Test | Normal Range | Units |
|------|--------------|-------|
| Glucose | 70-100 | mg/dL |
| Cholesterol | 125-200 | mg/dL |
| Creatinine | 0.7-1.3 | mg/dL |
| Hemoglobin | 13.5-17.5 | g/dL |
| WBC | 4.5-11.0 | ×10³/μL |
| Platelets | 150-400 | ×10³/μL |

**Note**: This feature demonstrates the interface but doesn't currently integrate with the ML model. Planned for future versions.

---

## 🔄 Data Flow: How Streaming Works

### Real-Time Data Pipeline

```
┌──────────────┐
│   Patient    │
│  Created     │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────┐
│  Wearable Device Simulator       │
│  • Generates vitals every 60s    │
│  • Stores in memory              │
│  • Serves via API endpoint       │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  Dashboard (Auto-refresh)        │
│  • Every 5 minutes               │
│  • Fetches /patients/{id}/current│
│  • Displays updated vitals       │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  ML Prediction Request           │
│  • Gets last 6 hours of data     │
│  • Creates 133 features          │
│  • XGBoost predicts risk         │
│  • Returns risk score 0-100%     │
└──────────────────────────────────┘
```

### How to See Live Streaming

**Step 1**: Create patient
```powershell
curl -X POST "http://localhost:8000/patients" \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "stream_test_001", "sampling_interval_seconds": 60}'
```

**Step 2**: Watch data stream
```powershell
# Get current reading (refreshes every 60 seconds)
curl "http://localhost:8000/patients/stream_test_001/current"
```

**Step 3**: Dashboard shows updates
- Enable auto-refresh
- Watch vitals update every 5 minutes
- Risk score recalculated each time

---

## 🎮 Complete User Workflows

### Workflow 1: Monitor Existing Patient

```
1. Open dashboard (http://localhost:8501)
2. Navigate to "📊 Overview"
3. View all patient cards
4. Identify HIGH/CRITICAL risk patients (red/orange cards)
5. Click "View Details" on concerning patient
6. Review:
   - Current vitals
   - Risk score
   - 6-hour trends
7. Take clinical action if needed
```

### Workflow 2: Add New Patient

```
1. Navigate to "➕ Add Patient"
2. Enter Patient ID: "new_patient_001"
3. Select interval: "5 minutes"
4. Select profile: "Healthy"
5. Click "Create Patient"
6. Wait 2-3 minutes for data collection
7. Navigate to "📊 Overview"
8. Find new patient card
9. Click "View Details"
10. Verify vitals are streaming
```

### Workflow 3: Test Deterioration Detection

```
1. Navigate to "👤 Patient Details"
2. Select patient: "demo_patient_001"
3. Note current risk score (e.g., 15%)
4. Click "⚠️ Hypertensive Crisis"
5. Wait 10 seconds
6. Click "🔄 Refresh" in sidebar
7. Observe:
   - BP increased (120/80 → 195/110)
   - Risk score increased (15% → 85%)
   - Alert appears in Overview
8. Click "✅ Resolve Events"
9. Wait 10 seconds, refresh
10. Verify vitals normalized
```

### Workflow 4: Monitor Trends

```
1. Patient Details page
2. Scroll to "Historical Trends"
3. Observe 6-hour charts
4. Look for patterns:
   - Gradual increase? → Slow deterioration
   - Sudden spike? → Acute event
   - High variability? → Unstable patient
   - Flat line? → Stable condition
5. Use trends to predict future deterioration
```

---

## ⚙️ Dashboard Settings & Configuration

### Auto-Refresh Settings

**How it works:**
- **Enabled**: Dashboard updates every 5 minutes automatically
- **Disabled**: Manual refresh only (click 🔄 button)

**When to use:**
- **Enable**: Active monitoring of critical patients
- **Disable**: System demos, presentations

**Performance impact:**
- Enabled: Higher API calls, more bandwidth
- Disabled: Lower resource usage

### System Status Indicators

**🟢 Online**: Service responding to health checks
**🔴 Offline**: Service down or unreachable

**If services are offline:**
```powershell
# Check if backend is running
curl http://localhost:8000/health
curl http://localhost:8001/health

# If not running, start them
.\start_system.ps1  # Windows
./start_system.sh   # Linux/Mac
```

---

## 🐛 Troubleshooting

### Dashboard won't load

**Symptom**: Browser shows "Can't reach this page"

**Solution**:
```powershell
# Kill any existing Streamlit processes
# Then restart
streamlit run streamlit_dashboard.py
```

### "Could not connect to API" error

**Symptom**: Dashboard loads but shows connection errors

**Solution**:
```powershell
# Check backend status
curl http://localhost:8000/health
curl http://localhost:8001/health

# If failed, restart backends
.\start_system.ps1
```

### Patients not showing up

**Symptom**: Created patient doesn't appear in list

**Solution**:
1. Wait 5 seconds after creation
2. Click "🔄 Refresh Now"
3. Check backend logs: `logs/application.log`

### Risk scores all the same

**Symptom**: All patients show similar risk (e.g., 50%)

**Cause**: Insufficient data accumulation

**Solution**:
1. Wait 5-10 minutes for data collection
2. Trigger events to create variation
3. Check if streaming is active:
   ```powershell
   curl http://localhost:8000/patients/patient_001/current
   ```

### Charts not updating

**Symptom**: Historical trends frozen

**Solution**:
1. Enable auto-refresh
2. Manual refresh: Click 🔄
3. Check if patient is generating data:
   ```powershell
   curl http://localhost:8000/patients/patient_001/history?hours=6
   ```

---

## 📊 Understanding Risk Scores

### Risk Score Calculation

```
Input: Last 6 hours of patient data
       ↓
Feature Engineering: 133 features created
       ↓
XGBoost Model: Predicts probability
       ↓
Risk Score: 0-100% (probability × 100)
```

### Risk Level Thresholds

| Score | Level | Action |
|-------|-------|--------|
| 0-30% | **LOW** | Routine monitoring |
| 30-50% | **MEDIUM** | Increased monitoring |
| 50-70% | **HIGH** | Close monitoring, notify clinician |
| 70-100% | **CRITICAL** | Immediate assessment required |

### What Influences Risk Score

**Increases risk:**
- Abnormal vitals (very high/low)
- Increasing trends over time
- High vital variability
- Multiple abnormal vitals simultaneously

**Decreases risk:**
- Normal, stable vitals
- Decreasing trends (if were high)
- Low variability
- Good SpO₂, normal temp

---

## 🎯 Best Practices

### For Demonstrations

1. **Create 3-5 patients** with different profiles
2. **Wait 5 minutes** for data accumulation
3. **Trigger 1-2 events** to show alerts
4. **Show trend charts** to demonstrate monitoring
5. **Explain trade-offs** (71.8% precision, 46.5% recall at default threshold)

### For Development

1. **Disable auto-refresh** to save resources
2. **Use manual refresh** when testing changes
3. **Check logs** for debugging: `logs/application.log`
4. **Test with 1 patient** first before scaling

### For Production (Hypothetical)

1. **Enable auto-refresh** for continuous monitoring
2. **Set up alerts** (email/SMS for critical events)
3. **Monitor system health** regularly
4. **Regular model retraining** with new data

---

## 🔗 Related Documentation

- [Main README](README.md) - Project overview
- [Data Documentation](DATA_DOCUMENTATION.md) - Data pipeline
- [Architecture Guide](ARCHITECTURE.md) - System design
- [Model Evaluation](MODEL_EVALUATION.md) - Performance metrics

---

**Questions?** Check the troubleshooting section or open an issue on GitHub.

**Dashboard Version**: 1.0.0  
**Last Updated**: January 2026