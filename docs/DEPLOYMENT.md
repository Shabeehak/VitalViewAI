# 🚀 VitalViewAI Deployment Guide

**Complete guide for deploying VitalViewAI from local development to production cloud hosting**

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment (Local)](#docker-deployment-local)
4. [Cloud Deployment (Production)](#cloud-deployment-production)
5. [Environment Configuration](#environment-configuration)
6. [Troubleshooting](#troubleshooting)
7. [Performance Optimization](#performance-optimization)
8. [Security Considerations](#security-considerations)
9. [CI/CD Pipeline](#cicd-pipeline)
10. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Prerequisites

### System Requirements

**Minimum:**
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 2GB free space
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 20.04+

**Recommended:**
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 5GB free space
- **GPU**: Not required (CPU-only XGBoost)

### Software Requirements

```bash
# Required
Python 3.11 or higher
Git
Docker Desktop (for containerized deployment)

# Optional but Recommended
Visual Studio Code
Postman (for API testing)
PostgreSQL (for production)
```

### Installation Verification

```bash
# Check Python version
python --version  # Should be 3.11+

# Check pip
pip --version

# Check Docker
docker --version
docker-compose --version

# Check Git
git --version
```

---

## Local Development Setup

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/Shabeehak/VitalViewAI.git
cd VitalViewAI

# Verify files
ls -la
# Should see: requirements.txt, docker files, .py files
```

### Step 2: Create Virtual Environment

**Windows:**
```powershell
# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\activate

# Verify activation (should see (venv) in prompt)
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Verify activation
which python  # Should show path with 'venv'
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep xgboost  # Should show xgboost 3.1.2
pip list | grep fastapi  # Should show fastapi 0.128.0
pip list | grep streamlit  # Should show streamlit 1.52.2
```

**Common Installation Issues:**

```bash
# If you get SSL errors
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# If xgboost fails on Windows
pip install xgboost --no-cache-dir

# If scikit-learn fails
pip install scikit-learn --no-build-isolation
```

### Step 4: Generate Data & Train Model

```bash
# Generate training data (takes 5-10 minutes)
python generate_diverse_training_data.py
# Creates: data/processed/features_multi.csv (1.1M samples)

# Train model (takes 3-5 minutes)
python train_models_improved.py
# Creates: models/xgboost_model.pkl
#          models/xgboost_model_metadata.json

# Verify model exists
ls -la models/
# Should see: xgboost_model.pkl (large file)
```

### Step 5: Start Services (Development)

**Option A: Using Startup Scripts**

**Windows:**
```powershell
# Start all services
.\start_system.ps1

# Services will start on:
# - Streaming API: http://localhost:8000
# - ML Server: http://localhost:8001
# - Dashboard: http://localhost:8501

# Stop services
.\stop_system.ps1
```

**macOS/Linux:**
```bash
# Make script executable
chmod +x start_system.sh

# Start all services
./start_system.sh

# Stop services (Ctrl+C or)
./stop_system.sh
```

**Option B: Manual Start (for debugging)**

Open 3 terminal windows:

**Terminal 1 - Streaming API:**
```bash
cd VitalViewAI
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

python streaming_api_server.py
# Started on: http://localhost:8000
```

**Terminal 2 - ML Server:**
```bash
cd VitalViewAI
venv\Scripts\activate

python ml_server.py
# Started on: http://localhost:8001
```

**Terminal 3 - Dashboard:**
```bash
cd VitalViewAI
venv\Scripts\activate

streamlit run streamlit_dashboard.py
# Started on: http://localhost:8501
```

### Step 6: Verify Installation

**Test Checklist:**
```bash
# 1. Check API Server
curl http://localhost:8000/health
# Should return: {"status": "healthy"}

# 2. Check ML Server
curl http://localhost:8001/health
# Should return: {"status": "healthy", "model_loaded": true}

# 3. Open Dashboard
# Open browser: http://localhost:8501
# Should see: Login page

# 4. Test Login
# Username: dr_smith
# Password: doctor123
# Should see: Dashboard with patient list
```

---

## Docker Deployment (Local)

### Why Docker?

✅ **Consistent environment** across all machines  
✅ **Isolates dependencies** from host system  
✅ **Production-ready** deployment  
✅ **Easy scaling** and replication  

### Architecture

```
Docker Container (vitalviewai):
├── Supervisor (Process Manager)
│   ├── API Server (localhost:8000)
│   ├── ML Server (localhost:8001)
│   └── Dashboard (0.0.0.0:8501) ← Public
├── Shared Volumes:
│   ├── ./data (patient data)
│   ├── ./models (ML models)
│   └── ./logs (application logs)
└── Port Mapping: 8501 → 8501
```

### Step 1: Verify Docker Installation

```bash
# Check Docker
docker --version
# Should show: Docker version 20.10+

# Check Docker Compose
docker-compose --version
# Should show: Docker Compose version 2.0+

# Test Docker
docker run hello-world
# Should complete without errors
```

### Step 2: Build Docker Image

**Using Unified Dockerfile (Recommended):**

```bash
# Ensure you're in project root
cd VitalViewAI

# Build image
docker build -f Dockerfile -t vitalviewai:latest .

# This will:
# 1. Install Python dependencies (takes 5-10 minutes)
# 2. Copy application files
# 3. Set up supervisor configuration
# 4. Configure environment variables

# Verify image
docker images | grep vitalviewai
# Should show: vitalviewai  latest  <image-id>  <size>
```

### Step 3: Run Container

```bash
# Run container
docker run -d \
  --name vitalviewai \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  vitalviewai:latest

# Flags explained:
# -d: Run in background (detached)
# --name: Container name
# -p: Port mapping (host:container)
# -v: Volume mounts (persist data)

# Check container status
docker ps
# Should show container running

# View logs
docker logs -f vitalviewai
# Should see:
# - API Server started
# - ML Server started
# - Dashboard started
```

### Step 4: Access Application

```bash
# Open browser
http://localhost:8501

# Login
# Username: dr_smith
# Password: doctor123
```

### Docker Management Commands

```bash
# View running containers
docker ps

# View all containers (including stopped)
docker ps -a

# Stop container
docker stop vitalviewai

# Start container
docker start vitalviewai

# Restart container
docker restart vitalviewai

# View logs
docker logs vitalviewai
docker logs -f vitalviewai  # Follow logs

# Execute command in container
docker exec -it vitalviewai bash
# Now you're inside the container

# Remove container
docker stop vitalviewai
docker rm vitalviewai

# Remove image
docker rmi vitalviewai:latest

# Clean up everything (careful!)
docker system prune -a
```

### Troubleshooting Docker Build

**Issue: Build fails at dependency installation**
```bash
# Clear Docker cache
docker builder prune -a

# Rebuild without cache
docker build --no-cache -f Dockerfile -t vitalviewai:latest .
```

**Issue: Container exits immediately**
```bash
# Check logs for errors
docker logs vitalviewai

# Common issues:
# - Model file missing: Download from Google Drive
# - Port already in use: Change port mapping
# - Permission denied: Check volume mounts
```

**Issue: Can't access dashboard at localhost:8501**
```bash
# Check if port is mapped correctly
docker ps
# Should show: 0.0.0.0:8501->8501/tcp

# Check if services are running inside container
docker exec vitalviewai ps aux
# Should show: supervisord, uvicorn, streamlit processes

# Check container logs
docker logs vitalviewai | grep -i error
```

---

## Cloud Deployment (Production)

### Deployment Options Comparison

| Platform | Difficulty | Cost | Best For |
|----------|-----------|------|----------|
| **Render** | ⭐ Easy | Free tier | Portfolio projects, demos |
| **Heroku** | ⭐⭐ Medium | $7+/month | Startups, prototypes |
| **DigitalOcean** | ⭐⭐ Medium | $5+/month | Small businesses |
| **AWS ECS** | ⭐⭐⭐⭐ Hard | $50+/month | Enterprise |
| **Google Cloud Run** | ⭐⭐⭐ Medium | Pay-per-use | Variable traffic |
| **Azure** | ⭐⭐⭐⭐ Hard | $30+/month | Enterprise |

**Recommended for VitalViewAI**: **Render** (best balance of ease + features)

---

## Render Deployment (Recommended)

### Why Render?

✅ **Free tier** with 750 hours/month  
✅ **Auto-deploy** from GitHub  
✅ **Automatic HTTPS**  
✅ **Simple setup** (5 minutes)  
✅ **No credit card** required for free tier  

### Architecture on Render

```
Render Web Service (vitalviewai):
├── Automatic build from GitHub
├── Docker container running all 3 services
├── Public URL: https://vitalviewai-xxxx.onrender.com
├── Health checks: /_stcore/health
└── Auto-restart on failure
```

### Prerequisites

1. **GitHub Account** (free)
2. **Render Account** (free - sign up at render.com)
3. **Code pushed to GitHub**

### Step 1: Prepare Your Repository

**Ensure these files exist in your repo:**

```bash
VitalViewAI/
├── Dockerfile              # ✅ REQUIRED - Unified deployment
├── requirements.txt        # ✅ REQUIRED - Python dependencies
├── .gitignore             # ✅ REQUIRED - Exclude large files
├── streaming_api_server.py # ✅ REQUIRED
├── ml_server.py           # ✅ REQUIRED
├── streamlit_dashboard.py # ✅ REQUIRED
├── predictor.py           # ✅ REQUIRED
├── models/
│   └── xgboost_model_metadata.json  # ✅ REQUIRED (small file)
└── src/                   # ✅ REQUIRED (all source files)
```

**Update .gitignore:**
```bash
# Large files - DO NOT push to GitHub
*.pkl
*.h5
*.joblib
data/processed/*.csv
models/xgboost_model.pkl

# Keep metadata
!models/*.json
!models/*.png
```

**Important: Model File Handling**

Your `predictor.py` should download the model from Google Drive:

```python
def _download_model_from_gdrive(self, model_path: str):
    """Download model from Google Drive if not present"""
    import requests
    import os
    
    # Your Google Drive file ID
    file_id = "1lnmxCHDiCCS1ro__Iwtqn6mT6tqXQbYH"
    
    # Google Drive direct download URL
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    logger.info(f"Downloading model from Google Drive...")
    
    # Download and save
    # ... (implementation in predictor.py)
```

### Step 2: Push to GitHub

```bash
# Add all files
git add .

# Commit
git commit -m "Prepare for Render deployment"

# Push to GitHub
git push origin main

# Verify on GitHub
# Go to: github.com/your-username/VitalViewAI
# Should see all files except large .pkl files
```

### Step 3: Create Render Web Service

**Method 1: Via Dashboard (Easiest)**

1. **Go to**: [render.com/dashboard](https://render.com/dashboard)
2. **Click**: "New +" → "Web Service"
3. **Connect GitHub**: 
   - Click "Connect Account"
   - Authorize Render
   - Select `VitalViewAI` repository
4. **Configure Service**:
   ```
   Name: vitalviewai
   Region: Oregon (or closest to you)
   Branch: main
   Root Directory: (leave blank)
   Environment: Docker
   Dockerfile Path: ./Dockerfile
   Build Command: (leave blank - Docker handles it)
   Start Command: (leave blank - Docker handles it)
   ```
5. **Select Plan**: Free
6. **Advanced Settings** (expand):
   ```
   Health Check Path: /_stcore/health
   Auto-Deploy: Yes
   ```
7. **Click**: "Create Web Service"

**Method 2: Via render.yaml (Infrastructure as Code)**

Create `render.yaml` in project root:

```yaml
services:
  - type: web
    name: vitalviewai
    env: docker
    dockerfilePath: ./Dockerfile
    region: oregon
    plan: free
    healthCheckPath: /_stcore/health
    envVars:
      - key: ENVIRONMENT
        value: production
      - key: LOG_LEVEL
        value: INFO
      - key: MODEL_PATH
        value: /app/models/xgboost_model.pkl
```

Then:
1. Push `render.yaml` to GitHub
2. Go to Render Dashboard
3. Click "New +" → "Blueprint"
4. Select your repository
5. Click "Apply"

### Step 4: Monitor Deployment

**Deployment Timeline:**
```
t=0:00  Build started
t=0:30  Installing system dependencies
t=2:00  Installing Python packages (requirements.txt)
t=7:00  Copying application files
t=7:30  Building Docker image
t=8:00  Pushing to Render registry
t=8:30  Starting container
t=9:00  Downloading model from Google Drive (2-3 min)
t=11:00 Services starting (API, ML, Dashboard)
t=12:00 🎉 Deployment complete!
```

**Watch the logs:**
- Click on your service in Render dashboard
- Go to "Logs" tab
- Look for these messages:
  ```
  📥 Downloading model from Google Drive...
  ✅ Model downloaded successfully!
  ✅ Model loaded successfully
  [program:api-server] INFO: Started server
  [program:ml-server] INFO: Started server
  [program:dashboard] You can now view your Streamlit app
  ```

### Step 5: Access Your Deployed App

**Your URL:**
```
https://vitalviewai-xxxx.onrender.com
```
(Render auto-generates the xxxx part)

**First Load:**
- ⏱️ Takes 30-60 seconds (free tier sleeps after inactivity)
- After first wake-up, subsequent loads are fast

**Test Login:**
```
Username: dr_smith
Password: doctor123
```

### Step 6: Custom Domain (Optional)

**If you own a domain:**

1. Go to Render Dashboard → Your Service → Settings
2. Scroll to "Custom Domains"
3. Click "Add Custom Domain"
4. Enter: `vitalviewai.yourdomain.com`
5. Add CNAME record in your DNS:
   ```
   Type: CNAME
   Name: vitalviewai
   Value: vitalviewai-xxxx.onrender.com
   ```
6. Wait for DNS propagation (5-60 minutes)
7. Access at: `https://vitalviewai.yourdomain.com`

---

## Environment Configuration

### Environment Variables

**Development (.env file):**

Create `.env` in project root:

```bash
# Server Configuration
ENVIRONMENT=development
LOG_LEVEL=DEBUG
API_PORT=8000
ML_PORT=8001
DASHBOARD_PORT=8501

# Model Configuration
MODEL_TYPE=xgboost
MODEL_PATH=models/xgboost_model.pkl

# Database (if using PostgreSQL)
DATABASE_URL=postgresql://user:password@localhost:5432/vitalview

# Security
SECRET_KEY=your-secret-key-here-change-in-production
JWT_EXPIRATION_HOURS=8
ENCRYPTION_KEY=generate-with-cryptography-fernet

# External Services (v2.0)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
TWILIO_PHONE_NUMBER=+1234567890

# Monitoring
SENTRY_DSN=your-sentry-dsn
PROMETHEUS_PORT=9090
```

**Production (Render Environment Variables):**

In Render Dashboard → Your Service → Environment:

```
ENVIRONMENT=production
LOG_LEVEL=INFO
MODEL_PATH=/app/models/xgboost_model.pkl
SECRET_KEY=<generate-secure-key>
```

**Generate Secure Keys:**

```python
# In Python:
from cryptography.fernet import Fernet

# Generate encryption key
print(Fernet.generate_key().decode())

# Generate secret key
import secrets
print(secrets.token_urlsafe(32))
```

### Configuration Management

**config.py:**

```python
import os
from pathlib import Path

class Config:
    """Base configuration"""
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Ports
    API_PORT = int(os.getenv("API_PORT", 8000))
    ML_PORT = int(os.getenv("ML_PORT", 8001))
    DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", 8501))
    
    # Model
    MODEL_TYPE = os.getenv("MODEL_TYPE", "xgboost")
    MODEL_PATH = os.getenv("MODEL_PATH", "models/xgboost_model.pkl")
    
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

# Load appropriate config
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig
}

current_config = config[os.getenv('ENVIRONMENT', 'development')]
```

---

## Troubleshooting

### Common Issues & Solutions

#### Issue 1: "Cannot connect to ML server"

**Symptoms:**
- Dashboard shows: "⚠️ Cannot connect to ML server"
- API requests fail with 502/503 errors

**Causes & Fixes:**

**A. Services not running:**
```bash
# Check if services are running
docker ps
# Should show container with status "Up"

# Check logs
docker logs vitalviewai | grep -i "started server"
# Should see all 3 services started

# Restart if needed
docker restart vitalviewai
```

**B. Model still downloading:**
```bash
# Check logs for model download
docker logs vitalviewai | grep -i "downloading model"
# If downloading, wait 2-3 minutes

# Check if model loaded
docker logs vitalviewai | grep -i "model loaded successfully"
# Should see: ✅ Model loaded successfully
```

**C. Wrong environment variables:**
```bash
# Check environment inside container
docker exec vitalviewai env | grep API

# Should show:
# API_BASE=http://127.0.0.1:8000
# ML_API_BASE=http://127.0.0.1:8001

# If wrong, rebuild with correct Dockerfile
```

---

#### Issue 2: "502 Bad Gateway" on Render

**Symptoms:**
- Render shows service as "Live" but browser shows 502

**Causes & Fixes:**

**A. Services still starting:**
```
# First deployment takes 10-15 minutes
# Wait and watch logs in Render dashboard
# Look for: "You can now view your Streamlit app"
```

**B. Health check failing:**
```yaml
# In render.yaml or dashboard settings:
healthCheckPath: /_stcore/health

# Test locally:
curl http://localhost:8501/_stcore/health
# Should return 200 OK
```

**C. Port configuration wrong:**
```dockerfile
# In Dockerfile, dashboard must listen on 0.0.0.0
# Check line:
command=streamlit run streamlit_dashboard.py --server.address=0.0.0.0 --server.port=8501

# API and ML servers should listen on 127.0.0.1 (localhost only):
command=uvicorn ml_server:app --host 127.0.0.1 --port 8001
```

---

#### Issue 3: "Model file not found"

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/xgboost_model.pkl'
```

**Fixes:**

**A. Verify Google Drive download code in predictor.py:**
```python
def __init__(self, model_path: str = "models/xgboost_model.pkl"):
    # Check if model exists, download if not
    if not os.path.exists(model_path):
        logger.warning(f"Model not found, downloading from Google Drive...")
        self._download_model_from_gdrive(model_path)
    
    # Load model
    self.model = joblib.load(model_path)
```

**B. Verify Google Drive link is public:**
```
1. Go to Google Drive
2. Right-click file → Share
3. Change to "Anyone with the link"
4. Copy file ID from URL
```

**C. Test download manually:**
```python
# Test in Python
import requests
file_id = "1lnmxCHDiCCS1ro__Iwtqn6mT6tqXQbYH"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
response = requests.get(url)
print(response.status_code)  # Should be 200
```

---

#### Issue 4: "Out of Memory" Error

**Symptoms:**
```
MemoryError: Unable to allocate array
Container killed (OOMKilled)
```

**Fixes:**

**A. Free tier limits:**
```
Render Free Tier: 512MB RAM
VitalViewAI needs: ~800MB with model loaded

Solution: Upgrade to Starter plan ($7/month, 2GB RAM)
```

**B. Optimize memory usage:**
```python
# In predictor.py, load model lazily
class HealthPredictor:
    def __init__(self):
        self.model = None  # Don't load immediately
    
    def predict(self, data):
        if self.model is None:
            self.model = joblib.load(self.model_path)
        # ... prediction logic
```

**C. Use smaller model:**
```python
# Re-train with fewer trees
XGBClassifier(
    n_estimators=100,  # Reduced from 500
    max_depth=4,       # Reduced from 6
    # ... other params
)
# Model size: 1.7GB → 500MB
```

---

#### Issue 5: Slow First Load

**Symptoms:**
- Render app takes 30-60 seconds to respond on first request
- Shows "Application Error" initially

**This is NORMAL for free tier:**
```
Free tier apps sleep after 15 minutes of inactivity
First request wakes the app (30-60 seconds)
Subsequent requests are fast
```

**Solutions:**

**A. Keep-alive ping (free):**
```python
# Use external service: cron-job.org
# Set up HTTP ping every 14 minutes to:
# https://your-app.onrender.com/health
```

**B. Upgrade to paid plan:**
```
Starter plan ($7/month): No sleep, always responsive
```

**C. Add loading page:**
```python
# In streamlit_dashboard.py
if st.session_state.get('first_load', True):
    with st.spinner("Waking up services... (30 seconds)"):
        time.sleep(30)
    st.session_state.first_load = False
```

---

#### Issue 6: High Risk Alerts Not Showing

**Symptoms:**
- Trigger deterioration events but only see MODERATE/HIGH, not CRITICAL

**Fix: Lower alert threshold:**

```python
# In streamlit_dashboard.py (around line 450)
# OLD:
if risk_score > 0.7:
    risk_level = "CRITICAL"

# NEW (for better demo):
if risk_score > 0.55:
    risk_level = "CRITICAL"
```

**Or trigger multiple events:**
```
1. Trigger "Hypoxia"
2. Wait 30 seconds
3. Trigger "Tachycardia"
4. Get prediction → Should show CRITICAL
```

---

## Performance Optimization

### Application Performance

**1. Model Loading:**
```python
# Load model once on startup, not per request
predictor = None

@app.on_event("startup")
async def startup_event():
    global predictor
    predictor = HealthPredictor()  # Load once

@app.post("/predict")
async def predict(data: PredictionRequest):
    # Use global predictor (already loaded)
    result = predictor.predict(data)
```

**2. Feature Engineering Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def engineer_features(vitals_hash):
    # Cache frequently-used feature combinations
    # ...
```

**3. Database Query Optimization:**
```python
# Use connection pooling
from sqlalchemy import create_engine
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20
)
```

### Infrastructure Scaling

**Horizontal Scaling (Multiple Instances):**

```yaml
# render.yaml
services:
  - type: web
    name: vitalviewai
    numInstances: 3  # Run 3 replicas
    autoscaling:
      minInstances: 2
      maxInstances: 10
      targetCPUPercent: 70
```

**Load Balancing:**
```
User Requests
      ↓
Load Balancer (Render)
      ↓
   ┌──┴──┐
   ↓     ↓
Instance 1  Instance 2  Instance 3
```

**Caching Layer:**
```python
import redis

# Redis for caching predictions
redis_client = redis.Redis(
    host='redis-server',
    port=6379,
    db=0
)

def get_prediction(patient_id, vitals):
    cache_key = f"pred:{patient_id}:{hash(vitals)}"
    
    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Compute prediction
    result = predictor.predict(vitals)
    
    # Cache for 5 minutes
    redis_client.setex(cache_key, 300, json.dumps(result))
    
    return result
```

---

## Security Considerations

### Production Security Checklist

**Application Security:**
```
✓ Use environment variables for secrets (not hardcoded)
✓ Enable HTTPS only (Render provides free SSL)
✓ Implement rate limiting (prevent DDoS)
✓ Validate all user inputs
✓ Sanitize database queries (prevent SQL injection)
✓ Use parameterized queries
✓ Implement CSRF protection
✓ Set secure HTTP headers
```

**Data Security:**
```
✓ Encrypt data at rest (AES-256)
✓ Encrypt data in transit (TLS 1.3)
✓ Hash patient IDs (SHA-256)
✓ Anonymize exported data
✓ Implement audit logging
✓ Regular security audits
```

**Implementation Example:**

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address

app = FastAPI()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")  # Max 10 requests per minute
async def predict(request: Request, data: PredictionRequest):
    # ... prediction logic

# CORS (restrict origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Don't use "*"
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Trusted hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
)

# Secure headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Render

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: |
          docker build -f Dockerfile -t vitalviewai:${{ github.sha }} .
      
      - name: Test Docker image
        run: |
          docker run -d --name test-container vitalviewai:${{ github.sha }}
          sleep 30
          docker logs test-container
          docker stop test-container

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - name: Deploy to Render
        run: |
          curl -X POST \
            -H "Authorization: Bearer ${{ secrets.RENDER_API_KEY }}" \
            https://api.render.com/v1/services/${{ secrets.RENDER_SERVICE_ID }}/deploys
```

**Setup:**
1. Go to GitHub repo → Settings → Secrets
2. Add:
   - `RENDER_API_KEY`: Get from Render → Account Settings → API Keys
   - `RENDER_SERVICE_ID`: Get from Render service URL

---

## Monitoring & Maintenance

### Health Monitoring

**Uptime Monitoring:**

Use [UptimeRobot](https://uptimerobot.com) (free):
```
1. Sign up at uptimerobot.com
2. Add monitor:
   - Type: HTTP(s)
   - URL: https://your-app.onrender.com/_stcore/health
   - Interval: 5 minutes
3. Get alerts via email when service is down
```

**Application Monitoring:**

```python
# Install Sentry (optional)
# pip install sentry-sdk

import sentry_sdk

sentry_sdk.init(
    dsn="your-sentry-dsn",
    traces_sample_rate=0.1,
    environment="production"
)
```

### Log Management

**View Logs:**
```bash
# On Render
# Go to Dashboard → Logs tab

# Download logs
# Click "Download Logs" button

# Search logs
# Use search bar: "error", "warning", "prediction"
```

**Log Rotation:**
```python
# In logging_config.py
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'logs/application.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5  # Keep 5 old files
)
```

### Backup Strategy

**Database Backups (when using PostgreSQL):**
```bash
# Automated daily backups
# In cron or Render cron jobs:
0 2 * * * pg_dump $DATABASE_URL > backup_$(date +\%Y\%m\%d).sql

# Restore:
psql $DATABASE_URL < backup_20260220.sql
```

**Model Backups:**
```python
# Version models with metadata
import datetime
import shutil

def backup_model():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy(
        'models/xgboost_model.pkl',
        f'models/backups/xgboost_model_{timestamp}.pkl'
    )
```

---

## Deployment Checklist

### Pre-Deployment

-  All tests passing locally
-  Model trained and validated
-  Environment variables configured
-  .gitignore updated (no large files)
-  Docker build successful locally
-  Documentation updated
-  Secrets stored securely (not in code)

### Deployment

-  Code pushed to GitHub
-  Render service created
-  Environment variables set in Render
-  Health check endpoint configured
-  Deployment logs monitored
-  Services started successfully
-  Model downloaded and loaded

### Post-Deployment

-  Application accessible via URL
-  Login works with demo accounts
-  Patient creation works
-  Predictions return results
-  Alerts display correctly
-  No errors in logs
-  Uptime monitoring configured
-  Documentation updated with live URL

---

## Getting Help

### Resources

**Official Documentation:**
- [Render Docs](https://render.com/docs)
- [FastAPI Docs](https://fastapi.tiangolo.com)
- [Streamlit Docs](https://docs.streamlit.io)
- [Docker Docs](https://docs.docker.com)

**Community:**
- [Render Community Forum](https://community.render.com)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/render)
- GitHub Issues: [github.com/Shabeehak/VitalViewAI/issues](https://github.com/Shabeehak/VitalViewAI/issues)

### Common Questions

**Q: Can I deploy for free?**  
A: Yes! Render free tier includes 750 hours/month. Perfect for portfolio projects.

**Q: Why does my app sleep?**  
A: Free tier apps sleep after 15 minutes of inactivity. First request wakes it up (30-60s delay).

**Q: How do I upgrade to paid?**  
A: Render Dashboard → Your Service → Settings → Plan → Select "Starter" ($7/month)

**Q: Can I use my own domain?**  
A: Yes! Add custom domain in Render settings, then update your DNS with CNAME record.

**Q: How do I rollback a deployment?**  
A: Render Dashboard → Deploys → Select previous deploy → "Redeploy"

---

## Conclusion

You've now learned how to:
- ✅ Set up local development environment
- ✅ Run services with Docker
- ✅ Deploy to production cloud (Render)
- ✅ Configure environment variables
- ✅ Troubleshoot common issues
- ✅ Monitor and maintain the application

**Your VitalViewAI system is production-ready!** 🚀

---

**Need help?** Open an issue on [GitHub](https://github.com/Shabeehak/VitalViewAI/issues)

**Last Updated:** February 2026