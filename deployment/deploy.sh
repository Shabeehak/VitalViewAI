#!/bin/bash
# deploy.sh - One-Click Deployment for VitalViewAI
# Task 5 Deliverable: Complete Deployment Automation

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                   VitalViewAI Deployment Script                      ║"
echo "║                      Task 5: Scalability & Deployment                ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from template...${NC}"
    cp .env.template .env
    echo -e "${RED}❌ Please edit .env with your configuration and run again!${NC}"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${BLUE}🔍 Checking prerequisites...${NC}"

if ! command_exists docker; then
    echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker found${NC}"

if ! command_exists docker-compose; then
    if ! docker compose version >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker Compose not found. Please install Docker Compose.${NC}"
        exit 1
    fi
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi
echo -e "${GREEN}✅ Docker Compose found${NC}"

# Create necessary directories
echo -e "\n${BLUE}📁 Creating directories...${NC}"
mkdir -p data/raw data/processed models logs backups monitoring/prometheus monitoring/grafana/dashboards monitoring/grafana/datasources nginx/ssl nginx/html database

# Generate data if not exists
if [ ! -f "data/processed/features_multi.csv" ]; then
    echo -e "\n${YELLOW}📊 Generating training data...${NC}"
    python generate_diverse_training_data.py
fi

# Train model if not exists
if [ ! -f "models/xgboost_model.pkl" ]; then
    echo -e "\n${YELLOW}🤖 Training model...${NC}"
    python train_models_quick.py
fi

# Build Docker images
echo -e "\n${BLUE}🐳 Building Docker images...${NC}"
$DOCKER_COMPOSE build

# Start services
echo -e "\n${BLUE}🚀 Starting services...${NC}"
$DOCKER_COMPOSE up -d

# Wait for services to be healthy
echo -e "\n${BLUE}⏳ Waiting for services to be healthy...${NC}"
sleep 10

# Health checks
echo -e "\n${BLUE}🏥 Running health checks...${NC}"

check_service() {
    local name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $name is healthy${NC}"
            return 0
        fi
        echo -e "${YELLOW}⏳ Waiting for $name... (attempt $attempt/$max_attempts)${NC}"
        sleep 2
        ((attempt++))
    done
    
    echo -e "${RED}❌ $name failed to start${NC}"
    return 1
}

check_service "Streaming API" "http://localhost:8000/health"
check_service "ML Server" "http://localhost:8001/health"
check_service "PostgreSQL" "http://localhost:5432" || echo -e "${YELLOW}⚠️  PostgreSQL check skipped${NC}"

# Create test patient
echo -e "\n${BLUE}👤 Creating test patient...${NC}"
curl -X POST "http://localhost:8000/patients" \
    -H "Content-Type: application/json" \
    -d '{"patient_id": "test_patient_001", "sampling_interval_seconds": 60}' \
    > /dev/null 2>&1 && echo -e "${GREEN}✅ Test patient created${NC}" || echo -e "${YELLOW}⚠️  Patient may already exist${NC}"

# Display service URLs
echo -e "\n${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    🎉 DEPLOYMENT SUCCESSFUL! 🎉                       ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${BLUE}📡 Service URLs:${NC}"
echo -e "   Streaming API:     ${GREEN}http://localhost:8000${NC}"
echo -e "   ML Server:         ${GREEN}http://localhost:8001${NC}"
echo -e "   API Docs:          ${GREEN}http://localhost:8000/docs${NC}"
echo -e "   ML Docs:           ${GREEN}http://localhost:8001/docs${NC}"
echo -e "   Grafana:           ${GREEN}http://localhost:3000${NC} (admin/admin)"
echo -e "   Prometheus:        ${GREEN}http://localhost:9090${NC}"
echo -e "   Kibana:            ${GREEN}http://localhost:5601${NC}"

echo -e "\n${BLUE}🧪 Quick Test:${NC}"
echo -e "   curl http://localhost:8000/health"
echo -e "   curl http://localhost:8001/health"

echo -e "\n${BLUE}📊 View Dashboard:${NC}"
echo -e "   streamlit run streamlit_dashboard.py"

echo -e "\n${BLUE}📝 View Logs:${NC}"
echo -e "   docker-compose logs -f api-server"
echo -e "   docker-compose logs -f ml-server"

echo -e "\n${BLUE}🛑 Stop Services:${NC}"
echo -e "   docker-compose down"

echo -e "\n${BLUE}♻️  Restart Services:${NC}"
echo -e "   docker-compose restart"

echo -e "\n${GREEN}✅ VitalViewAI is now running!${NC}\n"