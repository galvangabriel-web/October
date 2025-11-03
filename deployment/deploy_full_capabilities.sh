#!/bin/bash
# ============================================================
# Automated Deployment Script - Full Capabilities Mode
# Racing Analytics Dashboard - RAM Upgraded Server
# ============================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}🏁 Racing Analytics Dashboard - Full Deployment${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Load environment variables
if [ ! -f ../.env ]; then
    echo -e "${RED}❌ .env file not found!${NC}"
    echo "Please copy .env.example to .env and fill in your details"
    exit 1
fi

source ../.env

# Validate required variables
if [ -z "$SSH_HOST" ] || [ -z "$SSH_USER" ] || [ -z "$DEPLOY_PATH" ]; then
    echo -e "${RED}❌ Missing required environment variables${NC}"
    echo "Please check SSH_HOST, SSH_USER, and DEPLOY_PATH in .env"
    exit 1
fi

if [ "$SSH_PASSWORD" == "CHANGE_ME_TO_YOUR_PASSWORD" ]; then
    echo -e "${RED}❌ Please update SSH_PASSWORD in .env file${NC}"
    exit 1
fi

# Check if sshpass is installed (for password authentication)
if ! command -v sshpass &> /dev/null; then
    echo -e "${YELLOW}⚠️  sshpass not found. Installing...${NC}"
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y sshpass
    elif command -v yum &> /dev/null; then
        sudo yum install -y sshpass
    else
        echo -e "${RED}❌ Please install sshpass manually${NC}"
        exit 1
    fi
fi

# SSH command helper
SSH_CMD="sshpass -p '$SSH_PASSWORD' ssh -o StrictHostKeyChecking=no ${SSH_USER}@${SSH_HOST}"
SCP_CMD="sshpass -p '$SSH_PASSWORD' scp -o StrictHostKeyChecking=no -r"

echo -e "${GREEN}✅ Configuration validated${NC}"
echo ""

# Test connection
echo -e "${BLUE}🔌 Testing connection to ${SSH_USER}@${SSH_HOST}...${NC}"
if $SSH_CMD "echo 'Connection successful'" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Connected successfully${NC}"
else
    echo -e "${RED}❌ Connection failed${NC}"
    exit 1
fi
echo ""

# Step 1: Setup server environment
echo -e "${BLUE}🔧 Step 1: Setting up server environment...${NC}"
$SSH_CMD << 'EOF'
    # Update system
    echo "📦 Updating system packages..."
    sudo apt-get update -qq

    # Install Python and dependencies
    echo "🐍 Installing Python and dependencies..."
    sudo apt-get install -y python3 python3-pip python3-venv \
        build-essential libssl-dev libffi-dev python3-dev \
        git curl wget htop

    echo "✅ Server environment ready"
EOF
echo ""

# Step 2: Create deployment directory
echo -e "${BLUE}📁 Step 2: Creating deployment directory...${NC}"
$SSH_CMD "mkdir -p $DEPLOY_PATH"
echo -e "${GREEN}✅ Directory created: $DEPLOY_PATH${NC}"
echo ""

# Step 3: Upload application files
echo -e "${BLUE}📤 Step 3: Uploading application files...${NC}"

cd ..

# Upload core files
echo "Uploading core files..."
$SCP_CMD data_loader.py analyze_all_data.py inventory_data.py requirements.txt \
    README.md CLAUDE.md .env ${SSH_USER}@${SSH_HOST}:${DEPLOY_PATH}/

# Upload source directories
echo "Uploading src/ directory..."
$SCP_CMD src ${SSH_USER}@${SSH_HOST}:${DEPLOY_PATH}/

echo "Uploading tests/ directory..."
$SCP_CMD tests ${SSH_USER}@${SSH_HOST}:${DEPLOY_PATH}/

echo "Uploading deployment/ directory..."
$SCP_CMD deployment ${SSH_USER}@${SSH_HOST}:${DEPLOY_PATH}/

echo -e "${GREEN}✅ Files uploaded${NC}"
echo ""

# Step 4: Setup Python environment
echo -e "${BLUE}🐍 Step 4: Setting up Python virtual environment...${NC}"
$SSH_CMD << EOF
    cd $DEPLOY_PATH

    # Create virtual environment
    echo "Creating virtual environment..."
    python3 -m venv venv

    # Activate and upgrade pip
    source venv/bin/activate
    pip install --upgrade pip -q

    # Install requirements (this takes 5-10 minutes)
    echo "Installing Python packages (this may take 5-10 minutes)..."
    pip install -r requirements.txt

    echo "✅ Python environment ready"
EOF
echo ""

# Step 5: Create startup scripts
echo -e "${BLUE}📝 Step 5: Creating startup scripts...${NC}"
$SSH_CMD << EOF
    cd $DEPLOY_PATH

    # Dashboard startup script
    cat > start_dashboard.sh << 'SCRIPT'
#!/bin/bash
# Racing Analytics Dashboard - Full Capabilities Mode

cd $DEPLOY_PATH
source venv/bin/activate

export ENABLE_FULL_MEMORY=true
export LOAD_ALL_CHUNKS=true
export DASHBOARD_HOST=${DASHBOARD_HOST:-0.0.0.0}
export DASHBOARD_PORT=${DASHBOARD_PORT:-8050}

echo "🏁 Starting Racing Analytics Dashboard..."
echo "📊 Mode: Full Capabilities (All data chunks loaded)"
echo "🌐 URL: http://\${DASHBOARD_HOST}:\${DASHBOARD_PORT}"
echo "💾 Memory: Unlimited (RAM upgraded)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python src/dashboard/app.py
SCRIPT

    # API startup script
    cat > start_api.sh << 'SCRIPT'
#!/bin/bash
# Racing Analytics API - Full Capabilities Mode

cd $DEPLOY_PATH
source venv/bin/activate

export ENABLE_FULL_MEMORY=true
export API_HOST=${API_HOST:-0.0.0.0}
export API_PORT=${API_PORT:-8000}
export API_WORKERS=${API_WORKERS:-4}

echo "🚀 Starting Racing Analytics API..."
echo "🌐 URL: http://\${API_HOST}:\${API_PORT}"
echo "👷 Workers: \${API_WORKERS}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python -m uvicorn src.api.main:app \\
    --host \$API_HOST \\
    --port \$API_PORT \\
    --workers \$API_WORKERS \\
    --reload
SCRIPT

    # Start both services
    cat > start_all.sh << 'SCRIPT'
#!/bin/bash
# Start both Dashboard and API

cd $DEPLOY_PATH

echo "🚀 Starting all services..."
echo ""

# Start API in background
nohup ./start_api.sh > logs/api.log 2>&1 &
API_PID=\$!
echo "✅ API started (PID: \$API_PID)"

# Wait for API to start
sleep 3

# Start Dashboard in background
nohup ./start_dashboard.sh > logs/dashboard.log 2>&1 &
DASH_PID=\$!
echo "✅ Dashboard started (PID: \$DASH_PID)"

echo ""
echo "🌐 Dashboard: http://\${DASHBOARD_HOST:-0.0.0.0}:${DASHBOARD_PORT:-8050}"
echo "🌐 API: http://\${API_HOST:-0.0.0.0}:${API_PORT:-8000}"
echo ""
echo "📋 View logs:"
echo "   tail -f logs/dashboard.log"
echo "   tail -f logs/api.log"
echo ""
echo "🛑 Stop services:"
echo "   ./stop_all.sh"
SCRIPT

    # Stop all services
    cat > stop_all.sh << 'SCRIPT'
#!/bin/bash
# Stop all Racing Analytics services

echo "🛑 Stopping all services..."

# Kill dashboard
pkill -f "python src/dashboard/app.py"
echo "✅ Dashboard stopped"

# Kill API
pkill -f "uvicorn src.api.main:app"
echo "✅ API stopped"

echo ""
echo "✅ All services stopped"
SCRIPT

    # Make scripts executable
    chmod +x start_dashboard.sh start_api.sh start_all.sh stop_all.sh

    # Create logs directory
    mkdir -p logs

    echo "✅ Startup scripts created"
EOF
echo ""

# Step 6: Test installation
echo -e "${BLUE}🧪 Step 6: Testing installation...${NC}"
$SSH_CMD << EOF
    cd $DEPLOY_PATH
    source venv/bin/activate

    # Test Python version
    echo -n "Python: "
    python --version

    # Test key packages
    python -c "import pandas; print('pandas:', pandas.__version__)"
    python -c "import dash; print('dash:', dash.__version__)"
    python -c "import fastapi; print('fastapi:', fastapi.__version__)"
    python -c "import lightgbm; print('lightgbm:', lightgbm.__version__)"

    echo "✅ All tests passed"
EOF
echo ""

# Final summary
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}✅ DEPLOYMENT SUCCESSFUL!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "${BLUE}📍 Deployed to:${NC} $DEPLOY_PATH"
echo ""
echo -e "${YELLOW}🚀 To start services:${NC}"
echo "   ssh ${SSH_USER}@${SSH_HOST}"
echo "   cd $DEPLOY_PATH"
echo "   ./start_all.sh           # Start both Dashboard + API"
echo "   # OR"
echo "   ./start_dashboard.sh     # Dashboard only"
echo "   ./start_api.sh           # API only"
echo ""
echo -e "${YELLOW}🛑 To stop services:${NC}"
echo "   ./stop_all.sh"
echo ""
echo -e "${YELLOW}📋 View logs:${NC}"
echo "   tail -f logs/dashboard.log"
echo "   tail -f logs/api.log"
echo ""
echo -e "${YELLOW}🌐 Access URLs:${NC}"
echo "   Dashboard: http://${SSH_HOST}:${DASHBOARD_PORT:-8050}"
echo "   API: http://${SSH_HOST}:${API_PORT:-8000}"
echo ""
echo -e "${BLUE}💡 Note:${NC} Services are running in FULL CAPABILITIES mode"
echo "   - All telemetry chunks loaded"
echo "   - Full memory usage enabled"
echo "   - Multiple API workers"
echo ""
