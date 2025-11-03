#!/bin/bash
################################################################################
# Racing Dashboard - Automated Deployment Script
# Purpose: Deploy the racing telemetry dashboard to Linux server
# Date: 2025-11-02
################################################################################

set -e  # Exit on error

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOY_USER="tactical"
APP_NAME="racing-dashboard"
APP_DIR="/home/${DEPLOY_USER}/${APP_NAME}"
VENV_DIR="${APP_DIR}/venv"
LOG_DIR="${APP_DIR}/logs"

echo -e "${BLUE}=================================="
echo "Racing Dashboard - Deployment"
echo "==================================${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Step 1: System Updates
print_status "Step 1: Updating system packages..."
sudo apt-get update -y || sudo yum update -y || print_warning "Could not update packages"
print_success "System updated"
echo ""

# Step 2: Install Required System Packages
print_status "Step 2: Installing required system packages..."

# Detect package manager
if command -v apt-get &> /dev/null; then
    PKG_MANAGER="apt-get"
    PACKAGES="python3 python3-pip python3-venv python3-dev build-essential git nginx supervisor curl"
    sudo apt-get install -y $PACKAGES
elif command -v yum &> /dev/null; then
    PKG_MANAGER="yum"
    PACKAGES="python3 python3-pip python3-devel gcc gcc-c++ make git nginx supervisor curl"
    sudo yum install -y $PACKAGES
else
    print_error "Unsupported package manager. Please install packages manually."
    exit 1
fi

print_success "System packages installed"
echo ""

# Step 3: Check Python Version
print_status "Step 3: Verifying Python version..."
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ $PYTHON_MAJOR -eq 3 ] && [ $PYTHON_MINOR -ge 10 ]; then
    print_success "Python $PYTHON_VERSION installed (meets requirement 3.10+)"
else
    print_warning "Python $PYTHON_VERSION installed (3.10+ recommended)"
fi
echo ""

# Step 4: Create Application Directory
print_status "Step 4: Creating application directory..."
mkdir -p ${APP_DIR}
mkdir -p ${LOG_DIR}
cd ${APP_DIR}
print_success "Directory created: ${APP_DIR}"
echo ""

# Step 5: Clone/Copy Project Files
print_status "Step 5: Setting up project files..."
# Note: This assumes files are uploaded separately via scp/rsync
# If using git, uncomment the following:
# git clone <repository-url> ${APP_DIR}
print_warning "Project files should be uploaded via scp/rsync"
print_warning "Run from Windows: scp -P 5197 -r C:\\project\\data_analisys_car\\* tactical@200.58.107.214:${APP_DIR}/"
echo ""

# Step 6: Create Python Virtual Environment
print_status "Step 6: Creating Python virtual environment..."
python3 -m venv ${VENV_DIR}
print_success "Virtual environment created"
echo ""

# Step 7: Upgrade pip and Install Python Dependencies
print_status "Step 7: Installing Python dependencies..."
source ${VENV_DIR}/bin/activate

# Upgrade pip
${VENV_DIR}/bin/pip install --upgrade pip

# Install requirements
if [ -f "${APP_DIR}/requirements.txt" ]; then
    ${VENV_DIR}/bin/pip install -r ${APP_DIR}/requirements.txt
    print_success "Python dependencies installed"
else
    print_error "requirements.txt not found. Please upload project files first."
    exit 1
fi
echo ""

# Step 8: Configure Environment Variables
print_status "Step 8: Setting up environment variables..."
cat > ${APP_DIR}/.env << 'EOF'
# Racing Dashboard Environment Configuration
ENVIRONMENT=production
DEBUG=False

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
MAX_FILE_SIZE=524288000

# Dashboard Configuration
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050

# CORS Configuration
ALLOWED_ORIGINS=http://200.58.107.214,http://localhost:8050

# Logging
LOG_LEVEL=INFO
LOG_DIR=/home/tactical/racing-dashboard/logs
EOF

print_success "Environment variables configured"
echo ""

# Step 9: Create Systemd Service for API
print_status "Step 9: Creating systemd service for API..."
sudo tee /etc/systemd/system/racing-api.service > /dev/null << EOF
[Unit]
Description=Racing Dashboard API Service
After=network.target

[Service]
Type=simple
User=${DEPLOY_USER}
WorkingDirectory=${APP_DIR}
Environment="PATH=${VENV_DIR}/bin"
EnvironmentFile=${APP_DIR}/.env
ExecStart=${VENV_DIR}/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=10

# Logging
StandardOutput=append:${LOG_DIR}/api-stdout.log
StandardError=append:${LOG_DIR}/api-stderr.log

[Install]
WantedBy=multi-user.target
EOF

print_success "API service created"
echo ""

# Step 10: Create Systemd Service for Dashboard
print_status "Step 10: Creating systemd service for Dashboard..."
sudo tee /etc/systemd/system/racing-dashboard.service > /dev/null << EOF
[Unit]
Description=Racing Dashboard Web Service
After=network.target racing-api.service

[Service]
Type=simple
User=${DEPLOY_USER}
WorkingDirectory=${APP_DIR}
Environment="PATH=${VENV_DIR}/bin"
EnvironmentFile=${APP_DIR}/.env
ExecStart=${VENV_DIR}/bin/python src/dashboard/app.py
Restart=always
RestartSec=10

# Logging
StandardOutput=append:${LOG_DIR}/dashboard-stdout.log
StandardError=append:${LOG_DIR}/dashboard-stderr.log

[Install]
WantedBy=multi-user.target
EOF

print_success "Dashboard service created"
echo ""

# Step 11: Configure Nginx Reverse Proxy
print_status "Step 11: Configuring Nginx reverse proxy..."
sudo tee /etc/nginx/sites-available/racing-dashboard > /dev/null << 'EOF'
upstream dashboard_backend {
    server 127.0.0.1:8050;
    keepalive 64;
}

upstream api_backend {
    server 127.0.0.1:8000;
    keepalive 64;
}

server {
    listen 80;
    server_name 200.58.107.214;

    client_max_body_size 500M;

    # API endpoints
    location /api/ {
        proxy_pass http://api_backend/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Dashboard
    location / {
        proxy_pass http://dashboard_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Static files (if needed)
    location /assets/ {
        alias /home/tactical/racing-dashboard/src/dashboard/assets/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Enable site (for Debian/Ubuntu)
if [ -d "/etc/nginx/sites-enabled" ]; then
    sudo ln -sf /etc/nginx/sites-available/racing-dashboard /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
fi

# Test nginx configuration
sudo nginx -t
print_success "Nginx configured"
echo ""

# Step 12: Configure Firewall
print_status "Step 12: Configuring firewall..."
if command -v ufw &> /dev/null; then
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    sudo ufw allow 5197/tcp  # SSH port
    print_success "UFW rules added"
elif command -v firewall-cmd &> /dev/null; then
    sudo firewall-cmd --permanent --add-service=http
    sudo firewall-cmd --permanent --add-service=https
    sudo firewall-cmd --permanent --add-port=5197/tcp
    sudo firewall-cmd --reload
    print_success "Firewalld rules added"
else
    print_warning "No firewall detected. Please configure manually."
fi
echo ""

# Step 13: Set Permissions
print_status "Step 13: Setting permissions..."
sudo chown -R ${DEPLOY_USER}:${DEPLOY_USER} ${APP_DIR}
sudo chmod -R 755 ${APP_DIR}
print_success "Permissions set"
echo ""

# Step 14: Reload Systemd and Start Services
print_status "Step 14: Starting services..."
sudo systemctl daemon-reload
sudo systemctl enable racing-api
sudo systemctl enable racing-dashboard
sudo systemctl enable nginx
sudo systemctl start racing-api
sudo systemctl start racing-dashboard
sudo systemctl restart nginx

# Wait for services to start
sleep 5

# Check service status
API_STATUS=$(sudo systemctl is-active racing-api)
DASHBOARD_STATUS=$(sudo systemctl is-active racing-dashboard)
NGINX_STATUS=$(sudo systemctl is-active nginx)

if [ "$API_STATUS" = "active" ]; then
    print_success "Racing API service is running"
else
    print_error "Racing API service failed to start"
    sudo journalctl -u racing-api -n 50 --no-pager
fi

if [ "$DASHBOARD_STATUS" = "active" ]; then
    print_success "Racing Dashboard service is running"
else
    print_error "Racing Dashboard service failed to start"
    sudo journalctl -u racing-dashboard -n 50 --no-pager
fi

if [ "$NGINX_STATUS" = "active" ]; then
    print_success "Nginx service is running"
else
    print_error "Nginx service failed to start"
fi
echo ""

# Step 15: Deployment Summary
echo -e "${GREEN}=================================="
echo "Deployment Complete!"
echo "==================================${NC}"
echo ""
echo "Dashboard URL: http://200.58.107.214"
echo "API URL: http://200.58.107.214/api"
echo ""
echo "Services:"
echo "  - API: ${API_STATUS}"
echo "  - Dashboard: ${DASHBOARD_STATUS}"
echo "  - Nginx: ${NGINX_STATUS}"
echo ""
echo "Useful Commands:"
echo "  - Check API logs: sudo journalctl -u racing-api -f"
echo "  - Check Dashboard logs: sudo journalctl -u racing-dashboard -f"
echo "  - Restart API: sudo systemctl restart racing-api"
echo "  - Restart Dashboard: sudo systemctl restart racing-dashboard"
echo "  - Check service status: sudo systemctl status racing-api racing-dashboard"
echo ""
echo "Log files:"
echo "  - API: ${LOG_DIR}/api-stdout.log"
echo "  - Dashboard: ${LOG_DIR}/dashboard-stdout.log"
echo ""
