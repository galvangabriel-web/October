#!/bin/bash
################################################################################
# Racing Dashboard - LIGHTWEIGHT Deployment Script (3GB RAM)
# WARNING: This is for TESTING ONLY with small sample files
# Purpose: Deploy on low-memory servers with severe limitations
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
PASSWORD="1253*1253*Win1"

echo -e "${BLUE}=================================="
echo "Racing Dashboard - LIGHTWEIGHT Deployment"
echo "FOR TESTING ONLY - 3GB RAM"
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

# Display WARNING
echo -e "${RED}================================================"
echo "WARNING: LOW MEMORY DEPLOYMENT"
echo "================================================${NC}"
echo ""
echo "This server has only 3GB RAM (8GB minimum recommended)"
echo ""
echo "Limitations in this deployment:"
echo "  • Only 1 API worker (reduced performance)"
echo "  • Max upload size: 50MB (vs 500MB normal)"
echo "  • Memory limit per service: 1GB"
echo "  • No aggressive caching"
echo "  • Single user recommended"
echo "  • Expect crashes with large files"
echo ""
echo -e "${YELLOW}Use only for TESTING with small sample files!${NC}"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Step 1: Create Application Directory
print_status "Step 1: Creating application directory..."
mkdir -p ${APP_DIR}
mkdir -p ${LOG_DIR}
cd ${APP_DIR}
print_success "Directory created: ${APP_DIR}"
echo ""

# Step 2: Create Python Virtual Environment
print_status "Step 2: Creating Python virtual environment..."
python3 -m venv ${VENV_DIR}
print_success "Virtual environment created"
echo ""

# Step 3: Upgrade pip and Install Python Dependencies
print_status "Step 3: Installing Python dependencies (this may take 5-10 minutes)..."
source ${VENV_DIR}/bin/activate

# Upgrade pip
${VENV_DIR}/bin/pip install --upgrade pip

# Install requirements
if [ -f "${APP_DIR}/requirements.txt" ]; then
    ${VENV_DIR}/bin/pip install -r ${APP_DIR}/requirements.txt
    print_success "Python dependencies installed"
else
    print_error "requirements.txt not found. Please upload project files first."
    echo ""
    echo "Upload files using:"
    echo "  scp -P 5197 -r src requirements.txt data_loader.py tactical@200.58.107.214:${APP_DIR}/"
    exit 1
fi
echo ""

# Step 4: Configure Environment Variables
print_status "Step 4: Setting up environment variables..."
cat > ${APP_DIR}/.env << 'EOF'
# Racing Dashboard Environment Configuration (LIGHTWEIGHT MODE)
ENVIRONMENT=production
DEBUG=False

# API Configuration (LIGHTWEIGHT - 1 worker only)
API_HOST=0.0.0.0
API_PORT=8000
MAX_FILE_SIZE=52428800  # 50MB limit (vs 500MB normal)

# Dashboard Configuration
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050

# CORS Configuration
ALLOWED_ORIGINS=http://200.58.107.214,http://localhost:8050

# Logging
LOG_LEVEL=WARNING  # Reduced logging to save memory
LOG_DIR=/home/tactical/racing-dashboard/logs

# Memory optimization
PYTHONUNBUFFERED=1
MALLOC_TRIM_THRESHOLD_=100000
EOF

print_success "Environment variables configured (lightweight mode)"
echo ""

# Step 5: Create Systemd Service for API (LIGHTWEIGHT)
print_status "Step 5: Creating systemd service for API (1 worker only)..."
echo "$PASSWORD" | sudo -S tee /etc/systemd/system/racing-api.service > /dev/null << EOF
[Unit]
Description=Racing Dashboard API Service (LIGHTWEIGHT MODE)
After=network.target

[Service]
Type=simple
User=${DEPLOY_USER}
WorkingDirectory=${APP_DIR}
Environment="PATH=${VENV_DIR}/bin"
EnvironmentFile=${APP_DIR}/.env
ExecStart=${VENV_DIR}/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 30
Restart=always
RestartSec=10

# Memory limits for low-RAM server
MemoryMax=1G
MemoryHigh=900M

# Logging
StandardOutput=append:${LOG_DIR}/api-stdout.log
StandardError=append:${LOG_DIR}/api-stderr.log

[Install]
WantedBy=multi-user.target
EOF

print_success "API service created (lightweight mode)"
echo ""

# Step 6: Create Systemd Service for Dashboard (LIGHTWEIGHT)
print_status "Step 6: Creating systemd service for Dashboard..."
echo "$PASSWORD" | sudo -S tee /etc/systemd/system/racing-dashboard.service > /dev/null << EOF
[Unit]
Description=Racing Dashboard Web Service (LIGHTWEIGHT MODE)
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

# Memory limits for low-RAM server
MemoryMax=1G
MemoryHigh=900M

# Logging
StandardOutput=append:${LOG_DIR}/dashboard-stdout.log
StandardError=append:${LOG_DIR}/dashboard-stderr.log

[Install]
WantedBy=multi-user.target
EOF

print_success "Dashboard service created (lightweight mode)"
echo ""

# Step 7: Configure Nginx Reverse Proxy
print_status "Step 7: Configuring Nginx reverse proxy..."
echo "$PASSWORD" | sudo -S tee /etc/nginx/sites-available/racing-dashboard > /dev/null << 'EOF'
upstream dashboard_backend {
    server 127.0.0.1:8050;
    keepalive 8;  # Reduced for low memory
}

upstream api_backend {
    server 127.0.0.1:8000;
    keepalive 8;  # Reduced for low memory
}

server {
    listen 80;
    server_name 200.58.107.214;

    client_max_body_size 50M;  # Reduced from 500M

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
        proxy_read_timeout 120s;  # Reduced timeout
        proxy_connect_timeout 30s;

        # Buffer settings for low memory
        proxy_buffering off;
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
        proxy_read_timeout 120s;  # Reduced timeout
        proxy_connect_timeout 30s;

        # Buffer settings for low memory
        proxy_buffering off;
    }

    # Static files (if needed)
    location /assets/ {
        alias /home/tactical/racing-dashboard/src/dashboard/assets/;
        expires 7d;  # Reduced caching
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Enable site (for Debian/Ubuntu)
if [ -d "/etc/nginx/sites-enabled" ]; then
    echo "$PASSWORD" | sudo -S ln -sf /etc/nginx/sites-available/racing-dashboard /etc/nginx/sites-enabled/
    echo "$PASSWORD" | sudo -S rm -f /etc/nginx/sites-enabled/default
fi

# Test nginx configuration
echo "$PASSWORD" | sudo -S nginx -t
print_success "Nginx configured (lightweight mode)"
echo ""

# Step 8: Configure Firewall (if UFW exists)
print_status "Step 8: Configuring firewall..."
if command -v ufw &> /dev/null; then
    echo "$PASSWORD" | sudo -S ufw allow 80/tcp
    echo "$PASSWORD" | sudo -S ufw allow 443/tcp
    echo "$PASSWORD" | sudo -S ufw allow 5197/tcp  # SSH port
    print_success "UFW rules added"
else
    print_warning "UFW not installed, skipping firewall configuration"
fi
echo ""

# Step 9: Set Permissions
print_status "Step 9: Setting permissions..."
echo "$PASSWORD" | sudo -S chown -R ${DEPLOY_USER}:${DEPLOY_USER} ${APP_DIR}
echo "$PASSWORD" | sudo -S chmod -R 755 ${APP_DIR}
print_success "Permissions set"
echo ""

# Step 10: Reload Systemd and Start Services
print_status "Step 10: Starting services..."
echo "$PASSWORD" | sudo -S systemctl daemon-reload
echo "$PASSWORD" | sudo -S systemctl enable racing-api
echo "$PASSWORD" | sudo -S systemctl enable racing-dashboard
echo "$PASSWORD" | sudo -S systemctl enable nginx
echo "$PASSWORD" | sudo -S systemctl start racing-api
echo "$PASSWORD" | sudo -S systemctl start racing-dashboard
echo "$PASSWORD" | sudo -S systemctl restart nginx

# Wait for services to start
sleep 5

# Check service status
API_STATUS=$(echo "$PASSWORD" | sudo -S systemctl is-active racing-api)
DASHBOARD_STATUS=$(echo "$PASSWORD" | sudo -S systemctl is-active racing-dashboard)
NGINX_STATUS=$(echo "$PASSWORD" | sudo -S systemctl is-active nginx)

if [ "$API_STATUS" = "active" ]; then
    print_success "Racing API service is running"
else
    print_error "Racing API service failed to start"
    echo "$PASSWORD" | sudo -S journalctl -u racing-api -n 50 --no-pager
fi

if [ "$DASHBOARD_STATUS" = "active" ]; then
    print_success "Racing Dashboard service is running"
else
    print_error "Racing Dashboard service failed to start"
    echo "$PASSWORD" | sudo -S journalctl -u racing-dashboard -n 50 --no-pager
fi

if [ "$NGINX_STATUS" = "active" ]; then
    print_success "Nginx service is running"
else
    print_error "Nginx service failed to start"
fi
echo ""

# Step 11: Deployment Summary
echo -e "${GREEN}=================================="
echo "LIGHTWEIGHT Deployment Complete!"
echo "==================================${NC}"
echo ""
echo -e "${YELLOW}⚠ IMPORTANT LIMITATIONS:${NC}"
echo "  • Max file upload: 50MB"
echo "  • Only 1 API worker (slower)"
echo "  • Memory limit: 1GB per service"
echo "  • Single user recommended"
echo "  • Test with SMALL files only"
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
echo "  - Check logs: sudo journalctl -u racing-dashboard -f"
echo "  - Check memory: free -h"
echo "  - Restart if crash: sudo systemctl restart racing-api racing-dashboard"
echo ""
echo -e "${YELLOW}If services crash or become unresponsive:${NC}"
echo "  1. Check memory: free -h"
echo "  2. Restart services: sudo systemctl restart racing-api racing-dashboard"
echo "  3. Use smaller test files"
echo "  4. Consider upgrading to 8GB+ RAM for production use"
echo ""
