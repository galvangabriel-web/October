#!/bin/bash
################################################################################
# Racing Dashboard - Finish Deployment (Manual Sudo)
# Run this script ON THE SERVER after Python dependencies are installed
# Date: 2025-11-02
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Config
DEPLOY_USER="tactical"
APP_DIR="/home/${DEPLOY_USER}/racing-dashboard"
VENV_DIR="${APP_DIR}/venv"
LOG_DIR="${APP_DIR}/logs"

echo -e "${BLUE}=================================="
echo "Finishing Deployment"
echo "==================================${NC}"
echo ""

# Create logs directory
mkdir -p ${LOG_DIR}

# Step 1: Create .env file
echo -e "${BLUE}Step 1: Creating environment configuration...${NC}"
cat > ${APP_DIR}/.env << 'EOF'
ENVIRONMENT=production
DEBUG=False
API_HOST=0.0.0.0
API_PORT=8000
MAX_FILE_SIZE=52428800
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050
ALLOWED_ORIGINS=http://200.58.107.214,http://localhost:8050
LOG_LEVEL=WARNING
LOG_DIR=/home/tactical/racing-dashboard/logs
PYTHONUNBUFFERED=1
EOF

echo -e "${GREEN}✓${NC} Environment configured"
echo ""

# Step 2: Create systemd services (requires sudo)
echo -e "${BLUE}Step 2: Creating systemd services...${NC}"
echo "Creating racing-api.service..."

sudo tee /etc/systemd/system/racing-api.service > /dev/null << EOF
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
MemoryMax=1G
MemoryHigh=900M
StandardOutput=append:${LOG_DIR}/api-stdout.log
StandardError=append:${LOG_DIR}/api-stderr.log

[Install]
WantedBy=multi-user.target
EOF

echo "Creating racing-dashboard.service..."

sudo tee /etc/systemd/system/racing-dashboard.service > /dev/null << EOF
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
MemoryMax=1G
MemoryHigh=900M
StandardOutput=append:${LOG_DIR}/dashboard-stdout.log
StandardError=append:${LOG_DIR}/dashboard-stderr.log

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}✓${NC} Services created"
echo ""

# Step 3: Configure Nginx
echo -e "${BLUE}Step 3: Configuring Nginx...${NC}"

sudo tee /etc/nginx/sites-available/racing-dashboard > /dev/null << 'EOF'
upstream dashboard_backend {
    server 127.0.0.1:8050;
    keepalive 8;
}

upstream api_backend {
    server 127.0.0.1:8000;
    keepalive 8;
}

server {
    listen 80;
    server_name 200.58.107.214;
    client_max_body_size 50M;

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
        proxy_read_timeout 120s;
        proxy_connect_timeout 30s;
        proxy_buffering off;
    }

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
        proxy_read_timeout 120s;
        proxy_connect_timeout 30s;
        proxy_buffering off;
    }

    location /assets/ {
        alias /home/tactical/racing-dashboard/src/dashboard/assets/;
        expires 7d;
        add_header Cache-Control "public, immutable";
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/racing-dashboard /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

sudo nginx -t

echo -e "${GREEN}✓${NC} Nginx configured"
echo ""

# Step 4: Set permissions
echo -e "${BLUE}Step 4: Setting permissions...${NC}"
sudo chown -R ${DEPLOY_USER}:${DEPLOY_USER} ${APP_DIR}
sudo chmod -R 755 ${APP_DIR}
echo -e "${GREEN}✓${NC} Permissions set"
echo ""

# Step 5: Start services
echo -e "${BLUE}Step 5: Starting services...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable racing-api
sudo systemctl enable racing-dashboard
sudo systemctl enable nginx
sudo systemctl start racing-api
sudo systemctl start racing-dashboard
sudo systemctl restart nginx

sleep 3

# Check status
API_STATUS=$(sudo systemctl is-active racing-api)
DASHBOARD_STATUS=$(sudo systemctl is-active racing-dashboard)
NGINX_STATUS=$(sudo systemctl is-active nginx)

echo ""
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

if [ "$API_STATUS" = "active" ] && [ "$DASHBOARD_STATUS" = "active" ]; then
    echo -e "${GREEN}✓ All services running!${NC}"
    echo ""
    echo "⚠️  REMEMBER: This is a TESTING deployment (3GB RAM)"
    echo "    - Max upload: 50MB"
    echo "    - Use small sample files only"
    echo "    - May crash with large files"
    echo ""
    echo "Test with: http://200.58.107.214"
else
    echo -e "${RED}✗ Some services failed to start${NC}"
    echo ""
    echo "Check logs:"
    echo "  sudo journalctl -u racing-api -n 50"
    echo "  sudo journalctl -u racing-dashboard -n 50"
fi

echo ""
echo "Useful commands:"
echo "  - Check status: sudo systemctl status racing-api racing-dashboard"
echo "  - View logs: sudo journalctl -u racing-dashboard -f"
echo "  - Restart: sudo systemctl restart racing-api racing-dashboard"
echo "  - Check memory: free -h"
echo ""
