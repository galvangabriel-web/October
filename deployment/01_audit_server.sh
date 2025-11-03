#!/bin/bash
################################################################################
# Racing Dashboard Server Audit Script
# Purpose: Check if Linux server meets requirements for dashboard deployment
# Date: 2025-11-02
################################################################################

echo "=================================="
echo "Racing Dashboard - Server Audit"
echo "=================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check requirement
check_requirement() {
    local name=$1
    local result=$2
    local requirement=$3

    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $name: PASS ($requirement)"
    elif [ "$result" = "WARN" ]; then
        echo -e "${YELLOW}⚠${NC} $name: WARNING ($requirement)"
    else
        echo -e "${RED}✗${NC} $name: FAIL ($requirement)"
    fi
}

echo "=== SYSTEM INFORMATION ==="
echo "Hostname: $(hostname)"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"
echo ""

echo "=== CPU INFORMATION ==="
CPU_CORES=$(nproc)
CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)
echo "CPU Model: $CPU_MODEL"
echo "CPU Cores: $CPU_CORES"
if [ $CPU_CORES -ge 2 ]; then
    check_requirement "CPU Cores" "PASS" "Minimum 2 cores (Found: $CPU_CORES)"
else
    check_requirement "CPU Cores" "FAIL" "Minimum 2 cores (Found: $CPU_CORES)"
fi
echo ""

echo "=== MEMORY INFORMATION ==="
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_RAM_GB=$((TOTAL_RAM_KB / 1024 / 1024))
AVAILABLE_RAM_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
AVAILABLE_RAM_GB=$((AVAILABLE_RAM_KB / 1024 / 1024))
echo "Total RAM: ${TOTAL_RAM_GB} GB"
echo "Available RAM: ${AVAILABLE_RAM_GB} GB"

if [ $TOTAL_RAM_GB -ge 16 ]; then
    check_requirement "RAM (Optimal)" "PASS" "16GB+ recommended (Found: ${TOTAL_RAM_GB}GB)"
elif [ $TOTAL_RAM_GB -ge 8 ]; then
    check_requirement "RAM (Minimum)" "WARN" "8GB minimum (Found: ${TOTAL_RAM_GB}GB, 16GB recommended)"
else
    check_requirement "RAM" "FAIL" "Minimum 8GB (Found: ${TOTAL_RAM_GB}GB)"
fi
echo ""

echo "=== DISK SPACE ==="
DISK_INFO=$(df -h / | tail -1)
DISK_TOTAL=$(echo $DISK_INFO | awk '{print $2}')
DISK_USED=$(echo $DISK_INFO | awk '{print $3}')
DISK_AVAILABLE=$(echo $DISK_INFO | awk '{print $4}')
DISK_PERCENT=$(echo $DISK_INFO | awk '{print $5}')
echo "Root partition: $DISK_TOTAL total, $DISK_AVAILABLE available ($DISK_PERCENT used)"

DISK_AVAILABLE_GB=$(df -BG / | tail -1 | awk '{print $4}' | sed 's/G//')
if [ $DISK_AVAILABLE_GB -ge 20 ]; then
    check_requirement "Disk Space" "PASS" "20GB+ required (Available: ${DISK_AVAILABLE_GB}GB)"
else
    check_requirement "Disk Space" "FAIL" "20GB required (Available: ${DISK_AVAILABLE_GB}GB)"
fi
echo ""

echo "=== PYTHON INFORMATION ==="
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    echo "Python3 Version: $PYTHON_VERSION"

    if [ $PYTHON_MAJOR -eq 3 ] && [ $PYTHON_MINOR -ge 10 ]; then
        check_requirement "Python Version" "PASS" "Python 3.10+ required (Found: $PYTHON_VERSION)"
    elif [ $PYTHON_MAJOR -eq 3 ] && [ $PYTHON_MINOR -ge 8 ]; then
        check_requirement "Python Version" "WARN" "Python 3.10+ recommended (Found: $PYTHON_VERSION)"
    else
        check_requirement "Python Version" "FAIL" "Python 3.10+ required (Found: $PYTHON_VERSION)"
    fi
else
    echo "Python3: NOT INSTALLED"
    check_requirement "Python Version" "FAIL" "Python 3.10+ required (Not found)"
fi

# Check for python3-venv
if dpkg -l | grep -q python3-venv 2>/dev/null || rpm -qa | grep -q python3-venv 2>/dev/null; then
    check_requirement "Python venv" "PASS" "python3-venv installed"
else
    check_requirement "Python venv" "FAIL" "python3-venv not installed"
fi

# Check for pip
if command -v pip3 &> /dev/null; then
    PIP_VERSION=$(pip3 --version | awk '{print $2}')
    echo "Pip3 Version: $PIP_VERSION"
    check_requirement "Pip3" "PASS" "pip3 installed"
else
    check_requirement "Pip3" "FAIL" "pip3 not installed"
fi
echo ""

echo "=== NETWORK INFORMATION ==="
# Check if ports 8050 and 8000 are available
if command -v netstat &> /dev/null; then
    PORT_8050=$(netstat -tuln | grep ":8050 " | wc -l)
    PORT_8000=$(netstat -tuln | grep ":8000 " | wc -l)
    PORT_80=$(netstat -tuln | grep ":80 " | wc -l)
    PORT_443=$(netstat -tuln | grep ":443 " | wc -l)
elif command -v ss &> /dev/null; then
    PORT_8050=$(ss -tuln | grep ":8050 " | wc -l)
    PORT_8000=$(ss -tuln | grep ":8000 " | wc -l)
    PORT_80=$(ss -tuln | grep ":80 " | wc -l)
    PORT_443=$(ss -tuln | grep ":443 " | wc -l)
else
    PORT_8050=0
    PORT_8000=0
    PORT_80=0
    PORT_443=0
fi

if [ $PORT_8050 -eq 0 ]; then
    check_requirement "Port 8050" "PASS" "Available for Dashboard"
else
    check_requirement "Port 8050" "FAIL" "Port already in use"
fi

if [ $PORT_8000 -eq 0 ]; then
    check_requirement "Port 8000" "PASS" "Available for API"
else
    check_requirement "Port 8000" "FAIL" "Port already in use"
fi

if [ $PORT_80 -eq 0 ]; then
    check_requirement "Port 80 (HTTP)" "PASS" "Available for Nginx"
else
    check_requirement "Port 80 (HTTP)" "WARN" "Port already in use (may be nginx/apache)"
fi

if [ $PORT_443 -eq 0 ]; then
    check_requirement "Port 443 (HTTPS)" "PASS" "Available for Nginx"
else
    check_requirement "Port 443 (HTTPS)" "WARN" "Port already in use (may be nginx/apache)"
fi
echo ""

echo "=== INSTALLED PACKAGES ==="
# Check for common packages
PACKAGES=("git" "nginx" "build-essential" "gcc")
for pkg in "${PACKAGES[@]}"; do
    if command -v $pkg &> /dev/null 2>&1 || dpkg -l | grep -q "^ii  $pkg " 2>/dev/null || rpm -qa | grep -q "^$pkg" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $pkg: Installed"
    else
        echo -e "${RED}✗${NC} $pkg: Not installed (will be installed during deployment)"
    fi
done
echo ""

echo "=== SYSTEM LOAD ==="
LOAD=$(uptime | awk -F'load average:' '{print $2}' | xargs)
echo "Load Average: $LOAD"
echo "Uptime: $(uptime -p)"
echo ""

echo "=== FIREWALL STATUS ==="
if command -v ufw &> /dev/null; then
    echo "UFW Status:"
    sudo ufw status 2>/dev/null || echo "No sudo access to check UFW"
elif command -v firewall-cmd &> /dev/null; then
    echo "Firewalld Status:"
    sudo firewall-cmd --state 2>/dev/null || echo "No sudo access to check firewalld"
else
    echo "No firewall detected (ufw/firewalld)"
fi
echo ""

echo "=== DEPLOYMENT READINESS SUMMARY ==="
echo ""
echo "Server Details:"
echo "  - OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "  - CPU: $CPU_CORES cores"
echo "  - RAM: ${TOTAL_RAM_GB}GB total, ${AVAILABLE_RAM_GB}GB available"
echo "  - Disk: ${DISK_AVAILABLE_GB}GB available"
if command -v python3 &> /dev/null; then
    echo "  - Python: $PYTHON_VERSION"
else
    echo "  - Python: NOT INSTALLED"
fi
echo ""

# Overall assessment
CRITICAL_ISSUES=0

if [ $CPU_CORES -lt 2 ]; then ((CRITICAL_ISSUES++)); fi
if [ $TOTAL_RAM_GB -lt 8 ]; then ((CRITICAL_ISSUES++)); fi
if [ $DISK_AVAILABLE_GB -lt 20 ]; then ((CRITICAL_ISSUES++)); fi
if ! command -v python3 &> /dev/null; then ((CRITICAL_ISSUES++)); fi

if [ $CRITICAL_ISSUES -eq 0 ]; then
    echo -e "${GREEN}✓ SERVER IS READY FOR DEPLOYMENT${NC}"
    echo ""
    echo "You can proceed with the deployment using:"
    echo "  ./02_deploy_dashboard.sh"
    exit 0
else
    echo -e "${RED}✗ SERVER HAS CRITICAL ISSUES${NC}"
    echo ""
    echo "Please resolve the following before deployment:"
    if [ $CPU_CORES -lt 2 ]; then echo "  - Upgrade CPU to at least 2 cores"; fi
    if [ $TOTAL_RAM_GB -lt 8 ]; then echo "  - Upgrade RAM to at least 8GB (16GB recommended)"; fi
    if [ $DISK_AVAILABLE_GB -lt 20 ]; then echo "  - Free up disk space (need 20GB available)"; fi
    if ! command -v python3 &> /dev/null; then echo "  - Install Python 3.10+"; fi
    exit 1
fi
