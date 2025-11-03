#!/bin/bash
################################################################################
# Server Audit - Automated Upload and Execution
# Run from Git Bash on Windows
################################################################################

SERVER_IP="200.58.107.214"
SERVER_PORT="5197"
USERNAME="tactical"
PASSWORD="1253*1253*Win1"

echo "========================================"
echo "Racing Dashboard - Server Audit"
echo "========================================"
echo ""
echo "Server: ${USERNAME}@${SERVER_IP}:${SERVER_PORT}"
echo ""

# Check if we're in the right directory
if [ ! -f "deployment/01_audit_server.sh" ]; then
    echo "ERROR: Must run from project root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "[1/3] Testing SSH connection..."
echo ""

# Method 1: Try using sshpass if available
if command -v sshpass &> /dev/null; then
    echo "Using sshpass for automation..."

    # Upload script
    echo "[2/3] Uploading audit script..."
    sshpass -p "$PASSWORD" scp -P $SERVER_PORT -o StrictHostKeyChecking=no \
        deployment/01_audit_server.sh ${USERNAME}@${SERVER_IP}:/tmp/audit_server.sh

    if [ $? -eq 0 ]; then
        echo "✓ Upload successful"
    else
        echo "✗ Upload failed"
        exit 1
    fi

    # Run audit
    echo ""
    echo "[3/3] Running audit..."
    echo "========================================"
    echo ""

    sshpass -p "$PASSWORD" ssh -p $SERVER_PORT -o StrictHostKeyChecking=no \
        ${USERNAME}@${SERVER_IP} "chmod +x /tmp/audit_server.sh && /tmp/audit_server.sh"

    AUDIT_EXIT=$?

    echo ""
    echo "========================================"
    if [ $AUDIT_EXIT -eq 0 ]; then
        echo "✓ Audit completed successfully!"
        echo ""
        echo "Server is READY for deployment"
    else
        echo "⚠ Audit exit code: $AUDIT_EXIT"
    fi

else
    # Method 2: Manual entry required
    echo "sshpass not available - will prompt for password interactively"
    echo "Password: 1253*1253*Win1"
    echo ""

    echo "[2/3] Uploading audit script..."
    echo "Enter password when prompted: 1253*1253*Win1"
    echo ""

    scp -P $SERVER_PORT -o StrictHostKeyChecking=no \
        deployment/01_audit_server.sh ${USERNAME}@${SERVER_IP}:/tmp/audit_server.sh

    if [ $? -ne 0 ]; then
        echo "✗ Upload failed"
        exit 1
    fi

    echo ""
    echo "[3/3] Running audit..."
    echo "Enter password again when prompted: 1253*1253*Win1"
    echo "========================================"
    echo ""

    ssh -p $SERVER_PORT -o StrictHostKeyChecking=no \
        ${USERNAME}@${SERVER_IP} "chmod +x /tmp/audit_server.sh && /tmp/audit_server.sh"

    echo ""
    echo "========================================"
    echo "Audit complete - review output above"
fi

echo "========================================"
