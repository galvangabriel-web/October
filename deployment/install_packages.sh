#!/bin/bash
# Install missing packages on server
# Run on server via SSH

PASSWORD="1253*1253*Win1"

echo "Installing Python packages..."
echo ""

echo "$PASSWORD" | sudo -S apt-get update
echo "$PASSWORD" | sudo -S apt-get install -y python3-venv python3-pip python3-dev

echo ""
echo "Installation complete!"
python3 --version
pip3 --version
