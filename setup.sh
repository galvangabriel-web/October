#!/bin/bash
# Racing Data Analysis - Linux/Mac Setup Script
# This script creates a virtual environment and installs all dependencies

echo "========================================"
echo "Racing Data Analysis Setup"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo ""
    echo "Please install Python 3.10+ from:"
    echo "  - Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "  - macOS: brew install python3"
    echo "  - Or: https://www.python.org/downloads/"
    exit 1
fi

echo "[1/4] Python found:"
python3 --version
echo ""

echo "[2/4] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
    echo "Virtual environment created successfully"
fi
echo ""

echo "[3/4] Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi
echo ""

echo "[4/5] Creating data directories..."
mkdir -p data/processed data/models
echo "Data directories created successfully"
echo ""

echo "[5/5] Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi
echo ""

echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo ""
echo "To activate the virtual environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To run data analysis:"
echo "  python deep_data_analysis.py"
echo ""
