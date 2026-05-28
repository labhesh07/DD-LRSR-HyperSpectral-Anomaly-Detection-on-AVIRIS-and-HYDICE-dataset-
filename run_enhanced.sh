#!/bin/bash
# Simple shell script to run the enhanced demo on Linux/Mac

echo "========================================"
echo "Hyperspectral Anomaly Detection - Enhanced"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3 and try again"
    exit 1
fi

echo "Python found. Starting enhanced demo..."
echo ""

# Run the enhanced demo
python3 demo_enhanced.py

echo ""
echo "========================================"
echo "Execution completed!"
echo "Check the 'results' folder for outputs"
echo "========================================"

