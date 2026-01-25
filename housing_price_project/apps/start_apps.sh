#!/bin/bash

# Startup script for Streamlit apps and Flask API

echo "Starting Ames Housing Price Prediction System..."
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Check if Flask API is already running
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null ; then
    echo "WARNING: Flask API is already running on port 5000"
else
    echo "Starting Flask API on port 5000..."
    python -m src.serving.api &
    FLASK_PID=$!
    echo "Flask API started (PID: $FLASK_PID)"
    sleep 3  # Give API time to start
fi

# Check if User App is already running
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null ; then
    echo "WARNING: User App is already running on port 8501"
else
    echo "Starting User App on port 8501..."
    streamlit run apps/app_user.py --server.port 8501 &
    USER_APP_PID=$!
    echo "User App started (PID: $USER_APP_PID)"
fi

# Check if Monitoring App is already running
if lsof -Pi :8502 -sTCP:LISTEN -t >/dev/null ; then
    echo "WARNING: Monitoring App is already running on port 8502"
else
    echo "Starting Monitoring App on port 8502..."
    streamlit run apps/app_monitoring.py --server.port 8502 &
    MONITORING_APP_PID=$!
    echo "Monitoring App started (PID: $MONITORING_APP_PID)"
fi

echo ""
echo "All services started!"
echo ""
echo "Access the apps at:"
echo "  - User App:        http://localhost:8501"
echo "  - Monitoring App:   http://localhost:8502"
echo "  - Flask API:       http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
trap "echo ''; echo 'Stopping all services...'; kill $FLASK_PID $USER_APP_PID $MONITORING_APP_PID 2>/dev/null; exit" INT
wait
