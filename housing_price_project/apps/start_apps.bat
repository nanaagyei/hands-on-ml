@echo off
REM Startup script for Windows

echo Starting Ames Housing Price Prediction System...
echo.

cd /d "%~dp0\.."

REM Start Flask API
echo Starting Flask API on port 5000...
start "Flask API" cmd /k "python -m src.serving.api"
timeout /t 3 /nobreak >nul

REM Start User App
echo Starting User App on port 8501...
start "User App" cmd /k "streamlit run apps/app_user.py --server.port 8501"

REM Start Monitoring App
echo Starting Monitoring App on port 8502...
start "Monitoring App" cmd /k "streamlit run apps/app_monitoring.py --server.port 8502"

echo.
echo All services started!
echo.
echo Access the apps at:
echo   - User App:        http://localhost:8501
echo   - Monitoring App:   http://localhost:8502
echo   - Flask API:       http://localhost:5000
echo.
pause
