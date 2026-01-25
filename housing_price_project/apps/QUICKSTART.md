# Quick Start Guide

## Prerequisites

1. Python 3.8+ installed
2. All dependencies installed: `pip install -r requirements.txt`
3. Model file exists: `models/final_model.pkl`
4. Data files exist: `data/processed/ames_cleaned.csv`

## Quick Start (3 Steps)

### Step 1: Start Flask API
```bash
cd housing_price_project
python -m src.serving.api
```
Keep this terminal open. The API will run on http://localhost:5000

### Step 2: Start User App (New Terminal)
```bash
cd housing_price_project
streamlit run apps/app_user.py --server.port 8501
```
Access at: http://localhost:8501

### Step 3: Start Monitoring App (New Terminal)
```bash
cd housing_price_project
streamlit run apps/app_monitoring.py --server.port 8502
```
Access at: http://localhost:8502

## Or Use the Startup Script

**Linux/Mac:**
```bash
cd housing_price_project
./apps/start_apps.sh
```

**Windows:**
```bash
cd housing_price_project
apps\start_apps.bat
```

## Verify Everything Works

1. **Check API:** Visit http://localhost:5000/health - should return `{"status": "healthy"}`
2. **Check User App:** Visit http://localhost:8501 - should show home page
3. **Check Monitoring:** Visit http://localhost:8502 - should show dashboard

## Troubleshooting

- **Port already in use:** Stop the process using that port or change the port in the command
- **Import errors:** Make sure you're in the project root directory
- **Data not loading:** Check that CSV files exist in `data/processed/`
- **API connection failed:** Make sure Flask API is running first

## Next Steps

- Explore the house browser
- Try making a price prediction
- Check the monitoring dashboard
- Review the full README_APPS.md for detailed documentation
