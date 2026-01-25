# Streamlit Apps for Ames Housing Price Prediction

This directory contains two production-ready Streamlit applications for the Ames Housing Price Prediction system.

## Applications

### 1. User-Facing App (`app_user.py`)
A comprehensive application for end users to:
- Browse house listings with filtering and search
- View detailed house information
- Predict house prices using the trained model
- Compare houses side-by-side
- Explore neighborhood insights

**Port:** 8501

### 2. Monitoring Dashboard (`app_monitoring.py`)
An internal monitoring dashboard for:
- Real-time model performance tracking
- Data drift detection
- Prediction logs and analytics
- System health monitoring
- Retraining management
- Alert management

**Port:** 8502

## Prerequisites

1. **Python 3.8+**
2. **Flask API running** on port 5000
3. **Required packages** (install with `pip install -r requirements.txt`)

## Installation

1. Install dependencies:
```bash
cd housing_price_project/apps
pip install -r requirements.txt
```

2. Ensure the Flask API is running:
```bash
cd housing_price_project
python -m src.serving.api
```

## Running the Apps

### Option 1: Manual (Separate Terminals)

**Terminal 1 - Flask API:**
```bash
cd housing_price_project
python -m src.serving.api
```

**Terminal 2 - User App:**
```bash
cd housing_price_project
streamlit run apps/app_user.py --server.port 8501
```

**Terminal 3 - Monitoring App:**
```bash
cd housing_price_project
streamlit run apps/app_monitoring.py --server.port 8502
```

### Option 2: Using the Startup Script

```bash
cd housing_price_project
chmod +x apps/start_apps.sh
./apps/start_apps.sh
```

## Accessing the Apps

- **User App:** http://localhost:8501
- **Monitoring Dashboard:** http://localhost:8502
- **Flask API:** http://localhost:5000

## Features

### User App Features

1. **Home Page**
   - Quick statistics overview
   - Search functionality
   - Market overview charts
   - Navigation to all sections

2. **House Browser**
   - Table view with sorting and filtering
   - Card/grid view with house images
   - Advanced filters (neighborhood, price, year, style)
   - Export filtered results

3. **House Details**
   - Comprehensive house information
   - Quality ratings
   - Features and amenities
   - Similar houses recommendations
   - Add to comparison

4. **Price Predictor**
   - Interactive form with grouped features
   - Real-time validation
   - Prediction with confidence intervals
   - Feature contribution visualization
   - Prediction history

5. **Neighborhood Insights**
   - Statistics per neighborhood
   - Price distribution charts
   - Market trends

6. **House Comparison**
   - Compare up to 3 houses
   - Side-by-side feature comparison
   - Visual comparison charts

### Monitoring Dashboard Features

1. **Dashboard Overview**
   - Key metrics at a glance
   - Recent prediction trends
   - Active alerts
   - System status

2. **Performance Monitoring**
   - R², RMSE, MAE, MAPE metrics
   - Performance trends over time
   - Performance alerts
   - Historical performance data

3. **Data Drift Detection**
   - PSI scores per feature
   - Distribution comparisons
   - Drift alerts
   - Top drifted features

4. **Prediction Logs**
   - Searchable/filterable logs
   - Date range filtering
   - Price range filtering
   - Export capabilities

5. **System Health**
   - API health status
   - Response time monitoring
   - Model information
   - Error logs

6. **Retraining Management**
   - Retraining status
   - Retraining history
   - Manual retraining trigger
   - Retraining configuration

7. **Alerts & Issues**
   - Alert center
   - Severity levels
   - Alert acknowledgment
   - Issue tracking

## Configuration

Edit `config.py` to customize:
- API endpoints
- File paths
- Feature groups
- Monitoring thresholds
- Image sources

## Data Storage

Monitoring data is stored in JSON files in the `monitoring_data/` directory:
- `predictions.json` - Prediction logs
- `performance.json` - Performance history
- `drift.json` - Drift detection history
- `alerts.json` - Alert history

## Troubleshooting

### API Connection Issues
- Ensure Flask API is running on port 5000
- Check API health in monitoring dashboard
- Verify API endpoints in `config.py`

### Data Loading Issues
- Verify data files exist in `data/processed/`
- Check file paths in `config.py`
- Ensure proper permissions on data files

### Import Errors
- Install all requirements: `pip install -r requirements.txt`
- Verify Python path includes project root
- Check that all utility modules are in `apps/utils/`

## Development

### Adding New Features

1. **User App:**
   - Add new pages in `app_user.py`
   - Update navigation in `main()` function
   - Add utility functions in `apps/utils/`

2. **Monitoring App:**
   - Add new sections in `app_monitoring.py`
   - Integrate with monitoring classes in `src/serving/monitoring.py`
   - Update data storage as needed

### Customization

- **Styling:** Edit CSS in the `st.markdown()` sections
- **Charts:** Modify visualization functions in `apps/utils/visualizations.py`
- **Features:** Update feature groups in `config.py`

## Notes

- The apps use Streamlit's caching for performance
- Images are loaded from Unsplash (fallback URLs provided)
- Monitoring data persists between sessions
- Both apps can run concurrently

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments
3. Check Flask API logs
4. Review monitoring dashboard for system status
