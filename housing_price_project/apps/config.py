"""
Configuration file for Streamlit apps.
"""
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA = DATA_DIR / "processed" / "ames_cleaned.csv"
FEATURED_DATA = DATA_DIR / "features" / "ames_featured.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.pkl"

# API configuration
API_BASE_URL = "http://localhost:5000"
API_ENDPOINTS = {
    "health": f"{API_BASE_URL}/health",
    "predict": f"{API_BASE_URL}/predict",
    "predict_batch": f"{API_BASE_URL}/predict/batch",
    "model_info": f"{API_BASE_URL}/model/info",
    "model_stats": f"{API_BASE_URL}/model/stats",
}

# Monitoring data storage
MONITORING_DATA_DIR = PROJECT_ROOT / "monitoring_data"
MONITORING_DATA_DIR.mkdir(exist_ok=True)

# Image configuration
UNSPLASH_ACCESS_KEY = None  # Set if using Unsplash API
UNSPLASH_BASE_URL = "https://api.unsplash.com"
IMAGE_CACHE_DIR = PROJECT_ROOT / "image_cache"
IMAGE_CACHE_DIR.mkdir(exist_ok=True)

# Default house images (fallback if Unsplash not available)
DEFAULT_HOUSE_IMAGES = {
    "1Story": "https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=800",
    "2Story": "https://images.unsplash.com/photo-1568605117033-3c1820d4d5a3?w=800",
    "1.5Unf": "https://images.unsplash.com/photo-1564013799919-ab600027ffc6?w=800",
    "SFoyer": "https://images.unsplash.com/photo-1568605117033-3c1820d4d5a3?w=800",
    "SLvl": "https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=800",
    "default": "https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=800",
}

# Feature groups for predictor form
FEATURE_GROUPS = {
    "Basic Info": [
        "area", "Lot.Area", "Year.Built", "Year.Remod.Add"
    ],
    "Quality": [
        "Overall.Qual", "Overall.Cond", "Exter.Qual", "Exter.Cond",
        "Kitchen.Qual", "Bsmt.Qual", "Bsmt.Cond", "Heating.QC", "Garage.Qual"
    ],
    "Location": [
        "Neighborhood", "MS.Zoning", "Condition.1", "Condition.2"
    ],
    "Features": [
        "Bedroom.AbvGr", "Full.Bath", "Half.Bath", "Kitchen.AbvGr",
        "TotRms.AbvGrd", "Garage.Cars", "Garage.Area", "Fireplaces"
    ],
    "Amenities": [
        "Pool.Area", "Pool.QC", "Fence", "Misc.Feature", "Misc.Val"
    ],
    "Exterior": [
        "Roof.Style", "Roof.Matl", "Exterior.1st", "Exterior.2nd",
        "Foundation", "Mas.Vnr.Type", "Mas.Vnr.Area"
    ],
    "Basement": [
        "Total.Bsmt.SF", "BsmtFin.SF.1", "BsmtFin.SF.2", "Bsmt.Unf.SF",
        "Bsmt.Full.Bath", "Bsmt.Half.Bath", "Bsmt.Exposure", "BsmtFin.Type.1"
    ],
    "Garage": [
        "Garage.Type", "Garage.Yr.Blt", "Garage.Finish", "Garage.Cars",
        "Garage.Area", "Garage.Qual", "Garage.Cond"
    ],
    "Other": [
        "MS.SubClass", "Lot.Frontage", "Lot.Shape", "Land.Contour",
        "Utilities", "Lot.Config", "Land.Slope", "Street", "Alley",
        "Central.Air", "Electrical", "Paved.Drive", "Wood.Deck.SF",
        "Open.Porch.SF", "Enclosed.Porch", "X3Ssn.Porch", "Screen.Porch"
    ]
}

# Monitoring thresholds
DRIFT_THRESHOLDS = {
    "psi_warning": 0.1,
    "psi_alert": 0.2,
    "psi_critical": 0.25,
}

PERFORMANCE_THRESHOLDS = {
    "r2_warning": 0.85,
    "r2_alert": 0.80,
    "rmse_warning_multiplier": 1.2,  # 20% increase from baseline
}

# App settings
STREAMLIT_CONFIG = {
    "page_title": "Ames Housing Price Predictor",
    "page_icon": ":house:",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Cache settings
CACHE_TTL = 3600  # 1 hour
