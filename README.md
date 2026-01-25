# Hands-On Machine Learning Tutorials

A comprehensive collection of machine learning projects and implementations, focusing on building production-ready ML systems from scratch.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)

## 📚 Overview

This repository contains hands-on machine learning projects designed to teach fundamental ML concepts through practical implementation. The main focus is on:

- **Building ML models from scratch** - Understanding the mathematics and implementation details
- **Production-ready applications** - Creating deployable ML systems with monitoring and APIs
- **End-to-end ML pipelines** - From data cleaning to model deployment

## 🏗️ Project Structure

```
hands-on-ml/
├── linear_regression/          # Linear regression implementation from scratch
│   ├── core/                   # Core regression algorithms
│   ├── preprocessing/          # Data preprocessing utilities
│   ├── model_selection/       # Cross-validation and model selection
│   └── tests/                 # Unit tests
│
├── housing_price_project/      # Production-ready housing price prediction system
│   ├── notebooks/             # Jupyter notebooks for EDA, cleaning, feature engineering
│   ├── src/                   # Source code
│   │   ├── data/              # Data loading and cleaning
│   │   ├── features/          # Feature engineering and encoding
│   │   ├── models/            # Model implementations
│   │   └── serving/           # API and serving infrastructure
│   ├── apps/                  # Streamlit applications
│   │   ├── app_user.py        # User-facing prediction app
│   │   └── app_monitoring.py # Model monitoring dashboard
│   ├── models/                 # Trained models
│   └── data/                  # Datasets
│
└── notebooks/                  # Additional tutorial notebooks
    ├── Chapter2_homl.ipynb
    ├── Chapter3_homl.ipynb
    └── Chapter4_homl.ipynb
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/hands-on-ml.git
   cd hands-on-ml
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   # For the housing price project
   cd housing_price_project
   pip install -r apps/requirements.txt
   ```

## 📦 Main Projects

### 1. Linear Regression from Scratch

A complete implementation of linear regression algorithms without using scikit-learn, including:

- **Core Algorithms:**
  - Ordinary Least Squares (OLS)
  - Ridge Regression
  - Lasso Regression
  - Elastic Net

- **Features:**
  - Gradient descent optimizers
  - Cross-validation
  - Feature scaling and preprocessing
  - Comprehensive test suite

**Location:** `linear_regression/`

**Usage:**
```python
from linear_regression.core.linear_regression import LinearRegression
from linear_regression.preprocessing.scalers import StandardScaler

# Create and train model
model = LinearRegression()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model.fit(X_scaled, y)

# Make predictions
predictions = model.predict(X_test_scaled)
```

### 2. Ames Housing Price Prediction System

A production-ready machine learning system for predicting house prices using the Ames Housing Dataset. This project demonstrates:

- **Complete ML Pipeline:**
  - Exploratory Data Analysis (EDA)
  - Data cleaning and preprocessing
  - Feature engineering (target encoding, one-hot encoding)
  - Model training and evaluation
  - Model deployment

- **Production Features:**
  - RESTful API (Flask)
  - Interactive web applications (Streamlit)
  - Model monitoring dashboard
  - Data drift detection
  - Model retraining pipeline

**Location:** `housing_price_project/`

**Key Components:**

1. **Data Processing:**
   - `notebooks/01_eda.ipynb` - Exploratory data analysis
   - `notebooks/02_data_cleaning.ipynb` - Data cleaning pipeline
   - `notebooks/03_feature_engineering.ipynb` - Feature engineering
   - `notebooks/04_modeling.ipynb` - Model training and evaluation

2. **Source Code:**
   - `src/data/` - Data loading and cleaning utilities
   - `src/features/` - Feature engineering and encoding
   - `src/models/` - Model implementations
   - `src/serving/` - API, monitoring, and serving infrastructure

3. **Applications:**
   - `apps/app_user.py` - User-facing prediction interface
   - `apps/app_monitoring.py` - Model monitoring dashboard

## 🎯 Getting Started with Housing Price Project

### Step 1: Start the Flask API

The API serves the trained model and handles predictions:

```bash
cd housing_price_project
python -m src.serving.api
```

The API will be available at `http://localhost:5000`

**Endpoints:**
- `GET /health` - Health check
- `GET /model/info` - Model information
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions

### Step 2: Start the Streamlit Applications

**Option A: Use the startup script (recommended)**

```bash
cd housing_price_project/apps
bash start_apps.sh  # On Windows: start_apps.bat
```

**Option B: Start manually**

Terminal 1 - User App:
```bash
streamlit run apps/app_user.py --server.port 8501
```

Terminal 2 - Monitoring Dashboard:
```bash
streamlit run apps/app_monitoring.py --server.port 8502
```

### Step 3: Access the Applications

- **User App:** http://localhost:8501
  - Browse houses in the dataset
  - Predict house prices with custom features
  - Explore neighborhood insights

- **Monitoring Dashboard:** http://localhost:8502
  - Monitor model performance
  - Track data drift
  - View prediction logs
  - Manage alerts and retraining

## 📊 Features

### Housing Price Prediction System

#### User-Facing Application
- **House Browser:** Browse and filter houses from the dataset
- **Price Predictor:** Interactive form to predict house prices
- **Neighborhood Insights:** Visualizations and statistics by neighborhood
- **House Details:** Detailed information view for each house

#### Monitoring Dashboard
- **Performance Monitoring:** Track R², RMSE, and other metrics over time
- **Data Drift Detection:** Monitor feature distributions using PSI (Population Stability Index)
- **Prediction Logs:** View and export prediction history
- **System Health:** Monitor API status and model information
- **Alerts & Issues:** Track and manage model alerts
- **Retraining Management:** Trigger and monitor model retraining

#### API Features
- **Intelligent Feature Filling:** Uses similar houses from training data to fill missing features
- **Robust Predictions:** Handles extreme values and missing features gracefully
- **Confidence Intervals:** Provides prediction uncertainty estimates
- **Batch Processing:** Supports batch predictions for multiple houses

## 🛠️ Technology Stack

- **Machine Learning:**
  - NumPy, Pandas - Data manipulation
  - Scikit-learn - ML algorithms and utilities
  - Custom implementations - Linear regression from scratch

- **Web Framework:**
  - Flask - RESTful API
  - Streamlit - Interactive web applications

- **Visualization:**
  - Plotly - Interactive charts
  - Matplotlib - Static plots

- **Data Processing:**
  - Pandas - Data cleaning and transformation
  - Custom encoders - Target encoding, one-hot encoding

## 📖 Documentation

- **Housing Price Project:** See `housing_price_project/apps/README_APPS.md` for detailed app documentation
- **Quick Start Guide:** See `housing_price_project/apps/QUICKSTART.md` for setup instructions
- **Feature Engineering:** See `housing_price_project/apps/FEATURE_DEFAULTS_EXPLANATION.md` for feature handling details

## 🧪 Testing

Run tests for the linear regression implementation:

```bash
cd linear_regression
python -m pytest tests/
```

## 📝 Notebooks

The project includes comprehensive Jupyter notebooks:

1. **01_eda.ipynb** - Exploratory data analysis
2. **02_data_cleaning.ipynb** - Data cleaning pipeline
3. **03_feature_engineering.ipynb** - Feature engineering techniques
4. **04_modeling.ipynb** - Model training, evaluation, and selection

## 🎓 Learning Objectives

This repository is designed to teach:

1. **ML Fundamentals:**
   - Understanding linear regression from first principles
   - Feature engineering techniques
   - Model evaluation and selection

2. **Production ML:**
   - Building RESTful APIs for ML models
   - Creating monitoring dashboards
   - Handling data drift and model degradation
   - Implementing model retraining pipelines

3. **Best Practices:**
   - Code organization and modularity
   - Error handling and validation
   - Documentation and testing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Ames Housing Dataset - Used for the housing price prediction project
- "Hands-On Machine Learning" by Aurélien Géron - Inspiration for tutorial structure

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note:** This is an educational project focused on learning machine learning concepts through hands-on implementation. The code is designed to be clear and educational rather than optimized for production performance.
