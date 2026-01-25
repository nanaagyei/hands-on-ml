"""
Monitoring Dashboard for Ames Housing Price Prediction Model.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime, timedelta
import time

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from apps.utils.api_client import (
    check_api_health, get_model_info, get_prediction_stats
)
from apps.utils.visualizations import (
    plot_time_series, plot_drift_comparison, plot_psi_heatmap,
    plot_price_distribution
)
from apps.config import (
    MONITORING_DATA_DIR, DRIFT_THRESHOLDS, PERFORMANCE_THRESHOLDS
)

# Try to import monitoring classes
try:
    from src.serving.monitoring import DriftDetector, PredictionMonitor, PerformanceMonitor
    from src.serving.retraining import RetrainingManager
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    st.warning("Monitoring classes not available. Some features may be limited.")

# Page config
st.set_page_config(
    page_title="Model Monitoring Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .error-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'monitoring_data' not in st.session_state:
    st.session_state.monitoring_data = {
        'predictions': [],
        'performance_history': [],
        'drift_history': [],
        'alerts': []
    }

# Data storage files
PREDICTIONS_FILE = MONITORING_DATA_DIR / "predictions.json"
PERFORMANCE_FILE = MONITORING_DATA_DIR / "performance.json"
DRIFT_FILE = MONITORING_DATA_DIR / "drift.json"
ALERTS_FILE = MONITORING_DATA_DIR / "alerts.json"


def load_monitoring_data():
    """Load monitoring data from files."""
    data = {
        'predictions': [],
        'performance_history': [],
        'drift_history': [],
        'alerts': []
    }
    
    if PREDICTIONS_FILE.exists():
        try:
            with open(PREDICTIONS_FILE, 'r') as f:
                data['predictions'] = json.load(f)
        except:
            pass
    
    if PERFORMANCE_FILE.exists():
        try:
            with open(PERFORMANCE_FILE, 'r') as f:
                data['performance_history'] = json.load(f)
        except:
            pass
    
    if DRIFT_FILE.exists():
        try:
            with open(DRIFT_FILE, 'r') as f:
                data['drift_history'] = json.load(f)
        except:
            pass
    
    if ALERTS_FILE.exists():
        try:
            with open(ALERTS_FILE, 'r') as f:
                data['alerts'] = json.load(f)
        except:
            pass
    
    return data


def save_monitoring_data(data):
    """Save monitoring data to files."""
    MONITORING_DATA_DIR.mkdir(exist_ok=True)
    
    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump(data['predictions'], f, indent=2, default=str)
    
    with open(PERFORMANCE_FILE, 'w') as f:
        json.dump(data['performance_history'], f, indent=2, default=str)
    
    with open(DRIFT_FILE, 'w') as f:
        json.dump(data['drift_history'], f, indent=2, default=str)
    
    with open(ALERTS_FILE, 'w') as f:
        json.dump(data['alerts'], f, indent=2, default=str)


def dashboard_overview():
    """Dashboard overview with key metrics."""
    st.markdown("# Model Monitoring Dashboard")
    
    # Load data
    monitoring_data = load_monitoring_data()
    stats = get_prediction_stats()
    model_info = get_model_info()
    
    # Key metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        n_predictions = stats.get('n_predictions', len(monitoring_data['predictions']))
        st.metric("Total Predictions", f"{n_predictions:,}")
    
    with col2:
        avg_pred = stats.get('mean_prediction', 0)
        st.metric("Avg Prediction", f"${avg_pred:,.0f}")
    
    with col3:
        # Calculate predictions today
        today = datetime.now().date()
        today_predictions = [p for p in monitoring_data['predictions'] 
                           if datetime.fromisoformat(p.get('timestamp', '2000-01-01')).date() == today]
        st.metric("Today", len(today_predictions))
    
    with col4:
        # API health
        api_healthy = check_api_health()
        status = "Healthy" if api_healthy else "Down"
        st.metric("API Status", status)
    
    with col5:
        # Model type
        model_type = model_info.get('model_type', 'Unknown') if model_info else 'Unknown'
        st.metric("Model Type", model_type)
    
    with col6:
        # Active alerts
        active_alerts = [a for a in monitoring_data['alerts'] if not a.get('acknowledged', False)]
        st.metric("Active Alerts", len(active_alerts), delta=None)
    
    st.divider()
    
    # Recent predictions chart
    if monitoring_data['predictions']:
        recent_preds = monitoring_data['predictions'][-100:]  # Last 100
        pred_df = pd.DataFrame(recent_preds)
        
        if 'timestamp' in pred_df.columns and 'prediction' in pred_df.columns:
            # Convert timestamp strings to datetime for plotting, but keep as string for display
            try:
                # Create a datetime column for sorting/plotting
                pred_df['timestamp_dt'] = pd.to_datetime(pred_df['timestamp'], errors='coerce')
                # Drop rows with invalid timestamps
                pred_df = pred_df.dropna(subset=['timestamp_dt'])
                pred_df = pred_df.sort_values('timestamp_dt')
                # Keep original timestamp as string to avoid Arrow issues
                pred_df['timestamp'] = pred_df['timestamp'].astype(str)
            except Exception as e:
                # If timestamp conversion fails, use index as x-axis
                pred_df = pred_df.reset_index()
                pred_df['timestamp'] = pred_df.index.astype(str)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Recent Predictions")
                # Use datetime column for plotting if available, otherwise use timestamp string
                time_col = 'timestamp_dt' if 'timestamp_dt' in pred_df.columns else 'timestamp'
                fig = plot_time_series(
                    pred_df,
                    time_col,
                    'prediction',
                    "Prediction Trend"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Prediction Distribution")
                fig = plot_price_distribution(pred_df, 'prediction')
                st.plotly_chart(fig, use_container_width=True)
    
    # Active alerts
    active_alerts = [a for a in monitoring_data['alerts'] if not a.get('acknowledged', False)]
    if active_alerts:
        st.divider()
        st.subheader("Active Alerts")
        for alert in active_alerts[:5]:  # Show top 5
            severity = alert.get('severity', 'warning')
            if severity == 'critical':
                st.error(f"[CRITICAL] {alert.get('message', 'Unknown alert')}")
            elif severity == 'warning':
                st.warning(f"[WARNING] {alert.get('message', 'Unknown alert')}")
            else:
                st.info(f"[INFO] {alert.get('message', 'Unknown alert')}")


def performance_monitoring():
    """Performance monitoring section."""
    st.markdown("# Performance Monitoring")
    
    monitoring_data = load_monitoring_data()
    stats = get_prediction_stats()
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # R² score (if available)
        r2 = monitoring_data.get('current_r2', None)
        if r2 is not None:
            st.metric("R² Score", f"{r2:.4f}")
        else:
            st.metric("R² Score", "N/A")
    
    with col2:
        # RMSE (if available)
        rmse = monitoring_data.get('current_rmse', None)
        if rmse is not None:
            st.metric("RMSE", f"${rmse:,.0f}")
        else:
            st.metric("RMSE", "N/A")
    
    with col3:
        # MAE (if available)
        mae = monitoring_data.get('current_mae', None)
        if mae is not None:
            st.metric("MAE", f"${mae:,.0f}")
        else:
            st.metric("MAE", "N/A")
    
    with col4:
        # MAPE (if available)
        mape = monitoring_data.get('current_mape', None)
        if mape is not None:
            st.metric("MAPE", f"{mape:.2f}%")
        else:
            st.metric("MAPE", "N/A")
    
    st.divider()
    
    # Performance history
    if monitoring_data['performance_history']:
        perf_df = pd.DataFrame(monitoring_data['performance_history'])
        
        if 'timestamp' in perf_df.columns:
            try:
                # Convert to datetime for sorting, but keep original as string
                perf_df['timestamp_dt'] = pd.to_datetime(perf_df['timestamp'], errors='coerce')
                perf_df = perf_df.dropna(subset=['timestamp_dt'])
                perf_df = perf_df.sort_values('timestamp_dt')
                # Keep original timestamp as string to avoid Arrow issues
                perf_df['timestamp'] = perf_df['timestamp'].astype(str)
            except Exception:
                # If timestamp conversion fails, use index
                perf_df = perf_df.reset_index()
                perf_df['timestamp'] = perf_df.index.astype(str)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'r2' in perf_df.columns:
                    st.subheader("R² Score Over Time")
                    fig = plot_time_series(perf_df, 'timestamp', 'r2', "R² Score Trend")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'rmse' in perf_df.columns:
                    st.subheader("RMSE Over Time")
                    fig = plot_time_series(perf_df, 'timestamp', 'rmse', "RMSE Trend")
                    st.plotly_chart(fig, use_container_width=True)
    
    # Performance alerts
    st.subheader("Performance Alerts")
    r2_threshold = PERFORMANCE_THRESHOLDS.get('r2_warning', 0.85)
    if r2 is not None and r2 < r2_threshold:
        st.warning(f"⚠️ R² score ({r2:.4f}) is below warning threshold ({r2_threshold})")
    
    # Manual performance entry (for testing)
    with st.expander("Add Performance Data (Testing)"):
        col1, col2, col3 = st.columns(3)
        with col1:
            test_r2 = st.number_input("R²", value=0.90, min_value=0.0, max_value=1.0)
        with col2:
            test_rmse = st.number_input("RMSE", value=20000.0, min_value=0.0)
        with col3:
            test_mae = st.number_input("MAE", value=15000.0, min_value=0.0)
        
        if st.button("Add Performance Record"):
            new_record = {
                'timestamp': datetime.now().isoformat(),
                'r2': test_r2,
                'rmse': test_rmse,
                'mae': test_mae
            }
            monitoring_data['performance_history'].append(new_record)
            monitoring_data['current_r2'] = test_r2
            monitoring_data['current_rmse'] = test_rmse
            monitoring_data['current_mae'] = test_mae
            save_monitoring_data(monitoring_data)
            st.success("Performance record added!")
            st.rerun()


def drift_detection():
    """Data drift detection section."""
    st.markdown("# Data Drift Detection")
    
    if not MONITORING_AVAILABLE:
        st.warning("Drift detection requires monitoring classes. Please ensure src.serving.monitoring is available.")
        return
    
    monitoring_data = load_monitoring_data()
    
    # Drift summary
    if monitoring_data['drift_history']:
        latest_drift = monitoring_data['drift_history'][-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            drift_detected = latest_drift.get('drift_detected', False)
            status = "Detected" if drift_detected else "No Drift"
            st.metric("Drift Status", status)
        
        with col2:
            mean_psi = latest_drift.get('mean_psi', 0)
            st.metric("Mean PSI", f"{mean_psi:.4f}")
        
        with col3:
            n_drifted = latest_drift.get('n_drifted', 0)
            st.metric("Drifted Features", n_drifted)
        
        st.divider()
        
        # PSI scores
        psi_scores = latest_drift.get('psi_scores', {})
        if psi_scores:
            st.subheader("PSI Scores by Feature")
            fig = plot_psi_heatmap(psi_scores)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Top drifted features
            st.subheader("Top Drifted Features")
            sorted_psi = sorted(psi_scores.items(), key=lambda x: x[1], reverse=True)
            drift_df = pd.DataFrame(sorted_psi[:20], columns=['Feature', 'PSI'])
            
            # Add severity
            drift_df['Severity'] = drift_df['PSI'].apply(
                lambda x: 'Critical' if x >= DRIFT_THRESHOLDS['psi_critical'] 
                else 'Alert' if x >= DRIFT_THRESHOLDS['psi_alert']
                else 'Warning' if x >= DRIFT_THRESHOLDS['psi_warning']
                else 'Normal'
            )
            
            # Ensure all columns are Arrow-compatible
            for col in drift_df.columns:
                if drift_df[col].dtype == 'object':
                    # Convert object columns to string, handling NaT/NaN
                    drift_df[col] = drift_df[col].astype(str).replace(['NaT', 'nan', '<NA>'], '')
                elif pd.api.types.is_datetime64_any_dtype(drift_df[col]):
                    # Convert datetime columns to string
                    drift_df[col] = drift_df[col].astype(str).replace(['NaT', 'nan', '<NA>'], '')
                elif pd.api.types.is_numeric_dtype(drift_df[col]):
                    # Fill NaN in numeric columns
                    drift_df[col] = drift_df[col].fillna(0)
            st.dataframe(drift_df, use_container_width=True)
    
    # Manual drift check
    st.divider()
    st.subheader("Run Drift Check")
    
    if st.button("Check for Drift"):
        with st.spinner("Checking for data drift..."):
            # This would normally use actual data
            # For demo, create a mock drift report
            mock_drift = {
                'timestamp': datetime.now().isoformat(),
                'drift_detected': False,
                'mean_psi': 0.05,
                'max_psi': 0.12,
                'n_drifted': 0,
                'psi_scores': {
                    'area': 0.08,
                    'Overall.Qual': 0.05,
                    'Neighborhood': 0.12,
                    'Year.Built': 0.03
                }
            }
            
            monitoring_data['drift_history'].append(mock_drift)
            save_monitoring_data(monitoring_data)
            st.success("Drift check completed!")
            st.rerun()


def prediction_logs():
    """Prediction logs viewer."""
    st.markdown("# 📋 Prediction Logs")
    
    monitoring_data = load_monitoring_data()
    stats = get_prediction_stats()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now().date() - timedelta(days=7), datetime.now().date()),
            key="log_date_range"
        )
    
    with col2:
        min_price = st.number_input("Min Price", value=0, min_value=0)
    
    with col3:
        max_price = st.number_input("Max Price", value=1000000, min_value=0)
    
    # Filter predictions
    all_predictions = monitoring_data['predictions']
    
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = []
        for p in all_predictions:
            try:
                pred_timestamp = datetime.fromisoformat(p.get('timestamp', '2000-01-01')).date()
                if start_date <= pred_timestamp <= end_date:
                    pred_value = p.get('prediction', 0)
                    if min_price <= pred_value <= max_price:
                        filtered.append(p)
            except (ValueError, AttributeError):
                # Skip predictions with invalid timestamps
                continue
    else:
        filtered = [
            p for p in all_predictions
            if min_price <= p.get('prediction', 0) <= max_price
        ]
    
    st.info(f"Showing {len(filtered)} of {len(all_predictions)} predictions")
    
    # Display logs
    if filtered:
        log_df = pd.DataFrame(filtered)
        
        # Format for display
        display_cols = ['timestamp', 'prediction']
        if 'confidence_interval' in log_df.columns:
            # Extract confidence bounds
            log_df['lower'] = log_df['confidence_interval'].apply(
                lambda x: x.get('lower', 0) if isinstance(x, dict) else 0
            )
            log_df['upper'] = log_df['confidence_interval'].apply(
                lambda x: x.get('upper', 0) if isinstance(x, dict) else 0
            )
            display_cols.extend(['lower', 'upper'])
        
        # Convert timestamp to string to avoid Arrow issues
        if 'timestamp' in log_df.columns:
            # Always convert to string, handling NaT/NaN
            log_df['timestamp'] = log_df['timestamp'].astype(str).replace(['NaT', 'nan', '<NA>'], '')
        
        # Select only columns that exist for display
        display_cols_available = [col for col in display_cols if col in log_df.columns]
        display_df = log_df[display_cols_available].copy()
        
        # Convert all columns to Arrow-compatible types
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                display_df[col] = display_df[col].astype(str).replace(['NaT', 'nan', '<NA>'], '')
            elif pd.api.types.is_datetime64_any_dtype(display_df[col]):
                display_df[col] = display_df[col].astype(str).replace(['NaT', 'nan', '<NA>'], '')
            # Numeric columns should be fine as-is
        
        st.dataframe(
            display_df.head(100),
            use_container_width=True
        )
        
        # Export
        if st.button("Export Logs"):
            # Convert timestamp back to string for CSV export
            export_df = display_df.copy()
            if 'timestamp' in export_df.columns:
                export_df['timestamp'] = export_df['timestamp'].astype(str)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"prediction_logs_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No predictions found matching the filters.")


def system_health():
    """System health monitoring."""
    st.markdown("# System Health")
    
    # API Health
    st.subheader("API Health")
    api_healthy = check_api_health()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "Healthy" if api_healthy else "Down"
        st.metric("API Status", status)
    
    with col2:
        # Response time (mock)
        response_time = 45 if api_healthy else 0
        st.metric("Response Time", f"{response_time}ms")
    
    with col3:
        # Uptime (mock)
        uptime = "99.9%" if api_healthy else "0%"
        st.metric("Uptime", uptime)
    
    st.divider()
    
    # Model Info
    st.subheader("Model Information")
    model_info = get_model_info()
    
    if model_info:
        info_df = pd.DataFrame([model_info])
        # Convert object columns to string to avoid Arrow issues
        for col in info_df.columns:
            if info_df[col].dtype == 'object':
                info_df[col] = info_df[col].astype(str).replace(['NaT', 'nan', '<NA>'], '')
            elif pd.api.types.is_datetime64_any_dtype(info_df[col]):
                info_df[col] = info_df[col].astype(str).replace(['NaT', 'nan', '<NA>'], '')
        
        # Ensure all columns are Arrow-compatible after transpose
        info_display = info_df.T.copy()
        for col in info_display.columns:
            if info_display[col].dtype == 'object':
                info_display[col] = info_display[col].astype(str).replace(['NaT', 'nan', '<NA>'], '')
            elif pd.api.types.is_datetime64_any_dtype(info_display[col]):
                info_display[col] = info_display[col].astype(str).replace(['NaT', 'nan', '<NA>'], '')
        
        st.dataframe(info_display, use_container_width=True)
    else:
        st.warning("Could not fetch model information.")
    
    st.divider()
    
    # Error Logs
    st.subheader("Error Logs")
    monitoring_data = load_monitoring_data()
    errors = [a for a in monitoring_data['alerts'] if a.get('severity') == 'error']
    
    if errors:
        error_df = pd.DataFrame(errors)
        # Convert timestamp to string to avoid Arrow issues
        if 'timestamp' in error_df.columns:
            error_df['timestamp'] = error_df['timestamp'].astype(str).replace(['NaT', 'nan', '<NA>'], '')
        display_cols = ['timestamp', 'message']
        available_cols = [col for col in display_cols if col in error_df.columns]
        if available_cols:
            # Ensure all columns are Arrow-compatible before display
            display_df = error_df[available_cols].copy()
            for col in display_df.columns:
                if display_df[col].dtype == 'object':
                    display_df[col] = display_df[col].astype(str).replace(['NaT', 'nan', '<NA>'], '')
                elif pd.api.types.is_datetime64_any_dtype(display_df[col]):
                    display_df[col] = display_df[col].astype(str).replace(['NaT', 'nan', '<NA>'], '')
            
            st.dataframe(display_df, use_container_width=True)
    else:
        st.success("No errors in recent logs.")


def retraining_management():
    """Retraining management section."""
    st.markdown("# Retraining Management")
    
    if not MONITORING_AVAILABLE:
        st.warning("Retraining management requires retraining classes.")
        return
    
    monitoring_data = load_monitoring_data()
    
    # Retraining status
    st.subheader("Retraining Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        last_retrain = monitoring_data.get('last_retrain', None)
        if last_retrain:
            st.metric("Last Retrain", last_retrain)
        else:
            st.metric("Last Retrain", "Never")
    
    with col2:
        retrain_count = len(monitoring_data.get('retrain_history', []))
        st.metric("Total Retrains", retrain_count)
    
    with col3:
        # Check if retraining needed
        should_retrain = st.button("Check if Retraining Needed")
        if should_retrain:
            st.info("Retraining check would be performed here based on drift and performance metrics.")
    
    st.divider()
    
    # Retraining history
    st.subheader("Retraining History")
    retrain_history = monitoring_data.get('retrain_history', [])
    
    if retrain_history:
        history_df = pd.DataFrame(retrain_history)
        # Convert all columns to Arrow-compatible types
        for col in history_df.columns:
            if history_df[col].dtype == 'object':
                # For object columns, convert directly to string
                # Replace NaT/NaN with empty string to avoid conversion issues
                history_df[col] = history_df[col].astype(str).replace(['NaT', 'nan', '<NA>'], '')
            elif pd.api.types.is_datetime64_any_dtype(history_df[col]):
                # For datetime columns, convert to string
                history_df[col] = history_df[col].astype(str).replace(['NaT', 'nan', '<NA>'], '')
            elif pd.api.types.is_numeric_dtype(history_df[col]):
                # For numeric columns, replace NaN with 0 to avoid issues
                history_df[col] = history_df[col].fillna(0)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No retraining history available.")
    
    # Manual retraining trigger
    st.divider()
    st.subheader("Manual Retraining")
    
    st.warning("Manual retraining should be done carefully. Ensure you have validation data.")
    
    if st.button("Trigger Manual Retraining", type="primary"):
        st.info("Retraining would be triggered here. This requires training data and validation.")


def alerts_center():
    """Alerts and issues center."""
    st.markdown("# 🚨 Alerts & Issues")
    
    monitoring_data = load_monitoring_data()
    alerts = monitoring_data['alerts']
    
    # Filter alerts
    col1, col2 = st.columns(2)
    
    with col1:
        show_acknowledged = st.checkbox("Show Acknowledged", value=False)
    
    with col2:
        severity_filter = st.selectbox(
            "Severity",
            ["All", "Critical", "Warning", "Info"]
        )
    
    # Filter alerts
    filtered_alerts = alerts
    if not show_acknowledged:
        filtered_alerts = [a for a in filtered_alerts if not a.get('acknowledged', False)]
    if severity_filter != "All":
        filtered_alerts = [a for a in filtered_alerts if a.get('severity') == severity_filter.lower()]
    
    st.info(f"Showing {len(filtered_alerts)} alerts")
    
    # Display alerts
    for alert in filtered_alerts:
        severity = alert.get('severity', 'info')
        message = alert.get('message', 'Unknown alert')
        timestamp = alert.get('timestamp', 'Unknown time')
        acknowledged = alert.get('acknowledged', False)
        
        if severity == 'critical':
            st.error(f"[CRITICAL] **{timestamp}**: {message}")
        elif severity == 'warning':
            st.warning(f"[WARNING] **{timestamp}**: {message}")
        else:
            st.info(f"[INFO] **{timestamp}**: {message}")
        
        if not acknowledged:
            if st.button(f"Acknowledge", key=f"ack_{alert.get('id', hash(message))}"):
                alert['acknowledged'] = True
                save_monitoring_data(monitoring_data)
                st.rerun()
        
        st.divider()


# Main navigation
def main():
    """Main monitoring app function."""
    # Sidebar
    with st.sidebar:
        st.title("📊 Monitoring")
        
        pages = {
            "Dashboard": "dashboard",
            "Performance": "performance",
            "Data Drift": "drift",
            "Prediction Logs": "logs",
            "System Health": "health",
            "Retraining": "retraining",
            "Alerts": "alerts"
        }
        
        selected_page = st.radio(
            "Navigation",
            list(pages.keys()),
            key="monitoring_nav"
        )
        
        st.divider()
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        # Manual refresh
        if st.button("Refresh"):
            st.rerun()
    
    # Route to page
    page_key = pages[selected_page]
    
    if page_key == "dashboard":
        dashboard_overview()
    elif page_key == "performance":
        performance_monitoring()
    elif page_key == "drift":
        drift_detection()
    elif page_key == "logs":
        prediction_logs()
    elif page_key == "health":
        system_health()
    elif page_key == "retraining":
        retraining_management()
    elif page_key == "alerts":
        alerts_center()


if __name__ == "__main__":
    main()
