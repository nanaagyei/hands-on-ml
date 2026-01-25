"""
Flask API client utilities.
"""
import requests
import streamlit as st
from typing import Dict, Any, Optional
from apps.config import API_ENDPOINTS


def check_api_health():
    """Check if API is healthy."""
    try:
        response = requests.get(API_ENDPOINTS["health"], timeout=5)
        return response.status_code == 200
    except:
        return False


def get_model_info():
    """Get model information from API."""
    try:
        response = requests.get(API_ENDPOINTS["model_info"], timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching model info: {e}")
        return None


def predict_price(features: Dict[str, Any]) -> Optional[Dict]:
    """
    Predict house price from features.
    
    Parameters
    ----------
    features : dict
        Dictionary of feature names to values
        
    Returns
    -------
    dict or None
        Prediction result with 'prediction' and 'confidence_interval'
    """
    try:
        response = requests.post(
            API_ENDPOINTS["predict"],
            json={"features": features},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = response.json().get('error', 'Unknown error')
            st.error(f"Prediction error: {error_msg}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to API. Make sure the Flask API is running on port 5000.")
        return None
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None


def predict_batch(data: list) -> Optional[Dict]:
    """Predict prices for multiple houses."""
    try:
        response = requests.post(
            API_ENDPOINTS["predict_batch"],
            json={"data": data},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = response.json().get('error', 'Unknown error')
            st.error(f"Batch prediction error: {error_msg}")
            return None
    except Exception as e:
        st.error(f"Error making batch prediction: {e}")
        return None


def get_prediction_stats():
    """Get prediction statistics from API."""
    try:
        response = requests.get(API_ENDPOINTS["model_stats"], timeout=10)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        st.warning(f"Could not fetch prediction stats: {e}")
        return {}
