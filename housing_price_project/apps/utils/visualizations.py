"""
Visualization utilities for Streamlit apps.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def plot_price_distribution(df, price_col='price'):
    """Plot price distribution histogram."""
    fig = px.histogram(
        df, 
        x=price_col,
        nbins=50,
        title="Price Distribution",
        labels={price_col: "Price ($)", "count": "Number of Houses"}
    )
    fig.update_layout(
        template="plotly_white",
        height=400
    )
    return fig


def plot_price_vs_feature(df, feature_col, price_col='price'):
    """Plot price vs a feature."""
    # Remove trendline to avoid statsmodels dependency
    # Users can add it manually if statsmodels is installed
    # Note: trendline="ols" parameter removed to avoid statsmodels dependency
    fig = px.scatter(
        df,
        x=feature_col,
        y=price_col,
        title=f"Price vs {feature_col}",
        labels={feature_col: feature_col, price_col: "Price ($)"}
    )
    fig.update_layout(
        template="plotly_white",
        height=400
    )
    return fig


def plot_neighborhood_prices(df, top_n=15):
    """Plot average prices by neighborhood."""
    if 'Neighborhood' not in df.columns or 'price' not in df.columns:
        return None
    
    neighborhood_stats = df.groupby('Neighborhood')['price'].mean().sort_values(ascending=False).head(top_n)
    
    fig = px.bar(
        x=neighborhood_stats.values,
        y=neighborhood_stats.index,
        orientation='h',
        title=f"Average Price by Neighborhood (Top {top_n})",
        labels={"x": "Average Price ($)", "y": "Neighborhood"}
    )
    fig.update_layout(
        template="plotly_white",
        height=max(400, top_n * 30),
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig


def plot_confidence_interval(prediction, lower, upper):
    """Plot prediction with confidence interval."""
    fig = go.Figure()
    
    # Prediction point
    fig.add_trace(go.Scatter(
        x=[1],
        y=[prediction],
        mode='markers',
        marker=dict(size=15, color='blue'),
        name='Prediction'
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=[1, 1],
        y=[lower, upper],
        mode='lines',
        line=dict(color='blue', width=3),
        name='Confidence Interval'
    ))
    
    fig.add_trace(go.Scatter(
        x=[1, 1],
        y=[lower, upper],
        mode='markers',
        marker=dict(size=10, color='blue', symbol='line-ns-open'),
        showlegend=False
    ))
    
    fig.update_layout(
        title="Price Prediction with Confidence Interval",
        xaxis=dict(showticklabels=False, range=[0.5, 1.5]),
        yaxis_title="Price ($)",
        template="plotly_white",
        height=400,
        showlegend=True
    )
    
    return fig


def plot_feature_importance(feature_names, importances, top_n=20):
    """Plot feature importance."""
    # Sort by absolute importance
    sorted_data = sorted(zip(feature_names, importances), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_data[:top_n]
    
    names = [f[0] for f in top_features]
    values = [f[1] for f in top_features]
    colors = ['green' if v > 0 else 'red' for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=names,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.4f}" for v in values],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(400, top_n * 30),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def plot_comparison(houses_data):
    """Plot comparison of multiple houses."""
    if not houses_data:
        return None
    
    df = pd.DataFrame(houses_data)
    
    # Select numeric columns for comparison
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'price' in numeric_cols:
        numeric_cols.remove('price')
    
    # Limit to top features
    numeric_cols = numeric_cols[:10]
    
    if not numeric_cols:
        return None
    
    fig = go.Figure()
    
    for idx, row in df.iterrows():
        fig.add_trace(go.Bar(
            name=f"House {idx}",
            x=numeric_cols,
            y=[row[col] if pd.notna(row[col]) else 0 for col in numeric_cols]
        ))
    
    fig.update_layout(
        title="House Comparison",
        xaxis_title="Features",
        yaxis_title="Value",
        template="plotly_white",
        height=500,
        barmode='group'
    )
    
    return fig


def plot_time_series(data, x_col, y_col, title="Time Series"):
    """Plot time series data."""
    fig = px.line(
        data,
        x=x_col,
        y=y_col,
        title=title,
        markers=True
    )
    fig.update_layout(
        template="plotly_white",
        height=400
    )
    return fig


def plot_drift_comparison(reference_data, current_data, feature_name):
    """Plot distribution comparison for drift detection."""
    fig = go.Figure()
    
    # Reference distribution
    fig.add_trace(go.Histogram(
        x=reference_data,
        name='Reference',
        opacity=0.7,
        nbinsx=30
    ))
    
    # Current distribution
    fig.add_trace(go.Histogram(
        x=current_data,
        name='Current',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.update_layout(
        title=f"Distribution Comparison: {feature_name}",
        xaxis_title=feature_name,
        yaxis_title="Frequency",
        template="plotly_white",
        height=400,
        barmode='overlay'
    )
    
    return fig


def plot_psi_heatmap(psi_scores):
    """Plot PSI scores as heatmap."""
    if not psi_scores:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(list(psi_scores.items()), columns=['Feature', 'PSI'])
    df = df.sort_values('PSI', ascending=False)
    
    # Color scale based on thresholds
    colors = []
    for psi in df['PSI']:
        if psi >= 0.25:
            colors.append('red')
        elif psi >= 0.2:
            colors.append('orange')
        elif psi >= 0.1:
            colors.append('yellow')
        else:
            colors.append('green')
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['Feature'],
            y=df['PSI'],
            marker_color=colors,
            text=[f"{v:.3f}" for v in df['PSI']],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="PSI Scores by Feature",
        xaxis_title="Feature",
        yaxis_title="PSI Score",
        template="plotly_white",
        height=600,
        xaxis_tickangle=-45
    )
    
    return fig
