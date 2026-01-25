"""
User-facing Streamlit app for Ames Housing Price Prediction.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from apps.utils.data_loader import (
    load_house_data, get_neighborhoods, get_house_types, get_zoning_types,
    filter_houses, get_similar_houses, get_neighborhood_stats,
    encode_categorical_feature, get_categorical_encodings
)
from apps.utils.api_client import predict_price, check_api_health, get_model_info
from apps.utils.visualizations import (
    plot_price_distribution, plot_price_vs_feature,
    plot_neighborhood_prices, plot_confidence_interval,
    plot_comparison
)
from apps.utils.image_handler import get_house_image_url
from apps.config import FEATURE_GROUPS

# Page config
st.set_page_config(
    page_title="Ames Housing Price Predictor",
    page_icon=":house:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .house-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: box-shadow 0.3s;
    }
    .house-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .price-display {
        font-size: 2.5rem;
        font-weight: bold;
        color: #28a745;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_house' not in st.session_state:
    st.session_state.selected_house = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'comparison_houses' not in st.session_state:
    st.session_state.comparison_houses = []


def home_page():
    """Home page with overview and navigation."""
    st.markdown('<div class="main-header">Ames Housing Price Predictor</div>', unsafe_allow_html=True)
    
    # Load data
    df = load_house_data()
    
    if df.empty:
        st.error("Could not load house data. Please check data files.")
        return
    
    # Hero section with stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Houses", len(df))
    
    with col2:
        avg_price = df['price'].mean() if 'price' in df.columns else 0
        st.metric("Average Price", f"${avg_price:,.0f}")
    
    with col3:
        min_price = df['price'].min() if 'price' in df.columns else 0
        st.metric("Min Price", f"${min_price:,.0f}")
    
    with col4:
        max_price = df['price'].max() if 'price' in df.columns else 0
        st.metric("Max Price", f"${max_price:,.0f}")
    
    st.divider()
    
    # Quick search
    st.subheader("Quick Search")
    search_term = st.text_input("Search by neighborhood, style, or features", key="home_search")
    
    if search_term:
        filtered = filter_houses(df, {'search_term': search_term})
        if len(filtered) > 0:
            st.success(f"Found {len(filtered)} houses")
            st.dataframe(filtered[['price', 'area', 'Neighborhood', 'House.Style', 'Year.Built']].head(10), use_container_width=True)
        else:
            st.info("No houses found matching your search.")
    
    st.divider()
    
    # Market overview charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution")
        fig = plot_price_distribution(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Neighborhoods by Price")
        fig = plot_neighborhood_prices(df, top_n=10)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Navigation buttons
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Browse Houses", use_container_width=True, type="primary"):
            st.session_state.page = 'browser'
            st.rerun()
    
    with col2:
        if st.button("Predict Price", use_container_width=True, type="primary"):
            st.session_state.page = 'predictor'
            st.rerun()
    
    with col3:
        if st.button("Neighborhood Insights", use_container_width=True):
            st.session_state.page = 'insights'
            st.rerun()


def browser_page():
    """House browser with table and card views."""
    st.markdown('<div class="main-header">House Browser</div>', unsafe_allow_html=True)
    
    df = load_house_data()
    
    if df.empty:
        st.error("Could not load house data.")
        return
    
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        
        neighborhoods = ['All'] + get_neighborhoods(df)
        selected_neighborhood = st.selectbox("Neighborhood", neighborhoods)
        
        house_types = ['All'] + get_house_types(df)
        selected_type = st.selectbox("House Style", house_types)
        
        price_range = st.slider(
            "Price Range",
            min_value=int(df['price'].min()) if 'price' in df.columns else 0,
            max_value=int(df['price'].max()) if 'price' in df.columns else 1000000,
            value=(int(df['price'].min()) if 'price' in df.columns else 0, 
                   int(df['price'].max()) if 'price' in df.columns else 1000000)
        )
        
        year_range = st.slider(
            "Year Built",
            min_value=int(df['Year.Built'].min()) if 'Year.Built' in df.columns else 1900,
            max_value=int(df['Year.Built'].max()) if 'Year.Built' in df.columns else 2020,
            value=(int(df['Year.Built'].min()) if 'Year.Built' in df.columns else 1900,
                   int(df['Year.Built'].max()) if 'Year.Built' in df.columns else 2020)
        )
        
        search_term = st.text_input("Search")
    
    # Apply filters
    filters = {
        'neighborhood': selected_neighborhood if selected_neighborhood != 'All' else None,
        'house_style': selected_type if selected_type != 'All' else None,
        'min_price': price_range[0],
        'max_price': price_range[1],
        'min_year': year_range[0],
        'max_year': year_range[1],
        'search_term': search_term if search_term else None
    }
    
    filtered_df = filter_houses(df, filters)
    
    st.info(f"Showing {len(filtered_df)} of {len(df)} houses")
    
    # View toggle
    view_mode = st.radio("View Mode", ["Table", "Cards"], horizontal=True)
    
    if view_mode == "Table":
        # Table view with AgGrid
        try:
            from streamlit_aggrid import AgGrid, GridOptionsBuilder
            
            gb = GridOptionsBuilder.from_dataframe(
                filtered_df[['price', 'area', 'Neighborhood', 'House.Style', 
                            'Year.Built', 'Bedroom.AbvGr', 'Full.Bath', 'Overall.Qual']].head(100)
            )
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_selection('single')
            gb.configure_column('price', type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=0)
            
            grid_response = AgGrid(
                filtered_df[['price', 'area', 'Neighborhood', 'House.Style', 
                            'Year.Built', 'Bedroom.AbvGr', 'Full.Bath', 'Overall.Qual']].head(100),
                gridOptions=gb.build(),
                height=600,
                theme='streamlit',
                allow_unsafe_jscode=True
            )
            
            if grid_response['selected_rows']:
                selected_idx = grid_response['selected_rows'][0].get('_selectedRowNodeInfo', {}).get('nodeRowIndex', 0)
                if st.button("View Details"):
                    st.session_state.selected_house = filtered_df.iloc[selected_idx].name
                    st.session_state.page = 'details'
                    st.rerun()
        except ImportError:
            # Fallback to regular dataframe
            st.dataframe(
                filtered_df[['price', 'area', 'Neighborhood', 'House.Style', 
                            'Year.Built', 'Bedroom.AbvGr', 'Full.Bath', 'Overall.Qual']].head(100),
                use_container_width=True
            )
    else:
        # Card view
        cols = st.columns(3)
        for card_idx, (df_idx, row) in enumerate(filtered_df.head(30).iterrows()):
            with cols[card_idx % 3]:
                with st.container():
                    st.markdown(f'<div class="house-card">', unsafe_allow_html=True)
                    
                    # House image
                    img_url = get_house_image_url(row.get('House.Style'))
                    st.image(img_url, use_container_width=True)
                    
                    # Key info
                    st.markdown(f"### ${row['price']:,.0f}")
                    st.markdown(f"**{row.get('Neighborhood', 'N/A')}**")
                    st.markdown(f"Area: {row.get('area', 0):,.0f} sqft")
                    st.markdown(f"Bedrooms: {row.get('Bedroom.AbvGr', 0)} | Baths: {row.get('Full.Bath', 0)}")
                    st.markdown(f"Year: {row.get('Year.Built', 'N/A')}")
                    
                    if st.button("View Details", key=f"view_{df_idx}_{card_idx}"):
                        # Use the actual index from dataframe
                        st.session_state.selected_house = df_idx
                        st.session_state.page = 'details'
                        st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Export button
    if st.button("Export Filtered Results"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="filtered_houses.csv",
            mime="text/csv"
        )


def details_page():
    """House detail page."""
    if st.session_state.selected_house is None:
        st.warning("No house selected. Please go back to browser.")
        if st.button("Back to Browser"):
            st.session_state.page = 'browser'
            st.rerun()
        return
    
    df = load_house_data()
    house_idx = st.session_state.selected_house
    
    # Handle different index types
    if df.empty:
        st.error("No data available.")
        return
    
    # Try to find the house
    try:
        if house_idx in df.index:
            house = df.loc[house_idx]
        else:
            # Try as integer position
            try:
                house_idx = int(house_idx)
                if 0 <= house_idx < len(df):
                    house = df.iloc[house_idx]
                    house_idx = df.index[house_idx]
                else:
                    st.error("House not found.")
                    return
            except:
                st.error("House not found.")
                return
    except Exception as e:
        st.error(f"Error loading house: {e}")
        return
    
    # Back button
    if st.button("← Back to Browser"):
        st.session_state.page = 'browser'
        st.rerun()
    
    st.markdown(f'<div class="main-header">House Details</div>', unsafe_allow_html=True)
    
    # House image
    img_url = get_house_image_url(house.get('House.Style'))
    st.image(img_url, use_container_width=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Price", f"${house['price']:,.0f}")
    
    with col2:
        st.metric("Area", f"{house.get('area', 0):,.0f} sqft")
    
    with col3:
        st.metric("Year Built", int(house.get('Year.Built', 0)))
    
    with col4:
        st.metric("Overall Quality", house.get('Overall.Qual', 'N/A'))
    
    st.divider()
    
    # Detailed information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        info_data = {
            'Neighborhood': house.get('Neighborhood', 'N/A'),
            'House Style': house.get('House.Style', 'N/A'),
            'MS Zoning': house.get('MS.Zoning', 'N/A'),
            'Lot Area': f"{house.get('Lot.Area', 0):,.0f} sqft",
            'Year Remodeled': int(house.get('Year.Remod.Add', 0)),
            'Overall Condition': house.get('Overall.Cond', 'N/A'),
        }
        for key, value in info_data.items():
            st.write(f"**{key}:** {value}")
        
        st.subheader("Features")
        features_data = {
            'Bedrooms': house.get('Bedroom.AbvGr', 0),
            'Full Baths': house.get('Full.Bath', 0),
            'Half Baths': house.get('Half.Bath', 0),
            'Total Rooms': house.get('TotRms.AbvGrd', 0),
            'Kitchens': house.get('Kitchen.AbvGr', 0),
            'Fireplaces': house.get('Fireplaces', 0),
        }
        for key, value in features_data.items():
            st.write(f"**{key}:** {value}")
    
    with col2:
        st.subheader("Quality Ratings")
        quality_data = {
            'Overall Quality': house.get('Overall.Qual', 'N/A'),
            'Overall Condition': house.get('Overall.Cond', 'N/A'),
            'Exterior Quality': house.get('Exter.Qual', 'N/A'),
            'Kitchen Quality': house.get('Kitchen.Qual', 'N/A'),
            'Basement Quality': house.get('Bsmt.Qual', 'N/A'),
            'Garage Quality': house.get('Garage.Qual', 'N/A'),
        }
        for key, value in quality_data.items():
            st.write(f"**{key}:** {value}")
        
        st.subheader("Garage & Basement")
        garage_data = {
            'Garage Type': house.get('Garage.Type', 'N/A'),
            'Garage Cars': house.get('Garage.Cars', 0),
            'Garage Area': f"{house.get('Garage.Area', 0):,.0f} sqft",
            'Basement Area': f"{house.get('Total.Bsmt.SF', 0):,.0f} sqft",
        }
        for key, value in garage_data.items():
            st.write(f"**{key}:** {value}")
    
    st.divider()
    
    # Similar houses
    st.subheader("Similar Houses")
    similar = get_similar_houses(df, house_idx, n=5)
    if not similar.empty:
        display_cols = ['price', 'area', 'Neighborhood', 'House.Style', 'Year.Built']
        available_cols = [col for col in display_cols if col in similar.columns]
        if available_cols:
            st.dataframe(
                similar[available_cols],
                use_container_width=True
            )
    
    # Comparison
    if st.button("Add to Comparison"):
        if house_idx not in st.session_state.comparison_houses:
            st.session_state.comparison_houses.append(house_idx)
            st.success("Added to comparison!")
        else:
            st.info("House already in comparison.")


def predictor_page():
    """Price predictor page."""
    st.markdown('<div class="main-header">Price Predictor</div>', unsafe_allow_html=True)
    
    df = load_house_data()
    model_info = get_model_info()
    
    # Check API health
    if not check_api_health():
        st.error("API is not available. Please make sure the Flask API is running on port 5000.")
        st.info("Start the API with: `python -m src.serving.api`")
        return
    
    # Feature input form
    with st.form("prediction_form"):
        st.subheader("House Features")
        
        # Basic Info
        with st.expander("Basic Information", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                area = st.number_input("Living Area (sqft)", min_value=0, value=1500, step=100)
                lot_area = st.number_input("Lot Area (sqft)", min_value=0, value=10000, step=100)
            with col2:
                year_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=2000)
                year_remod = st.number_input("Year Remodeled", min_value=1800, max_value=2024, value=2000)
        
        # Quality
        with st.expander("Quality Ratings"):
            col1, col2 = st.columns(2)
            with col1:
                overall_qual = st.slider("Overall Quality", 1, 10, 7)
                overall_cond = st.slider("Overall Condition", 1, 10, 5)
                exter_qual = st.selectbox("Exterior Quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=2)
            with col2:
                kitchen_qual = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=2)
                bsmt_qual = st.selectbox("Basement Quality", ["Ex", "Gd", "TA", "Fa", "Po", "None"], index=2)
                garage_qual = st.selectbox("Garage Quality", ["Ex", "Gd", "TA", "Fa", "Po", "None"], index=2)
        
        # Location
        with st.expander("Location"):
            neighborhoods = get_neighborhoods(df)
            neighborhood = st.selectbox("Neighborhood", neighborhoods if neighborhoods else ["NAmes"])
            # Get available zoning types from data
            zoning_types = get_zoning_types(df)
            ms_zoning = st.selectbox("MS Zoning", zoning_types if zoning_types else ["RL", "RM", "FV", "RH", "C (all)"], index=0)
        
        # Features
        with st.expander("Features"):
            col1, col2 = st.columns(2)
            with col1:
                bedrooms = st.number_input("Bedrooms", min_value=0, value=3, step=1)
                full_bath = st.number_input("Full Bathrooms", min_value=0, value=2, step=1)
                half_bath = st.number_input("Half Bathrooms", min_value=0, value=0, step=1)
            with col2:
                garage_cars = st.number_input("Garage Cars", min_value=0, value=2, step=1)
                garage_area = st.number_input("Garage Area (sqft)", min_value=0, value=500, step=50)
                fireplaces = st.number_input("Fireplaces", min_value=0, value=1, step=1)
        
        # Submit button
        submitted = st.form_submit_button("Predict Price", type="primary", use_container_width=True)
    
    if submitted:
        # Get encodings for categorical features
        encodings = get_categorical_encodings()
        
        # Prepare features dictionary
        features = {
            'area': float(area),
            'Lot.Area': float(lot_area),
            'Year.Built': float(year_built),
            'Year.Remod.Add': float(year_remod),
            'Overall.Qual': float(overall_qual),
            'Overall.Cond': float(overall_cond),
            'Bedroom.AbvGr': float(bedrooms),
            'Full.Bath': float(full_bath),
            'Half.Bath': float(half_bath),
            'Garage.Cars': float(garage_cars),
            'Garage.Area': float(garage_area),
            'Fireplaces': float(fireplaces),
        }
        
        # Map quality ratings to numeric
        quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
        features['Exter.Qual'] = quality_map.get(exter_qual, 3)
        features['Kitchen.Qual'] = quality_map.get(kitchen_qual, 3)
        features['Bsmt.Qual'] = quality_map.get(bsmt_qual, 3)
        features['Garage.Qual'] = quality_map.get(garage_qual, 3)
        
        # Encode categorical features
        neighborhood_encoded = encode_categorical_feature('Neighborhood', neighborhood, encodings)
        if neighborhood_encoded is not None:
            features['Neighborhood'] = neighborhood_encoded
        else:
            st.warning(f"Could not encode Neighborhood '{neighborhood}'. Using average encoded value.")
            # Try to get average encoded value as fallback
            if 'Neighborhood' in encodings and 'mapping' in encodings['Neighborhood']:
                mapping = encodings['Neighborhood']['mapping']
                if mapping:
                    avg_encoded = sum(mapping.values()) / len(mapping)
                    features['Neighborhood'] = avg_encoded
                else:
                    features['Neighborhood'] = 0.0
            else:
                features['Neighborhood'] = 0.0
        
        # MS.Zoning is one-hot encoded - set the appropriate column to 1.0, others to 0.0
        ms_zoning_encoded = encode_categorical_feature('MS.Zoning', ms_zoning, encodings)
        if ms_zoning_encoded is not None and isinstance(ms_zoning_encoded, dict):
            # For one-hot encoding, set the matching column to 1.0 and others to 0.0
            if 'MS.Zoning' in encodings and 'columns' in encodings['MS.Zoning']:
                # Set all MS.Zoning columns to 0 first
                for col in encodings['MS.Zoning']['columns']:
                    features[col] = 0.0
                # Then set the matching column to 1.0
                features.update(ms_zoning_encoded)
        else:
            st.info(f"ℹ️ MS.Zoning '{ms_zoning}' encoding not found. Model will use default (0) for all MS.Zoning columns.")
            # Still set all MS.Zoning columns to 0 to avoid errors
            if 'MS.Zoning' in encodings and 'columns' in encodings['MS.Zoning']:
                for col in encodings['MS.Zoning']['columns']:
                    features[col] = 0.0
        
        # Make prediction with enhanced loading state
        with st.spinner("Finding similar houses and calculating prediction..."):
            result = predict_price(features)
        
        if result:
            prediction = result.get('prediction', 0)
            conf_interval = result.get('confidence_interval', {})
            lower = conf_interval.get('lower', prediction * 0.9)
            upper = conf_interval.get('upper', prediction * 1.1)
            
            # Display prediction
            st.divider()
            st.markdown(f'<div class="price-display">${prediction:,.0f}</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Lower Bound", f"${lower:,.0f}")
            with col2:
                st.metric("Upper Bound", f"${upper:,.0f}")
            
            # Confidence interval plot
            fig = plot_confidence_interval(prediction, lower, upper)
            st.plotly_chart(fig, use_container_width=True)
            
            # Save to history
            st.session_state.prediction_history.append({
                'features': features,
                'prediction': prediction,
                'confidence': (lower, upper)
            })


def insights_page():
    """Neighborhood insights page."""
    st.markdown('<div class="main-header">Neighborhood Insights</div>', unsafe_allow_html=True)
    
    df = load_house_data()
    
    if df.empty:
        st.error("Could not load house data.")
        return
    
    # Neighborhood statistics
    stats = get_neighborhood_stats(df)
    
    if not stats.empty:
        st.subheader("Neighborhood Statistics")
        st.dataframe(stats, use_container_width=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_neighborhood_prices(df, top_n=20)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'area' in df.columns:
                fig = plot_price_vs_feature(df, 'area')
                st.plotly_chart(fig, use_container_width=True)


def comparison_page():
    """House comparison page."""
    st.markdown('<div class="main-header">⚖️ House Comparison</div>', unsafe_allow_html=True)
    
    if not st.session_state.comparison_houses:
        st.info("No houses selected for comparison. Go to house details to add houses.")
        return
    
    df = load_house_data()
    houses_data = []
    
    for house_idx in st.session_state.comparison_houses[:3]:  # Limit to 3
        try:
            if house_idx in df.index:
                houses_data.append(df.loc[house_idx])
            else:
                # Try as integer position
                try:
                    idx = int(house_idx)
                    if 0 <= idx < len(df):
                        houses_data.append(df.iloc[idx])
                except:
                    pass
        except:
            pass
    
    if houses_data:
        comparison_df = pd.DataFrame(houses_data)
        display_cols = ['price', 'area', 'Neighborhood', 'House.Style', 
                       'Year.Built', 'Bedroom.AbvGr', 'Full.Bath', 'Overall.Qual']
        available_cols = [col for col in display_cols if col in comparison_df.columns]
        if available_cols:
            st.dataframe(comparison_df[available_cols], use_container_width=True)
        
        # Comparison chart
        fig = plot_comparison(comparison_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Clear Comparison"):
            st.session_state.comparison_houses = []
            st.rerun()


# Main navigation
def main():
    """Main app function."""
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        
        pages = {
            "Home": "home",
            "Browse Houses": "browser",
            "Predict Price": "predictor",
            "Neighborhood Insights": "insights",
            "Compare Houses": "comparison"
        }
        
        for page_name, page_key in pages.items():
            if st.button(page_name, use_container_width=True, 
                        type="primary" if st.session_state.page == page_key else "secondary"):
                st.session_state.page = page_key
                st.rerun()
        
        st.divider()
        
        # API status
        if check_api_health():
            st.success("API Connected")
        else:
            st.error("API Disconnected")
    
    # Route to appropriate page
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'browser':
        browser_page()
    elif st.session_state.page == 'details':
        details_page()
    elif st.session_state.page == 'predictor':
        predictor_page()
    elif st.session_state.page == 'insights':
        insights_page()
    elif st.session_state.page == 'comparison':
        comparison_page()


if __name__ == "__main__":
    main()
