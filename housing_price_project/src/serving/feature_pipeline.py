"""
Feature engineering pipeline for predictions.
Takes user inputs and generates all 204 features expected by the model.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
_main_root = _project_root.parent

if str(_main_root) not in sys.path:
    sys.path.insert(0, str(_main_root))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.features.engineering import AmesFeaturesEngineeer, LogTransformer
from src.features.encoders import TargetEncoder, OneHotEncoder, OrdinalEncoder


def engineer_features_from_user_input(user_features, featured_data_path=None, reference_year=2010):
    """
    Take user-provided features and generate all 204 features expected by the model.
    
    This replicates the feature engineering pipeline from training:
    1. Ordinal encoding
    2. Target encoding (Neighborhood)
    3. One-hot encoding (MS.Zoning, etc.)
    4. Feature engineering (House_Age, Total_SF, etc.)
    5. Log transformation (area features)
    
    Parameters
    ----------
    user_features : dict
        User-provided features (e.g., {'area': 1500, 'Overall.Qual': 7, ...})
    featured_data_path : Path, optional
        Path to ames_featured.csv for getting encodings
    reference_year : int
        Reference year for age calculations (default: 2010)
        
    Returns
    -------
    features_dict : dict
        All 204 features in the format expected by the model
    """
    # Load featured data to get encodings if available
    if featured_data_path is None:
        featured_data_path = _project_root / "data" / "features" / "ames_featured.csv"
    
    featured_df = None
    cleaned_df = None
    if featured_data_path.exists():
        try:
            featured_df = pd.read_csv(featured_data_path)
            # Also load cleaned data for encoding mappings
            cleaned_path = _project_root / "data" / "processed" / "ames_cleaned.csv"
            if cleaned_path.exists():
                cleaned_df = pd.read_csv(cleaned_path)
        except Exception as e:
            print(f"Error loading featured data: {e}")
    
    # Start with user features
    features = user_features.copy()
    
    # Set defaults for missing required features
    defaults = {
        'MS.SubClass': 20,
        'Lot.Frontage': 70.0,
        'Lot.Shape': 2,  # IR1
        'Land.Slope': 2,  # Gtl
        'Overall.Cond': 5,
        'Mas.Vnr.Area': 0.0,
        'Exter.Cond': 2,  # TA
        'Bsmt.Cond': 3,  # TA
        'Bsmt.Exposure': 1,  # No
        'BsmtFin.Type.1': 4,  # Unf
        'BsmtFin.SF.1': 0.0,
        'BsmtFin.Type.2': 1,  # Unf
        'BsmtFin.SF.2': 0.0,
        'Bsmt.Unf.SF': 500.0,
        'Total.Bsmt.SF': 1000.0,
        'Heating.QC': 2,  # TA
        'X1st.Flr.SF': features.get('area', 1500),
        'X2nd.Flr.SF': 0.0,
        'Low.Qual.Fin.SF': 0.0,
        'Bsmt.Full.Bath': 0.0,
        'Bsmt.Half.Bath': 0.0,
        'Kitchen.AbvGr': 1,
        'TotRms.AbvGrd': 6,
        'Functional': 7,  # Typ
        'Fireplace.Qu': 0,  # None
        'Garage.Yr.Blt': features.get('Year.Built', 2000),
        'Garage.Finish': 1,  # Unf
        'Garage.Cond': 3,  # TA
        'Paved.Drive': 2,  # Y
        'Wood.Deck.SF': 0.0,
        'Open.Porch.SF': 0.0,
        'Enclosed.Porch': 0.0,
        'X3Ssn.Porch': 0.0,
        'Screen.Porch': 0.0,
        'Pool.Area': 0.0,
        'Pool.QC': 0,  # None
        'Fence': 0,  # None
        'Misc.Val': 0,
        'Mo.Sold': 6,  # June
        'Yr.Sold': reference_year,
        'Street': 'Pave',
        'Alley': 'None',
        'Land.Contour': 'Lvl',
        'Utilities': 'AllPub',
        'Lot.Config': 'Inside',
        'Condition.1': 'Norm',
        'Condition.2': 'Norm',
        'Bldg.Type': '1Fam',
        'House.Style': '1Story',
        'Roof.Style': 'Gable',
        'Roof.Matl': 'CompShg',
        'Exterior.1st': 'VinylSd',
        'Exterior.2nd': 'VinylSd',
        'Mas.Vnr.Type': 'None',
        'Foundation': 'PConc',
        'BsmtFin.Type.1': 'Unf',
        'BsmtFin.Type.2': 'Unf',
        'Heating': 'GasA',
        'Central.Air': 'Y',
        'Electrical': 'SBrkr',
        'Garage.Type': 'Attchd',
        'Misc.Feature': 'None',
        'Sale.Type': 'WD',
        'Sale.Condition': 'Normal',
    }
    
    # Fill in defaults for missing features
    for key, default_val in defaults.items():
        if key not in features:
            features[key] = default_val
    
    # Convert to DataFrame for processing
    df = pd.DataFrame([features])
    
    # Step 1: Ordinal encoding (already numeric in user input, but ensure consistency)
    ordinal_mappings = {
        'Exter.Qual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'Exter.Cond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'Bsmt.Qual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'Bsmt.Cond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'Bsmt.Exposure': ['None', 'No', 'Mn', 'Av', 'Gd'],
        'BsmtFin.Type.1': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
        'BsmtFin.Type.2': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
        'Heating.QC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'Kitchen.Qual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'Functional': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
        'Fireplace.Qu': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'Garage.Finish': ['None', 'Unf', 'RFn', 'Fin'],
        'Garage.Qual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'Garage.Cond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'Paved.Drive': ['N', 'P', 'Y'],
        'Pool.QC': ['None', 'Fa', 'TA', 'Gd', 'Ex'],
        'Fence': ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],
    }
    
    # Convert quality strings to numeric if needed
    for col, mapping in ordinal_mappings.items():
        if col in df.columns and df[col].dtype == 'object':
            # Map string values to numeric
            reverse_map = {val: idx for idx, val in enumerate(mapping)}
            df[col] = df[col].map(reverse_map).fillna(0)
        elif col in df.columns:
            # Already numeric, ensure it's in valid range
            df[col] = df[col].clip(0, len(mapping) - 1)
    
    # Step 2: Target encode Neighborhood (if we have the mapping)
    if 'Neighborhood' in df.columns and featured_df is not None and cleaned_df is not None:
        if df['Neighborhood'].dtype == 'object':
            # Create mapping from cleaned to featured
            neighborhood_map = {}
            for idx in cleaned_df.index:
                if idx < len(featured_df):
                    orig_val = cleaned_df.loc[idx, 'Neighborhood']
                    encoded_val = featured_df.loc[idx, 'Neighborhood']
                    if pd.notna(orig_val) and pd.notna(encoded_val):
                        key = str(orig_val)
                        if key not in neighborhood_map:
                            neighborhood_map[key] = float(encoded_val)
            
            # Apply mapping
            if neighborhood_map:
                df['Neighborhood'] = df['Neighborhood'].astype(str).map(neighborhood_map)
                # Use median if mapping not found
                if df['Neighborhood'].isna().any():
                    median_val = featured_df['Neighborhood'].median()
                    df['Neighborhood'] = df['Neighborhood'].fillna(median_val)
    
    # Step 3: One-hot encode categorical features
    onehot_cols = [
        'MS.Zoning', 'Street', 'Alley', 'Land.Contour', 'Lot.Config',
        'Condition.1', 'Condition.2', 'Bldg.Type', 'House.Style',
        'Roof.Style', 'Roof.Matl', 'Exterior.1st', 'Exterior.2nd',
        'Mas.Vnr.Type', 'Foundation', 'Heating', 'Electrical',
        'Garage.Type', 'Misc.Feature', 'Sale.Type', 'Sale.Condition',
        'Central.Air', 'Utilities'
    ]
    
    # Get all one-hot column names from featured data
    onehot_features = {}
    if featured_df is not None:
        for col in onehot_cols:
            if col in df.columns:
                # Find matching one-hot columns in featured data
                matching_cols = [c for c in featured_df.columns if c.startswith(f'{col}_')]
                if matching_cols:
                    # Initialize all to 0
                    for match_col in matching_cols:
                        onehot_features[match_col] = 0.0
                    
                    # Set the matching column to 1
                    orig_value = str(df[col].iloc[0])
                    for match_col in matching_cols:
                        col_value = match_col.replace(f'{col}_', '')
                        # Try to match
                        if orig_value in col_value or col_value.startswith(orig_value):
                            onehot_features[match_col] = 1.0
                            break
    
    # Step 4: Feature engineering
    engineer = AmesFeaturesEngineeer(reference_year=reference_year)
    df_engineered = engineer.fit_transform(df)
    
    # Step 5: Log transform area features
    area_features = [col for col in df_engineered.columns 
                    if 'SF' in col or 'Area' in col or col == 'area']
    log_transformer = LogTransformer(columns=area_features)
    df_final = log_transformer.fit_transform(df_engineered)
    
    # Convert to dict with all features
    final_features = df_final.iloc[0].to_dict()
    
    # Add one-hot encoded features
    final_features.update(onehot_features)
    
    return final_features
