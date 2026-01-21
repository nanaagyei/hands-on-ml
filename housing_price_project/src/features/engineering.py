# src/features/engineering.py
"""
Feature Engineering for Ames Housing Dataset.

This is where DOMAIN KNOWLEDGE becomes code.
These features aren't in the raw data — we CREATE them
based on understanding of what drives house prices.
"""

import numpy as np


class AmesFeaturesEngineeer:
    """
    Create domain-specific features for Ames Housing data.
    
    Feature Categories:
    1. Age features (how old is the house?)
    2. Area features (total square footage, ratios)
    3. Quality features (interactions, aggregations)
    4. Binary flags (has garage? has pool?)
    5. Temporal features (season sold, years since remodel)
    """
    
    def __init__(self, reference_year=2010):
        """
        Parameters
        ----------
        reference_year : int
            Year to use for age calculations (latest sale year in data)
        """
        self.reference_year = reference_year
        self.feature_names_ = []
        self._is_fitted = False
    
    def fit(self, X, y=None):
        """
        Fit doesn't learn anything for most features,
        but maintains consistent API.
        """
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """
        Create engineered features.
        
        Parameters
        ----------
        X : DataFrame
            Must have the expected Ames columns
            
        Returns
        -------
        X_new : DataFrame
            Original data + new features
        """
        import pandas as pd
        
        X_new = X.copy()
        
        # =====================================================================
        # AGE FEATURES
        # =====================================================================
        
        # House age at time of sale
        if 'Year.Built' in X.columns and 'Yr.Sold' in X.columns:
            X_new['House_Age'] = X['Yr.Sold'] - X['Year.Built']
            X_new['House_Age'] = X_new['House_Age'].clip(lower=0)  # No negative ages
        
        # Years since remodel (0 if never remodeled)
        if 'Year.Remod.Add' in X.columns and 'Yr.Sold' in X.columns:
            X_new['Years_Since_Remodel'] = X['Yr.Sold'] - X['Year.Remod.Add']
            X_new['Years_Since_Remodel'] = X_new['Years_Since_Remodel'].clip(lower=0)
        
        # Was house remodeled? (remodel year different from built year)
        if 'Year.Built' in X.columns and 'Year.Remod.Add' in X.columns:
            X_new['Was_Remodeled'] = (X['Year.Remod.Add'] != X['Year.Built']).astype(int)
        
        # Is house new? (sold same year as built)
        if 'Year.Built' in X.columns and 'Yr.Sold' in X.columns:
            X_new['Is_New'] = (X['Yr.Sold'] == X['Year.Built']).astype(int)
        
        # =====================================================================
        # AREA FEATURES  
        # =====================================================================
        
        # Total square footage (all livable space)
        area_cols = ['area', 'Total.Bsmt.SF']  # 'area' is above grade living area
        if all(c in X.columns for c in area_cols):
            X_new['Total_SF'] = X['area'] + X['Total.Bsmt.SF'].fillna(0)
        
        # Total porch area
        porch_cols = ['Open.Porch.SF', 'Enclosed.Porch', 'X3Ssn.Porch', 'Screen.Porch']
        existing_porch = [c for c in porch_cols if c in X.columns]
        if existing_porch:
            X_new['Total_Porch_SF'] = X[existing_porch].fillna(0).sum(axis=1)
        
        # Total outdoor area (porch + deck)
        if 'Total_Porch_SF' in X_new.columns and 'Wood.Deck.SF' in X.columns:
            X_new['Total_Outdoor_SF'] = X_new['Total_Porch_SF'] + X['Wood.Deck.SF'].fillna(0)
        
        # Basement finished ratio (what % of basement is finished?)
        if 'Total.Bsmt.SF' in X.columns and 'Bsmt.Unf.SF' in X.columns:
            total_bsmt = X['Total.Bsmt.SF'].fillna(0)
            unf_bsmt = X['Bsmt.Unf.SF'].fillna(0)
            # Avoid division by zero
            X_new['Bsmt_Finished_Ratio'] = np.where(
                total_bsmt > 0,
                (total_bsmt - unf_bsmt) / total_bsmt,
                0
            )
        
        # Above grade vs total ratio (how much is above ground?)
        if 'area' in X.columns and 'Total_SF' in X_new.columns:
            X_new['Above_Grade_Ratio'] = np.where(
                X_new['Total_SF'] > 0,
                X['area'] / X_new['Total_SF'],
                1
            )
        
        # =====================================================================
        # BATHROOM FEATURES
        # =====================================================================
        
        bath_cols = ['Full.Bath', 'Half.Bath', 'Bsmt.Full.Bath', 'Bsmt.Half.Bath']
        if all(c in X.columns for c in bath_cols):
            # Total bathrooms (half baths count as 0.5)
            X_new['Total_Bathrooms'] = (
                X['Full.Bath'].fillna(0) + 
                X['Bsmt.Full.Bath'].fillna(0) + 
                0.5 * X['Half.Bath'].fillna(0) + 
                0.5 * X['Bsmt.Half.Bath'].fillna(0)
            )
        
        # =====================================================================
        # QUALITY FEATURES
        # =====================================================================
        
        # Overall quality × condition interaction
        if 'Overall.Qual' in X.columns and 'Overall.Cond' in X.columns:
            X_new['Qual_Cond_Product'] = X['Overall.Qual'] * X['Overall.Cond']
            X_new['Qual_Cond_Sum'] = X['Overall.Qual'] + X['Overall.Cond']
        
        # Quality per square foot (is high quality justified by size?)
        if 'Overall.Qual' in X.columns and 'area' in X.columns:
            X_new['Qual_Per_SF'] = X['Overall.Qual'] / (X['area'] / 1000)  # per 1000 SF
        
        # =====================================================================
        # GARAGE FEATURES
        # =====================================================================
        
        # Has garage?
        if 'Garage.Cars' in X.columns:
            X_new['Has_Garage'] = (X['Garage.Cars'].fillna(0) > 0).astype(int)
        
        # Garage area per car (efficiency)
        if 'Garage.Area' in X.columns and 'Garage.Cars' in X.columns:
            garage_cars = X['Garage.Cars'].fillna(0)
            X_new['Garage_Area_Per_Car'] = np.where(
                garage_cars > 0,
                X['Garage.Area'].fillna(0) / garage_cars,
                0
            )
        
        # =====================================================================
        # BINARY FLAGS
        # =====================================================================
        
        # Has pool?
        if 'Pool.Area' in X.columns:
            X_new['Has_Pool'] = (X['Pool.Area'].fillna(0) > 0).astype(int)
        
        # Has fireplace?
        if 'Fireplaces' in X.columns:
            X_new['Has_Fireplace'] = (X['Fireplaces'].fillna(0) > 0).astype(int)
        
        # Has 2nd floor?
        if 'X2nd.Flr.SF' in X.columns:
            X_new['Has_2nd_Floor'] = (X['X2nd.Flr.SF'].fillna(0) > 0).astype(int)
        
        # Has basement?
        if 'Total.Bsmt.SF' in X.columns:
            X_new['Has_Basement'] = (X['Total.Bsmt.SF'].fillna(0) > 0).astype(int)
        
        # Has central air?
        if 'Central.Air' in X.columns:
            X_new['Has_Central_Air'] = (X['Central.Air'] == 'Y').astype(int)
        
        # =====================================================================
        # TEMPORAL FEATURES
        # =====================================================================
        
        # Season sold (might affect price)
        if 'Mo.Sold' in X.columns:
            month = X['Mo.Sold']
            X_new['Sold_Spring'] = month.isin([3, 4, 5]).astype(int)
            X_new['Sold_Summer'] = month.isin([6, 7, 8]).astype(int)
            X_new['Sold_Fall'] = month.isin([9, 10, 11]).astype(int)
            X_new['Sold_Winter'] = month.isin([12, 1, 2]).astype(int)
        
        # =====================================================================
        # NEIGHBORHOOD FEATURES (if target encoding not used)
        # =====================================================================
        
        # These would typically come from target encoding
        # But we can create neighborhood-based features differently
        
        self.feature_names_ = [c for c in X_new.columns if c not in X.columns]
        
        return X_new
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    def get_new_feature_names(self):
        """Return list of newly created features."""
        return self.feature_names_.copy()


class PolynomialFeatures:
    """
    Generate polynomial and interaction features.
    
    For features [a, b]:
    - degree=2: [a, b, a², ab, b²]
    - interaction_only=True: [a, b, ab]
    
    ⚠️ Warning: Feature explosion!
    - 10 features, degree=2 → 65 features
    - 10 features, degree=3 → 285 features!
    
    Use sparingly on carefully selected features.
    """
    
    def __init__(self, degree=2, interaction_only=False, include_bias=False):
        """
        Parameters
        ----------
        degree : int
            Maximum polynomial degree
        interaction_only : bool
            If True, only create interaction terms (no x², x³, etc.)
        include_bias : bool
            If True, include a constant column of 1s
        """
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        
        self.n_input_features_ = None
        self.n_output_features_ = None
        self._is_fitted = False
    
    def fit(self, X, y=None):
        """Learn the number of input features."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_input_features_ = X.shape[1]
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """Generate polynomial features."""
        if not self._is_fitted:
            raise RuntimeError("Not fitted.")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        if n_features != self.n_input_features_:
            raise ValueError(f"Expected {self.n_input_features_} features, got {n_features}")
        
        # Start with original features (or bias)
        features = [np.ones((n_samples, 1))] if self.include_bias else []
        features.append(X)
        
        # Generate higher degree terms
        if self.degree >= 2:
            if self.interaction_only:
                # Only interactions (x_i * x_j for i < j)
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        features.append((X[:, i] * X[:, j]).reshape(-1, 1))
            else:
                # Full polynomial
                # Degree 2: x_i² and x_i * x_j
                for i in range(n_features):
                    for j in range(i, n_features):
                        features.append((X[:, i] * X[:, j]).reshape(-1, 1))
                
                # Higher degrees if requested
                if self.degree >= 3:
                    for i in range(n_features):
                        for j in range(i, n_features):
                            for k in range(j, n_features):
                                features.append((X[:, i] * X[:, j] * X[:, k]).reshape(-1, 1))
        
        result = np.hstack(features)
        self.n_output_features_ = result.shape[1]
        
        return result
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class LogTransformer:
    """
    Apply log transformation to skewed features.
    
    Formula: log(x + 1) to handle zeros
    
    Why log transform?
    - Makes right-skewed distributions more normal
    - Stabilizes variance
    - Can improve linear model performance
    
    Common in housing data: price, area, lot size
    """
    
    def __init__(self, columns=None):
        """
        Parameters
        ----------
        columns : list, optional
            Specific columns to transform.
            If None, transforms all numeric columns.
        """
        self.columns = columns
        self.transformed_columns_ = []
        self._is_fitted = False
    
    def fit(self, X, y=None):
        """Identify columns to transform."""
        import pandas as pd
        
        if self.columns is not None:
            self.transformed_columns_ = self.columns
        else:
            # Auto-detect skewed numeric columns
            self.transformed_columns_ = []
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    skewness = X[col].skew()
                    if abs(skewness) > 0.5:  # Moderately or highly skewed
                        self.transformed_columns_.append(col)
        
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """Apply log(x + 1) transformation."""
        import pandas as pd
        
        if not self._is_fitted:
            raise RuntimeError("Not fitted.")
        
        X_new = X.copy()
        
        for col in self.transformed_columns_:
            if col in X_new.columns:
                # log(x + 1) handles zeros safely
                X_new[col] = np.log1p(X_new[col].clip(lower=0))
        
        return X_new
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """Reverse the log transformation: exp(x) - 1"""
        import pandas as pd
        
        X_new = X.copy()
        
        for col in self.transformed_columns_:
            if col in X_new.columns:
                X_new[col] = np.expm1(X_new[col])
        
        return X_new