# Feature Defaults Strategy - Using Training Data

## Overview

We use two CSV files to improve predictions when features are missing:

1. **`ames_cleaned.csv`** (80 columns): Original cleaned data with categorical strings
2. **`ames_featured.csv`** (206 columns = 204 features + 2 targets): Feature-engineered, all numeric, ready for model

## How They're Used

### 1. Feature Defaults from `ames_featured.csv`

**Purpose**: Provide realistic default values for missing features using actual training data medians.

**Implementation**:
- Load `ames_featured.csv` (the same data the model was trained on)
- Compute **median** for each of the 204 features
- Use these medians as defaults for missing features

**Why medians instead of scaler mean?**
- Medians are actual feature values from training data
- More robust to outliers
- Better represents "typical" house characteristics
- Scaler mean is the mean of **scaled** features, which may not be as intuitive

### 2. Categorical Encoding from Both Files

**Purpose**: Map user-friendly categorical strings to model-expected numeric values.

**Implementation**:
- Use `ames_cleaned.csv` to get original categorical values (e.g., "RL", "NAmes")
- Use `ames_featured.csv` to get encoded values (e.g., target-encoded Neighborhood, one-hot MS.Zoning columns)
- Create mappings between original and encoded values

**Example**:
- User selects: `Neighborhood = "NAmes"`
- System looks up: `ames_cleaned.csv` has "NAmes" → `ames_featured.csv` has encoded value `145697.148`
- System uses: `145697.148` for the Neighborhood feature

## Priority Order for Missing Features

When a feature is missing, we use defaults in this order:

1. **Feature medians from `ames_featured.csv`** (best - actual training data values)
2. **Scaler mean** (fallback - mean of scaled features)
3. **Zero** (last resort)

## Benefits

1. **More Realistic Predictions**: Using actual training data medians gives more realistic feature combinations
2. **Better Handling of Missing Features**: 88% missing features is still problematic, but medians help
3. **Fallback Protection**: If featured CSV unavailable, falls back to scaler mean, then zeros

## Current Status

✅ **Implemented**: Feature defaults from `ames_featured.csv`  
✅ **Implemented**: Categorical encoding using both files  
✅ **Implemented**: Fallback chain (data defaults → scaler mean → zeros)  
✅ **Implemented**: Logging to track which default method is used

## Expected Improvement

With feature medians from training data:
- Predictions should be more stable
- Log predictions should be closer to reasonable range (9-13.5 instead of 1200+)
- Should reduce need for fallback ($180,000) predictions

## Testing

After restarting the Flask API, test with the recommended feature values from `RECOMMENDED_FEATURE_VALUES.md`. You should see:
- More reasonable predictions (not always $180,000)
- Log predictions in reasonable range (≤ 15)
- Better prediction accuracy even with many missing features
