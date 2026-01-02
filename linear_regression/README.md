# Linear Regression Package

A from-scratch implementation of linear regression using NumPy.

## Installation

### Option 1: Install in Editable Mode (Recommended)

This is the best approach if you want to use the package from any directory:

```bash
# Activate your virtual environment first
source ../venv/bin/activate  # On macOS/Linux
# or
..\venv\Scripts\activate  # On Windows

# Navigate to the linear_regression directory
cd /path/to/hands-on-ml/linear_regression

# Install in editable mode
pip install -e .
```

After installation, you can import from anywhere:
```python
from core.linear_regression import SimpleLinearRegression
```

### Option 2: Use the Notebook's Built-in Path Resolution

The notebook (`notebooks/01_fundamentals.ipynb`) includes automatic path resolution that works from any directory. Just run the notebook - it will automatically find and add the `linear_regression` directory to Python's path.

## Usage

```python
from core.linear_regression import SimpleLinearRegression
import numpy as np

# Create and train the model
model = SimpleLinearRegression()
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(f"R² score: {model.score(X, y)}")
```

## Project Structure

```
linear_regression/
├── core/
│   ├── __init__.py
│   ├── linear_regression.py  # SimpleLinearRegression class
│   ├── metrics.py
│   ├── optimizers.py
│   └── regularized.py
├── notebooks/
│   └── 01_fundamentals.ipynb
├── preprocessing/
├── tests/
├── utils/
└── setup.py
```

