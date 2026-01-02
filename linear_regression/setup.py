"""
Setup file for linear_regression package.
Install in editable mode with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="linear-regression",
    version="0.1.0",
    description="Linear regression implementation from scratch",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
)
