"""
Setup file for svm package.
Install in editable mode with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="svm",
    version="0.1.0",
    description="Support Vector Machine implementation from scratch",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
)