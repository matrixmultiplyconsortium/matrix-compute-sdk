"""Fallback setup.py for editable installation and legacy tooling."""
from setuptools import setup, find_packages

setup(
    name="matrix-compute-sdk",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
