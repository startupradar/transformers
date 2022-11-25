from setuptools import setup, find_packages

setup(
    name="startupradar-transformers",
    version="0.1.0",
    packages=find_packages(include=["startupradar.*"]),
    install_requires=[
        "pandas",
        "scikit-learn",
        "requests",
        "black",
        "tldextract",
        "flake8",
        "pytest",
        "numpy",
        "cachecontrol[filecache]",
    ],
)
