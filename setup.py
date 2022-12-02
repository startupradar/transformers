from setuptools import setup, find_namespace_packages

setup(
    name="startupradar-transformers",
    version="0.1.0",
    packages=find_namespace_packages(include=["startupradar.*"]),
    install_requires=[
        "pandas",
        "scikit-learn",
        "requests",
        "tldextract",
        "numpy",
        "cachecontrol[filecache]",
    ],
)
