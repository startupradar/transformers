from setuptools import setup, find_namespace_packages

setup(
    name="startupradar-transformers",
    version="0.2.0",
    packages=find_namespace_packages(include=["startupradar.*"]),
    install_requires=[
        "pandas",
        "scikit-learn>=1.2",
        "requests",
        "tldextract",
        "numpy",
        "minimalkv",
    ],
)
