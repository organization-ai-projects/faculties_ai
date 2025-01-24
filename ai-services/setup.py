from setuptools import find_packages, setup

setup(
    name="ai_services",
    version="0.1.0",
    author="Votre Nom",
    description="Package local pour gérer les services AI",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.5.0",
        "numpy>=1.26",
        "pandas>=2.0",
        "fastapi>=0.80",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "flake8"],
    },
)
