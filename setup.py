from setuptools import setup, find_packages

setup(
    name="handover-prediction",
    version="0.1.0",
    description="A machine learning system for predicting cellular network handovers based on user mobility patterns",
    author="UnifyAir",
    author_email="hello@unifyair.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23.5,<2.0.0",
        "pandas>=2.2.3",
        "matplotlib>=3.10.1",
        "tensorflow>=2.15.0",
        "scikit-learn>=1.3.0",
        "seaborn>=0.13.0",
        "joblib>=1.3.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine-learning mobility-prediction handover-prediction cellular-networks",
)
