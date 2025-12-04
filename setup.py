"""
PySenseDF - AI-Powered Native Python DataFrame
==============================================

Setup configuration for PyPI distribution
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pysensedf",
    version="0.4.0",
    author="Idriss Bado",
    author_email="idrissbadoolivier@gmail.com",
    description="AI-powered Python DataFrame that beats Pandas - 27-92x faster with NumPy backend, smart caching, parallel processing, natural language queries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/idrissbado/PySenseDF",
    packages=find_packages(exclude=("tests", "examples")),
    package_data={
        "pysensedf.datasets": ["*.csv"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "perf": [
            "numpy>=1.21.0",
            "numba>=0.55.0",
        ],
        "ml": [
            "scikit-learn>=1.0.0",
            "xgboost>=1.5.0",
        ],
        "ai": [
            "transformers>=4.20.0",
            "torch>=1.12.0",
        ],
        "cloud": [
            "boto3>=1.20.0",
            "azure-storage-blob>=12.0.0",
            "google-cloud-storage>=2.0.0",
        ],
        "data": [
            "pyarrow>=6.0.0",
            "openpyxl>=3.0.0",
            "sqlalchemy>=1.4.0",
        ],
        "pipeline": [
            "pipelinescript>=0.1.0",
        ],
        "full": [
            "numpy>=1.21.0",
            "numba>=0.55.0",
            "scikit-learn>=1.0.0",
            "pyarrow>=6.0.0",
            "openpyxl>=3.0.0",
            "sqlalchemy>=1.4.0",
            "pipelinescript>=0.1.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    keywords="dataframe pandas data-analysis ai ml sql lazy-evaluation data-science",
    project_urls={
        "Documentation": "https://github.com/idrissbado/PySenseDF/blob/main/README.md",
        "Source": "https://github.com/idrissbado/PySenseDF",
        "Tracker": "https://github.com/idrissbado/PySenseDF/issues",
    },
)
