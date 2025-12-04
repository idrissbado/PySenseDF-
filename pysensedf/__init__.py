"""
PySenseDF - Top-level package
==============================

The revolutionary DataFrame that kills Pandas.
"""

from .core.dataframe import DataFrame
from . import datasets

__version__ = "0.3.0"
__author__ = "Idriss Bado"
__email__ = "idrissbadoolivier@gmail.com"

__all__ = ["DataFrame", "datasets"]
