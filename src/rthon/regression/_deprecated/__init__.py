"""
DEPRECATED MODULE - DO NOT IMPORT

This module contains deprecated Python implementations that have been
replaced by high-performance C extensions. Importing from this module
will raise an ImportError to prevent accidental usage.
"""

raise ImportError(
    "This module contains deprecated Python implementations that have been "
    "replaced by C extensions. Do not import from rthon.regression._deprecated. "
    "All functionality is available through the main rthon.regression module."
)