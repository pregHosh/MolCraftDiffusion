"""
MolecularDiffusion - A molecular diffusion framework.

This package provides tools and models for molecular diffusion processes.
"""

__version__ = "0.1.0"
__author__ = "Thanapat Worakul"
__email__ = "thanapat.worakul@epfl.ch"

# Import main modules to make them available at package level
from . import core
from . import data
from . import modules
from . import utils
from . import callbacks
from . import runmodes

__all__ = [
    "core",
    "data", 
    "modules",
    "utils",
    "callbacks",
    "runmodes"
] 