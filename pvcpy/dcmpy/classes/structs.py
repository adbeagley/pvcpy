"""Module to contain helper objects"""

__all__ = [
    "StackDims",
    "IjkSpacing",
    "XyzSpacing",
    "DirectionCosines",
    "PixelSpacing"
]

from typing import NamedTuple
import numpy as np


class StackDims(NamedTuple):
    """Named Tuple holding dimensions along i, j, k image dimensions"""
    rows: int
    cols: int
    slices: int


class IjkSpacing(NamedTuple):
    """Named Tuple holding spacing values for image i, j, k dimensions"""
    i: float
    j: float
    k: float


class XyzSpacing(NamedTuple):
    """Named Tuple holding spacing values for global x, y, z dimensions"""
    x: float
    y: float
    z: float


class DirectionCosines(NamedTuple):
    """Direction cosines for first row and column of pixel data"""
    row: np.ndarray
    col: np.ndarray


class PixelSpacing(NamedTuple):
    """In-Plane spacing of pixels"""
    row: float
    col: float
