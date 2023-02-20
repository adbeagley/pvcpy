"""Helper functions"""

from typing import Union
import numpy as np
from scipy.ndimage import binary_erosion


def choose_np_dtype_int(sign: int, bits: int):
    """Description:
    Picks 1 of 8 possible numpy data types for integers as
    defined by the platform

    Parameters:
    ----------
    sign: int
        0 for unsigned integer, 1 for two's complement

    bits: int
        the number of bits that should be used for the integer

    Returns:
    --------
    numpy.dtype
        specifices type of integer for this platform that can
        contain the desired number of bits

    Raises:
    ------
    ValueError:
        if bits equals 1 or is too large for
        available data types on y_spacingstem

    ValueError: if sign is not equal to either 1 or 0
    """
    uint_types = (np.ubyte, np.ushort, np.uint, np.ulonglong)
    int_types = (np.byte, np.short, np.int_, np.longlong)
    if bits == 1:
        raise ValueError("Numpy cannot handle bits allocated"
                         "for pixel sample being equal to 1")
    if bits % 8 != 0:
        raise ValueError("DICOM standards requires pixel"
                         "bits to be a multiple of 8")

    if sign == 0:
        val = (2**bits) - 1
        type_list = uint_types
    elif sign == 1:
        val = (2**(bits - 1)) - 1
        type_list = int_types
    else:
        raise ValueError("Pixel Representation metadata"
                         "must be equal to 0 or 1")

    for int_dtype in type_list:
        info = np.iinfo(int_dtype)
        if val <= info.max:
            return np.dtype(int_dtype)

    raise ValueError("Bits allocated for pixel sample"
                     "too large for system")


def in_dtype_range(
    dtype: np.dtype,
    max_val: Union[int, float],
    min_val: Union[int, float],
):
    """Description:
    Returns True is max and min values fit
    within the dtype and False otherwise
    """
    vals = sorted([max_val, min_val])
    dtype_info = np.iinfo(dtype)
    if (vals[1] <= dtype_info.max) and (vals[0] >= dtype_info.min):
        return True
    return False

def as_binary_mask(
    img: np.ndarray,
    reverse: bool = False,
    high: int = 1,
    low: int = 0
):
    """Description:
    Function to create and return a binary mask from numpy array
    containing only 2 unique values

    Parameters:
    ----------
    img: NDArray
        image to create binary mask from

    reverse: bool
        If False (default), the most common image value is assumed to be the
        background and the other value as the foreground. Set to True to
        reverse these assumptions and set the most common image value as the
        foreground and the other value as the background.

    high: int
        Label to use for the foreground.

    low: int | nan
        Label to use for the background.

    Returns:
    ----------
    NDArray[int8 | float]
        Image array with values converted to the binary labels high and low.
    """
    [unique_vals, unique_counts] = np.unique(img, return_counts=True)
    if len(unique_vals) != 2:
        raise ValueError("Mask DICOM not restricted to two unique HU values")

    if reverse is False:
        mask_val = unique_vals[np.argmin(unique_counts)]
    else:
        mask_val = unique_vals[np.argmax(unique_counts)]

    if np.isnan(low):
        return np.array(np.where(img == mask_val, high, low))

    return np.array(np.where(img == mask_val, high, low), dtype=np.int8)

def get_surface(img: np.ndarray):
    """Returns surface of the binary image after eroding it"""
    interior = binary_erosion(img)
    surface = img - interior
    return surface
