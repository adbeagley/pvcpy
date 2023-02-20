"""Functions to read DICOM stacks"""

__all__ = ["read_dcms", "read_dcm_stack", "read_ct_scan"]

from os import PathLike
from pathlib import Path
from typing import Optional

from pydicom import dcmread

from .utilities import as_binary_mask
from .classes import (
    DcmList,
    CtScan,
)


def read_dcms(stack_dir: PathLike, force: bool = False):
    """Read slices inside a directory into a DcmList

    Paramters:
    ----------
    stack_dir: PathLike
        PathLike pointing to the directory/folder containing the DICOM image
        slices to read.

    force: bool
        If False (default), raises an InvalidDicomError if the file is missing
        the File Meta Information header. Set to True to force reading even if
        no File Meta Information header is found.

    Returns:
    ---------
    DcmList
        DcmList object containing the DICOM images from the stack.
    """
    stack_dir = Path(stack_dir)
    return DcmList([dcmread(fpath, force=force)
                    for fpath in stack_dir.glob("*.dcm")])


def read_dcm_stack(
    stack_dir: PathLike,
    modality: Optional[str] = None,
    strict: bool = True,
    force: bool = False
):
    """Read slices inside a directory into a DcmStack, if modality is "CT"
    then return them as a CtScan object.

    Parameters:
    -----------
    stack_dir: PathLike
        PathLike pointing to the directory/folder containing the DICOM image
        slices to read.

    modality: str
        Imaging modality for the DICOM stack. Currently only CT images are
        supported.

    strict: bool
        If True (default), raises an Error if the modality is CT and required
        CT metadata for reconstruction as a CtScan object is missing. Set to
        False to force conversion to CtScan object even if required metadata
        is missing (note: may cause CtScan methods to fail or return erroneous
        results).

    force: bool
        If False (default), raises an InvalidDicomError if the file is missing
        the File Meta Information header. Set to True to force reading even if
        no File Meta Information header is found.
    """
    dcms = read_dcms(stack_dir, force)
    if modality == "CT":
        return CtScan(dcms, strict=strict)
    return dcms.as_stack()


def read_ct_scan(
    stack_dir: PathLike,
    strict: bool = True,
    binary_data: bool = False,
    high: int = 1,
    low: int = 0,
    force: bool = False
):
    """Read slices inside a directory into a DcmStack, assumes modality is "CT"
    and returns a CtScan object.

    Parameters:
    -----------
    stack_dir: PathLike
        PathLike pointing to the directory/folder containing the DICOM image
        slices to read.

    strict: bool
        If True (default), raises an Error if the modality is CT and required
        CT metadata for reconstruction as a CtScan object is missing. Set to
        False to force conversion to CtScan object even if required metadata
        is missing (note: may cause CtScan methods to fail or return erroneous
        results).

    binary_data: bool
        If False (default), read and rescale DICOM pixel data as per the DICOM
        standard. Set to True to cast pixel data to a binary image labelled
        with the foreground as "high" and background as "low" (assuming that
        the most common value in the image is the background). Raises an error
        if the pixel data contains more than 2 unique values.

    high: int
        Label to set foreground to if binary_data is True.

    low: int | nan
        Label to set background to if binary_data is True.

    force: bool
        If False (default), raises an InvalidDicomError if the file is missing
        the File Meta Information header. Set to True to force reading even if
        no File Meta Information header is found.
    """
    dcms = read_dcms(stack_dir, force)
    dcms = CtScan(dcms, strict=strict)
    if binary_data is True:
        dcms.pixel_array = as_binary_mask(dcms.pixel_array, high=high, low=low)
    return dcms
