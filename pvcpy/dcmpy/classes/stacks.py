"""
DcmStack Subclasses that are modality specific

Notes:
------
Should probably use UID to check if things are a CT scan
or not instead of modality?
"""
__all__ = ["CtScan", "ScScan"]

import logging
from os import PathLike
from pathlib import Path
from typing import Set, Tuple, Union

import numpy as np
from pydicom.pixel_data_handlers import gdcm_handler

from .base import (
    DcmList,
    DcmStack
)
from ..utilities import (
    keywords,
    uid,
    choose_np_dtype_int,
    in_dtype_range
)
from .structs import (
    StackDims,
    IjkSpacing,
    PixelSpacing
)


class ImgScan(DcmStack):
    """Super Class for all Scan classes"""
    __modalities__: Set[str] = set(("",))    # set valid modalities for class

    _required_tags: Tuple[str] = ("",)  # set required tags for class

    _constant_tags: Tuple[str] = ("",)  # set required constant tags for class

    # keywords corresponding to pixel spacing data
    _pixel_spacing_keywords: Tuple[str] = (
        keywords.PixelSpacing,                      # (0028,0030)
        keywords.ImagerPixelSpacing,                # (0018,1164)
        keywords.NominalScannedPixelSpacing,        # (0018,2010)
        keywords.ImagePlanePixelSpacing,            # (3002,0011)
        keywords.CompensatorPixelSpacing,           # (300A,00E9)
        keywords.DetectorElementSpacing,            # (0018,7022)
        keywords.PresentationPixelSpacing,          # (0070,0101)
        keywords.PrinterPixelSpacing,               # (2010,0376)
        keywords.ObjectPixelSpacingInCenterOfBeam   # (0018,9404)
    )
    # can be any of the following but different
    # modalities use various subsets of these

    # pylint: disable = unused-argument
    def __new__(cls, dcms: Union[DcmList, DcmStack], **kwargs):
        if not gdcm_handler.is_available():
            raise ModuleNotFoundError("ImgScan objects require gdcm for "
                                      "handling dicom pixel data")
        return super().__new__(cls, dcms)

    def __init__(
        self,
        dcms: Union[DcmList, DcmStack],
        strict: bool = True
    ) -> None:
        super().__init__()
        self._strict = strict
        if len(self) > 0:
            self.validate()
        else:
            raise RuntimeError("Empty DICOM stack!")

    @classmethod
    def modality(cls):
        """Image modality for this class"""
        return cls.__modalities__

    @property
    def strict(self):
        """Bool indicating if errors will be raised when
        metadata does not comply with DICOM standard"""
        return self._strict

    @property
    def dimensions(self) -> StackDims:
        """Description:
        Read only property

        Returns:
        --------
        dimensions: StackDims
            named tuple with number of Rows, Columns,
            and Slices in the sequence

        Raises:
        -------
        AttributeError:
            if Rows or Columns metadata is not available or constant
        """
        dcm_extent = StackDims(self.get_constant_metadata(keywords.Rows),
                               self.get_constant_metadata(keywords.Columns),
                               len(self))
        if None in dcm_extent:
            raise RuntimeError("Invalid number of Rows or Columns")
        return dcm_extent

    @property
    def pixel_spacing(self):
        """
        In-Plane spacing of pixels

        Returns:
        --------
        tuple: (row_spacing, column_spacing)

        Notes:
        ------
        Uses PixelSpacing which may not always be present,
        should pick some other values to fall back on
        """
        for key in self._pixel_spacing_keywords:
            pix_spacing = self.get_constant_metadata(key)
            if pix_spacing is not None:
                return PixelSpacing(*pix_spacing)
        raise RuntimeError(
            "Data for any of the following tags:"
            f"{self._pixel_spacing_keywords}"
            "is not present for all slices."
        )

    @property
    def slice_spacing(self):
        """Description:
        Read only property

        Returns:
        --------
        slice_spacing: float
            distance between origins slices in the stack

        Notes:
        ------
        If distance between slices is not uniform
        then the mean of the distances is returned
        """
        if len(self) == 1:
            return 0.0
        slice_spacing = set(self.slice_distances)
        if len(slice_spacing) == 1:
            slice_spacing = slice_spacing.pop()
        else:
            slice_spacing = np.nanmean(self.slice_distances)
        return slice_spacing

    @property
    def spacing(self) -> IjkSpacing:
        """Description:
        Read only property

        Returns:
        --------
        dcm_spacing: IjkSpacing
            spacing between voxels in the image coordinate system
        """
        row_spacing, col_spacing = self.pixel_spacing
        slice_spacing = self.slice_spacing
        dcm_spacing = IjkSpacing(col_spacing, row_spacing, slice_spacing)
        return dcm_spacing

    def save(self, path: PathLike):
        """Description:
        Saves the dicom sequence at the specificed path location
        (creates it if it does not already exist)
        and updates path property to point to saved sequence
        """
        self.validate()
        path = Path(path)
        if not path.exists():
            if path.suffix != "":
                raise ValueError("Path to save DICOM stack "
                                 "point be a directory")
            path.mkdir()
        if not path.is_dir():
            raise ValueError("Path to save DICOM stack "
                             "must point to a directory")

        digits = self._count_digits(len(self)-1)+1
        for k in range(self.n_slices):
            new_file = path.joinpath(
                self._create_file_name(path.stem, digits, k))
            self[k].save_as(new_file, False)

    def slice_pixel_dtype(self, k: int):
        """Description:
        Returns the numpy data type of pixels in slice k of the
        sequence based on PixelRepresentation and BitsAllocated metadata

        Parameters:
        ----------
        k: int
            slice id number in the stack

        Notes:
        ------
        Possibly move this over to ImgScan or CtScan classes
        since it may be modality specific
        """
        sign = self[k].PixelRepresentation
        bits = self[k].BitsAllocated
        return choose_np_dtype_int(sign, bits)

    def validate(self):
        """
        Confirms the modality of the slices is correct and performs
        validation of required and constant metadata for image type

        Notes:
        ------
        Throws warnings if strict is False but Raises Errors if strict is True
        """
        # check modality
        self._is_allowed_value(keywords.Modality, self.__modalities__)

        # check required tags are present
        for tag in self._required_tags:
            if not self.has_metadata(tag):
                err_msg = f"{tag} data is not present for all slices."
                if self.strict:
                    raise RuntimeError(err_msg)
                logging.warning(err_msg)

        # check required tags with constant values are present and constant
        for tag in self._constant_tags:
            if not self.has_constant_metadata(tag):
                err_msg = f"{tag} data is not constant."
                self._throw_error(err_msg)

    def _is_allowed_value(self, tag: str, allowed_vals: Set[str]):
        """Checks if the value for the given tag is constant
        and one of the allowed values provided"""
        value = self.get_constant_metadata(tag)
        if value is None:
            err_msg = f"{tag} data is missing or not constant."
            self._throw_error(err_msg)
        elif value not in allowed_vals:
            err_msg = (f"{value} is an inavlid value for {tag} data "
                       f"in class: <{type(self).__name__}>.")
            self._throw_error(err_msg)

    def _create_file_name(self, file_prefix: str, digits: int, num: int):
        """Description:
        Combines strings to create a dcm file name so that the files will
        have a prefix followed by an underscore and a number in the form
        0XXX to ensure proper alphanumeric sorting by operating systems
        """
        n_digits = self._count_digits(num)
        name = file_prefix + "_"
        for _ in range(digits - n_digits):
            name = name + "0"
        return name+str(num) + ".dcm"

    def _count_digits(self, num: int):
        """Description:
        Counts and returns the number of digits in an integer
        """
        digits = 0
        if num == 0:
            return 1
        while num > 0:
            num = num // 10
            digits += 1
        return digits

    def _throw_error(self, err_msg: str):
        """If strict is True raises a RuntimeError
        with the given error message, else throws a warning."""
        if self.strict:
            raise RuntimeError(err_msg)
        logging.warning(err_msg)

    def _properties_summary(self, title: str):
        """Creates header with class name and properties summarized"""
        pad = 30
        strings = [title,
                   "-"*2*len(title),
                   f"{'Dimensions: ':{pad}} {self.dimensions}",
                   f"{'Orthogonal Axes:':{pad}} {self.orthognal_axes}",
                   f"{'Direction Cosines:':{pad}} {self.direction_cosines}",
                   f"{'Pixel Spacing:':{pad}} {self.pixel_spacing}",
                   f"{'Uniform Slice Spacing:':{pad}} {self.uniform_spacing}",
                   f"{'Slice Spacing:':{pad}} {self.slice_spacing}",
                   "-"*2*len(title)]
        return strings


class CtScan(ImgScan):
    """Stack of DICOM slices for a CT scan"""
    __modalities__: Set[str] = set(("CT", "OT"))

    _pixel_spacing_keywords: Tuple[str] = (keywords.PixelSpacing,)

    _required_tags: Tuple[str] = (keywords.SOPInstanceUID,
                                  keywords.ImagePositionPatient,
                                  keywords.RescaleIntercept,
                                  keywords.RescaleSlope,
                                  keywords.PixelRepresentation,
                                  keywords.BitsAllocated,)

    _constant_tags: Tuple[str] = (keywords.StudyInstanceUID,
                                  keywords.SeriesInstanceUID,
                                  keywords.ImageOrientationPatient,
                                  keywords.Rows,
                                  keywords.Columns,
                                  keywords.SamplesPerPixel)

    _allowable_rescale_types: Set[str] = set(("HU",
                                              "Houndsfield Unit",
                                              "Hounsfield Unit",
                                              ""))

    def validate(self):
        super().validate()
        if self.has_metadata(keywords.RescaleType):
            self._is_allowed_value(keywords.RescaleType,
                                   self._allowable_rescale_types)

    def is_aligned(self, dcm_stack: "CtScan"):
        """Returns True if the provided dcm_stack is spatially
        aligned with this instance and False otherwise.
        """
        tags = (keywords.PixelSpacing,
                keywords.ImageOrientationPatient,
                keywords.ImagePositionPatient)

        if self.dimensions != dcm_stack.dimensions:
            return False

        for tag in tags:
            self_vals = self.get_stack_metadata(tag)
            other_vals = self.get_stack_metadata(tag)
            if self_vals != other_vals:
                return False
        return True

    @property
    def pixel_array(self):
        """Description:
        Returns uncompressed and rescaled pixel data for the DICOM sequence

        Notes:
        ------
        Computed each time the property is called
        """
        dicom_img_3d = np.empty(self.dimensions, dtype=np.float64)
        for k in range(self.n_slices):
            dicom_img_3d[:, :, k] = self.get_slice_pixel_array(k)
        return dicom_img_3d

    @pixel_array.setter
    def pixel_array(self, data: np.ndarray):
        """Description:
        Updates pixel data in the DICOM sequence to equal img_data_3d
        and updates required UID metadata.

        Parameters:
        ------------
        img_data_3d: ndarry
            numpy array with same shape as DICOM sequence extent
            that contains the pixel values

        Notes:
        -----
        Compresses to byte string using C ordering so input should use
        matrix indexing
        (i traverses image rows, j traverses image columns)
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("pixel data must be a numpy array")

        if data.shape != self.dimensions:
            raise ValueError("pixel data dimensions != stack dimensions")

        study_uid = uid.generate_uid()
        series_uid = uid.generate_uid()

        for k in range(self.n_slices):
            self[k].StudyInstanceUID = study_uid
            self[k].SeriesInstanceUID = series_uid
            self.set_slice_pixel_array(k, data[:, :, k])

    def get_slice_pixel_array(self, k: int) -> np.ndarray:
        """Description:
        Gets an uncompressed and rescaled array of pixel
        values from slice k in the sequence

        Parameters:
        -----------
        k: int
            number of the slice in the DICOM sequence to
            retrieve pixel data from

        Returns:
        --------
        array_image: ndarry
            array containing the values at each pixel

        Notes:
        ------
        Compresses to byte string using C ordering so input
        should use matrix indexing
        (i traverses image rows, j traverses image columns)
        """
        arr = gdcm_handler.get_pixeldata(self[k]).astype(np.float64)
        image = arr.reshape((self[k].Rows, self[k].Columns), order="C")
        intercept = self.get_slice_tag_value(k, keywords.RescaleIntercept)
        slope = self.get_slice_tag_value(k, keywords.RescaleSlope)
        intercept = 0.0 if intercept is None else intercept
        slope = 1.0 if slope is None else slope
        array_image = image*slope + intercept
        return array_image

    def set_slice_pixel_array(self, k: int, arr: np.ndarray):
        """Description:
        Updates the pixel array data of slice k in the sequence
        using ImplicitVRLittleEndian transfer syntax and SOPInstaceUIDs.

        Parameters:
        -----------
        k: int
            slice number to act on

        arr: ndarray
            input pixel data with shape (rows, columns)

        Notes:
        ------
        Compresses to byte string using C ordering so input
        should use matrix indexing
        (i traverses image rows, j traverses image columns)

        Uses pixel representation and bits allocated metadata for slice to
        determine pixel data type. Also checks that the new pixel data will
        fit in the pixel data type without data loss after rescaling. If it
        will not then it calculates a new rescale equation so that the maximum
        and minimum values will fit within the datatype. First tries fitting
        by changing to the Rescale Intercept and if that fails then also
        modifies the Rescale Slope.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError("arr must be a numpy array")
        transfer_syntax = uid.ImplicitVRLittleEndian
        sop_inst_uid = uid.generate_uid()
        self[k].is_little_endian = True
        self[k].is_implicit_VR = True
        self[k].file_meta.TransferSyntaxUID = transfer_syntax
        self[k].SOPInstanceUID = sop_inst_uid
        self[k].file_meta.MediaStorageSOPInstanceUID = sop_inst_uid
        self[k].PixelData = self._rescale_image(k, arr)

    def _rescale_image(self, k: int, arr: np.ndarray):
        """Description:
        Calculates the required Rescale Slope and Intercept to
        store the image (arr) in the slice (k) without data loss
        and returns the rescaled image as a byte string of the
        correct data type for storage
        """
        arr_max = np.nanmax(arr)
        arr_min = np.nanmin(arr)
        np_dtype = self.slice_pixel_dtype(k)

        intercept = self.get_slice_tag_value(k, "RescaleIntercept")
        if intercept is None:
            intercept = 0
            self.set_slice_tag_value(k, "RescaleIntercept", "DS", intercept)

        slope = self.get_slice_tag_value(k, "RescaleSlope")
        if slope is None:
            slope = 1
            self.set_slice_tag_value(k, "RescaleSlope", "DS", slope)

        re_max = (arr_max - intercept) / slope
        re_min = (arr_min - intercept) / slope

        if not in_dtype_range(np_dtype, re_max, re_min):
            slope, intercept = self._calc_rescale_vals(np_dtype,
                                                       arr_max,
                                                       arr_min,
                                                       slope)
            self.set_slice_tag_value(k, "RescaleSlope", "DS", slope)
            self.set_slice_tag_value(k, "RescaleIntercept", "DS", intercept)

        arr = np.array((arr - intercept) / slope, dtype=np_dtype)
        return arr.tobytes()

    def _calc_rescale_vals(
        self,
        dtype: np.dtype,
        arr_max: Union[int, float],
        arr_min: Union[int, float],
        slope: Union[int, float]
    ):
        """Description:
        Determines Rescale Slope and Intercept Values that prevent
        data loss due to precision errors

        Notes:
        ------
        First tries to fit maximum and minimum image values into the data
        type range by adjusting the rescale intercept so that the minimum
        image value is equal to the smallest value storable in the data type.
        If that fails then it calculates a linear rescaling equation so that
        the mininimum equal value is mapped to the smallest value storable in
        the data type and the maximum image value is mapped to the largest
        value storable in the data type.
        """
        dtype_info = np.iinfo(dtype)
        max_min = np.array([arr_max, arr_min]).reshape((2, 1))
        intercept = max_min[1] - dtype_info.min*slope
        re_coeffs = np.array([slope, intercept[0]]).reshape((2, 1))
        re_vals = (max_min - re_coeffs[1]) / re_coeffs[0]
        if not in_dtype_range(dtype, re_vals[0], re_vals[1]):
            bnds_mtrx = np.array([[dtype_info.max, 1], [dtype_info.min, 1]])
            re_coeffs = np.linalg.solve(bnds_mtrx, max_min)
        return re_coeffs[0], re_coeffs[1]
