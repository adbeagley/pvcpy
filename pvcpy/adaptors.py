"""Adaptor functions to convert between numpy and VTK arrays/matrices."""
from typing import Union, Optional
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from vtkmodules.all import (  # pylint: disable = no-name-in-module
    vtkMatrix3x3,
    vtkMatrix4x4,
    vtkIdList,
    vtkTransform,
    vtkDataArray,
    vtkBitArray,
    vtkStringArray,
    vtkCharArray,
    vtkPoints,
)
from vtkmodules.util.numpy_support import (
    vtk_to_numpy,
    numpy_to_vtk,
    numpy_to_vtkIdTypeArray,
    VTK_ID_TYPE_SIZE
)

__all__ = [
    "convert_3d_img_to_vtkarray",
    "convert_points",
    "convert_array",
    "convert_matrix",
    "convert_transform_matrix",
    "convert_id_list",
    "numpy_to_idarr",
]


def _get_vtk_id_type():
    """Return the numpy datatype responding to ``vtk.vtkIdTypeArray``."""
    if VTK_ID_TYPE_SIZE == 4:
        return np.int32
    if VTK_ID_TYPE_SIZE == 8:
        return np.int64
    return np.int32

IdType = _get_vtk_id_type()


def convert_3d_img_to_vtkarray(
    img: NDArray,
    name: Optional[str] = None,
    deep: bool = False,
):
    """Convert 3d numpy image to a flattened vtkDataArray.

    Parameters:
    ------------
    img: NDArray
        A 3-d ndarray with pixel data.

    Returns:
    --------
    vtkarray: vtkDataArray
        vtkDataArray of the appropriate type containing the
        flattened pixel data.
    """
    img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError(
            f"Expected a 3-d image, not a {img.ndim}-d image as input."
        )
    return convert_array(img.flatten("F"), name=name, deep=deep)


def convert_points(
    points: Union[NDArray, vtkPoints],
    deep: bool = True,
    force_float: bool = False
):
    """Convert between ArrayLikes and vtkPoints.

    Parameters
    ----------
    points : NDArray | vtkPoints
        Points to convert. ArrayLike Should be 1 or 2 dimensional.
        If 1 dimensional, points should be listed x1, y1, z1, ... xn, yn, zn.
        Accepts a single point or several points.

    deep : bool, default: True
        Perform a deep copy of the array or points.

    force_float : bool, default: False
        Casts the datatype to ``float64`` if points datatype is
        non-float.  Set this to ``False`` to allow non-float types,
        though this may lead to truncation of intermediate floats
        when transforming datasets. Only applicable if
        ``points`` is a :class:`numpy.ndarray`.

    Returns
    -------
    NDArray | vtkPoints
        The converted points object.
    """
    if isinstance(points, vtkPoints):
        points = points.GetData()
        points = vtk_to_numpy(points)
        if deep:
            return np.array(points)
        return points
    if isinstance(points, np.ndarray):
        points = np.asanyarray(points)  # cast to array if needed

        # verify is numeric
        if not np.issubdtype(points.dtype, np.number):
            raise TypeError("Points must be a numeric type")

        if force_float:
            if not np.issubdtype(points.dtype, np.floating):
                warn(
                    "Points is not a float type. This can cause issues when "
                    "transforming or applying filters. Casting to "
                    "``np.float32``. Disable this by passing "
                    "``force_float=False``."
                )
                points = points.astype(np.float64)

        # check dimensionality
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        elif points.ndim > 2:
            raise ValueError(
                "Dimension of ``points`` should be 1 or 2, " f"not {points.ndim}"
            )

        # verify shape
        if points.shape[1] != 3:
            raise ValueError(
                "Points array must contain three values per "
                f"point. Shape is {points.shape} and should be "
                "(X, 3)"
            )

        # points must be contiguous
        points = np.require(points, requirements=["C"])
        vtkpts = vtkPoints()
        vtk_arr = numpy_to_vtk(points, deep=deep)
        vtkpts.SetData(vtk_arr)
        return vtkpts

    raise TypeError(f"Expected NDArray or vtkPoints, not {type(points)} " "as input.")


def convert_transform_matrix(
    matrix: Union[NDArray, vtkTransform]
) -> Union[NDArray, vtkTransform]:
    """Converts between a numpy.ndarray and a vtkTransform.

    Parameters:
    -----------
    matrix: NDArray | vtkTransform
        The matrix or transform to convert. The NDArray must have shape (3, 3)
        or (4, 4).

    Returns:
    --------
    NDArray | vtkTransform
        Converted transform matrix.

        If input was an NDArray with shape (3, 3) then it returns a
        vtkTransform with the (3, 3) array defining the 9 components
        of a 4x4 affine transformation matrix with no translation.

        If the input was an NDArray with shape (4, 4) then it returns a
        vtkTransform with the transformation matrix defined by the NDArray.

        If the input was a vtkTransform then it returns a NDArray with shape
        (4, 4) containing the transformation matrix.
    """
    if isinstance(matrix, np.ndarray):
        if matrix.shape == (3, 3):
            matrix = np.pad(matrix, pad_width=((0, 1), (0, 1)))
            matrix[-1, -1] = 1
        matrix = convert_matrix(matrix)
        transform = vtkTransform()
        transform.SetMatrix(matrix)
        return transform
    if isinstance(matrix, vtkTransform):
        matrix = matrix.GetMatrix()
        return convert_matrix(matrix)

    raise TypeError(
        f"Expected NDArray or vtkTransform, not {type(matrix)}" " as input."
    )


def convert_matrix(matrix: Union[NDArray, vtkMatrix3x3, vtkMatrix4x4]):
    """Convert between numpy arrays of shape (3, 3) or (4, 4) and
    vtkMatrix3x3 or vtkMatrix4x4.

    Parameters:
    -----------
    matrix: NDArray | vtkMatrix3x3 | vtkMatrix4x4
        Matrix to convert. numpy array with shape (3, 3) or (4, 4),
        vtkMatrix3x3, or vtkMatrix4x4

    Returns:
    --------
    NDArray | vtkMatrix3x3 | vtkMatrix4x4
        The converted matrix. If input was numpy array with shape (3, 3) then
        returns a vtkMatrix3x3 and if shape was (4, 4) then returns a
        vtkMatrix4x4. If input was a vtkMatrix4x4 then returns a numpy array
        with shape (4, 4) and if input was a vtkMatrix3x3 then returns a numpy
        array with shape (3, 3).
    """
    if isinstance(matrix, (vtkMatrix3x3, vtkMatrix4x4)):
        return _array_from_vtkmatrix(matrix)
    if isinstance(matrix, np.ndarray):
        return _vtkmatrix_from_array(matrix)

    raise TypeError(
        "Expected NDArray, vtkMatrix3x3, or vtkMatrix4x4 as "
        f"input, got {type(matrix)} instead."
    )


def _array_from_vtkmatrix(matrix: Union[vtkMatrix3x3, vtkMatrix4x4]):
    """Convert a vtk matrix to an array.

    Parameters
    ----------
    matrix : vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4
        The vtk matrix to be converted to a ``numpy.ndarray``.
        Returned ndarray has shape (3, 3) or (4, 4) as appropriate.

    Returns
    -------
    numpy.ndarray
        Numpy array containing the data from ``matrix``.
    """
    if isinstance(matrix, vtkMatrix3x3):
        shape = (3, 3)
    elif isinstance(matrix, vtkMatrix4x4):
        shape = (4, 4)
    else:
        raise TypeError(
            "Expected vtkMatrix3x3 or vtkMatrix4x4 as input,"
            f" got {type(matrix).__name__} instead."
        )
    array = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            array[i, j] = matrix.GetElement(i, j)
    return array


def _vtkmatrix_from_array(array: np.ndarray):
    """Convert a ``numpy.ndarray`` or array-like to a vtk matrix.

    Parameters
    ----------
    array : numpy.ndarray or array-like
        The array or array-like to be converted to a vtk matrix.
        Shape (3, 3) gets converted to a ``vtk.vtkMatrix3x3``, shape (4, 4)
        gets converted to a ``vtk.vtkMatrix4x4``. No other shapes are valid.

    Returns
    -------
    vtkMatrix3x3 or vtkMatrix4x4
        VTK matrix.
    """
    array = np.asarray(array)
    if array.shape == (3, 3):
        matrix = vtkMatrix3x3()
    elif array.shape == (4, 4):
        matrix = vtkMatrix4x4()
    else:
        raise ValueError(f"Invalid shape {array.shape}, must be (3, 3) " " or (4, 4).")
    m, n = array.shape  # pylint: disable=invalid-name
    for i in range(m):
        for j in range(n):
            matrix.SetElement(i, j, array[i, j])
    return matrix


def numpy_to_idarr(arr: NDArray, deep: bool = False, return_arr: bool = False):
    """Safely convert a numpy array to a vtkIdTypeArray."""
    arr = np.asarray(arr)

    # np.asarray will eat anything, so we have to weed out bogus inputs
    if not issubclass(arr.dtype.type, (np.bool_, np.integer)):
        raise TypeError(
            "Indices must be either a mask or an integer array-like."
        )

    if arr.dtype == np.bool_:
        arr = arr.nonzero()[0].astype(IdType)
    elif arr.dtype != IdType:
        arr = arr.astype(IdType)
    elif not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr, dtype=IdType)

    # must ravel or segfault when saving MultiBlock
    vtk_idarr = numpy_to_vtkIdTypeArray(arr.ravel(), deep=deep)
    if return_arr:
        return vtk_idarr, arr
    return vtk_idarr


def convert_id_list(id_list: Union[vtkIdList, NDArray[np.integer]]):
    """Convert between a vtkIdList and a NumPy array.

    Parameters
    ----------
    id_list: vtkIdList or NDArray[integer]
        A vtkIdList or numpy array to convert. The numpy array must be 1D and
        of integer data type.

    Returns
    -------
    vtkIdList or numpy.ndarray
        The converted id list. If input is vtkIdList then returns numpy.ndarray
        or if input is numpy.ndarray then returns vtkIdList.
    """
    if isinstance(id_list, vtkIdList):
        return np.array(
            [id_list.GetId(i) for i in range(id_list.GetNumberOfIds())],
            dtype=np.integer,
        )
    if isinstance(id_list, np.ndarray):
        if not np.issubdtype(id_list.dtype, np.integer):
            raise TypeError("Expected an array of integer ids.")
        if id_list.ndim != 1:
            raise ValueError("Expected a 1D array of integers ids.")
        result = vtkIdList()
        for val in id_list:
            result.InsertNextId(val)
        return result
    raise TypeError(f"Expected a vtkIdlist or NDArray, not {type(id_list)}")


def convert_array(
    arr: Union[np.ArrayLike, vtkDataArray],
    name: str = None,
    deep: bool = False,
    array_type: Optional[int] = None,
):
    """Convert a NumPy array to a vtkDataArray or vice versa.

    Parameters
    ----------
    arr : np.ndarray or vtkDataArray
        A numpy array or vtkDataArry to convert.

    name : str, optional
        The name of the data array for VTK.

    deep : bool, default: False
        If True, then deep copy values into the new array.

    array_type : int, optional
        VTK array type ID as specified in specified in ``vtkType.h``.

    Returns
    -------
    vtkDataArray, numpy.ndarray, or DataFrame
        The converted array.  If input is a :class:`numpy.ndarray` then
        returns ``vtkDataArray`` or is input is ``vtkDataArray`` then
        returns NumPy ``ndarray``.
    """
    if arr is None:
        return
    if isinstance(arr, (list, tuple)):
        arr = np.array(arr)
    if isinstance(arr, np.ndarray):
        if arr.dtype == np.dtype("O"):
            arr = arr.astype("|S")
        arr = np.ascontiguousarray(arr)
        if arr.dtype.type in (np.str_, np.bytes_):
            # This handles strings
            vtk_data = convert_string_array(arr)
        else:
            # This will handle numerical data
            arr = np.ascontiguousarray(arr)
            vtk_data = numpy_to_vtk(num_array=arr, deep=deep, array_type=array_type)
        if isinstance(name, str):
            vtk_data.SetName(name)
        return vtk_data
    # Otherwise input must be a vtkDataArray
    if not isinstance(arr, (vtkDataArray, vtkBitArray, vtkStringArray)):
        raise TypeError(f"Invalid input array type ({type(arr)}).")

    if isinstance(arr, vtkBitArray):
        arr = vtk_bit_array_to_char(arr)  # Handle booleans
    if isinstance(arr, vtkStringArray):
        arr = convert_string_array(arr)  # Handle string arrays
    else:
        arr = vtk_to_numpy(arr)  # Convert from vtkDataArry to NumPy
    if deep:
        return np.array(arr)  # create deep copy if requested
    return arr


def vtk_bit_array_to_char(vtkarr_bint: vtkBitArray):
    """Cast vtk bit array to a char array.

    Parameters
    ----------
    vtkarr_bint : vtk.vtkBitArray
        VTK binary array.

    Returns
    -------
    vtk.vtkCharArray
        VTK char array.

    Notes
    -----
    This performs a copy.
    """
    vtkarr = vtkCharArray()
    vtkarr.DeepCopy(vtkarr_bint)
    return vtkarr


def convert_string_array(
    arr: Union[vtkStringArray, NDArray[np.str_]], name: Optional[str] = None
):
    """Convert a numpy array of strings to a vtkStringArray or vice versa.

    Parameters
    ----------
    arr : numpy.ndarray
        Numpy string array to convert.

    name : str, optional
        Name to set the vtkStringArray to.

    Returns
    -------
    vtkStringArray
        VTK string array.

    Notes
    -----
    Note that this is terribly inefficient. If you have ideas on how
    to make this faster, please consider opening a pull request.
    """
    if isinstance(arr, np.ndarray):
        # VTK default fonts only support ASCII.
        # See https://gitlab.kitware.com/vtk/vtk/-/issues/16904
        # Check formatting to avoid segfault
        if np.issubdtype(arr.dtype, np.str_) and not "".join(arr).isascii():
            raise ValueError(
                "String array contains non-ASCII characters that are not supported by VTK."
            )
        vtkarr = vtkStringArray()
        ########### OPTIMIZE ###########
        for val in arr:
            vtkarr.InsertNextValue(val)
        ################################
        if isinstance(name, str):
            vtkarr.SetName(name)
        return vtkarr
    # Otherwise it is a vtk array and needs to be converted back to numpy
    ############### OPTIMIZE ###############
    nvalues = arr.GetNumberOfValues()
    return np.array([arr.GetValue(i) for i in range(nvalues)], dtype="|U")
    ########################################
