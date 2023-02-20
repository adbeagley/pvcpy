"""
# Written by: Aren Beagley
# contains abstract superclass containing methods for
working with dcm metadata that DcmObj objects inheiret from
"""
__all__ = ["DcmList", "DcmStack"]

from abc import abstractmethod
from typing import (
    Union,
    Iterable
)
import numpy as np
from pydicom import Dataset
from scipy.stats import linregress

from ..utilities import keywords
from .decorators import abstract_class
from .structs import DirectionCosines


@abstract_class
class AbstractDcmContainer():
    """Methods for interacting with slice metadata"""

    pos_tag = keywords.ImagePositionPatient
    orn_tag = keywords.ImageOrientationPatient

    def has_metadata(self, key: str):
        """Returns true if metadata is present in all slices"""
        metadata = [bool(dcm_slice.dir(key)) for dcm_slice in self]
        if all(metadata):
            return True
        return False

    def has_constant_metadata(self, key: str):
        """Returns true if metadata is present and constant in all slices"""
        metadata = [dcm_slice.data_element(key).value
                    for dcm_slice in self
                    if dcm_slice.dir(key)]

        if len(metadata) != len(self):
            return False

        if all([x == metadata[0] for x in metadata]):
            return True
        return False

    def get_constant_metadata(self, key: str):
        """
        Returns value for metadata tag if it is constant,
        else returns None
        """
        if self.has_constant_metadata(key):
            return self[0].data_element(key).value
        return None

    @abstractmethod
    def __getitem__(self, index) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> "AbstractDcmContainer":
        raise NotImplementedError

    @abstractmethod
    def __next__(self) -> Dataset:
        raise NotImplementedError


class DcmList(list, AbstractDcmContainer):
    """
    Mutable sequence that can only contain pydicom.DataSet objects.
    """

    def __init__(self, dcms: Iterable = None):
        """
        Mutable sequence that can only contain pydicom.DataSet objects.

        If no argument is given, the constructor creates a new empty DcmList.
        The argument must be an iterable of pydicom.Dataset
        objects if specified.
        """

        if dcms is None:
            super().__init__()
        elif isinstance(dcms, Iterable):
            super().__init__(dcms)
        else:
            raise TypeError("Invalid type for dcms parameter.")
        self._iter_num = None

    def get(self, index: int) -> Union[Dataset, None]:
        """Description:
        Returns the pydicom.dataset for the given index or returns
        None if there is no dataset at that index
        """
        try:
            return self[index]
        except KeyError:
            print("Index is not a valid key")
            return None

    def sort(self):
        """Returns a sorted dcm sequence"""

        # check if length is greater than 1
        if len(self) < 2:
            return self  # copied into a sorted thing

        if not self.has_constant_metadata(self.orn_tag):
            raise RuntimeError(
                f"{self.orn_tag} metadata is missing for 1 or more slices"
                "or is not constant for all slices."
            )

        if not self.has_metadata(self.pos_tag):
            raise RuntimeError(
                f"{self.pos_tag} metadata is missing for 1 or more slices."
            )

        orienation = tuple(self[0].data_element(self.orn_tag).value)
        r_vec = np.cross(orienation[3:], orienation[:3])

        # sort based on positions offset along r_vec
        sort_keys = np.array(
            [np.dot(r_vec,
                    np.asarray(dcm_slice.data_element(self.pos_tag).value))
             for dcm_slice in self]
        )
        return DcmList([self[i] for i in sort_keys.argsort()])

    def as_stack(self):
        """Returns sorted slices as a DcmStack"""
        return DcmStack(self)

    def __setitem__(self, index, value):
        if isinstance(value, Dataset):
            super().__setitem__(index, value)
        else:
            raise TypeError(
                f"Items must be a pydicom Dataset object not {type(value)}."
            )

    # pylint: disable = useless-super-delegation
    def __getitem__(self, index) -> Dataset:
        """For Type hinting"""
        return super().__getitem__(index)

    def __iter__(self):
        self._iter_num = 0
        return self

    def __next__(self):
        if self._iter_num < len(self):
            result = self[self._iter_num]
            self._iter_num += 1
            return result
        else:
            raise StopIteration

    def __str__(self) -> str:
        tag_dict = dict()
        for dcm in self:
            lines = str(dcm).splitlines()
            for elem in lines:
                if ":" in elem:
                    desc, val = elem.split(":", maxsplit=1)
                    if not desc.startswith(" "):
                        if desc not in tag_dict:
                            tag_dict.update({desc: [val]})
                        else:
                            tag_dict[desc].append(val)

        # how is there a key with no values???
        # for the ones not present in all slices
        header = "<" + type(self).__name__ + "> Metadata Summary"
        strings = [header,
                   "-"*2*len(header),
                   f"Number of Slices: {len(self)}",
                   "-"*2*len(header)]
        for key in sorted(tag_dict.keys()):
            values = tag_dict[key]
            if len(values) == len(self):
                if all(x == values[0] for x in values):
                    strings.append(key + ":" + values[0])
                else:
                    strings.append(key + ": VARIABLE")
            else:
                strings.append(key + ": MISSING IN ONE OR MORE SLICES")
        strings.append("-"*2*len(header))
        return "\n".join(strings)


class DcmStack(tuple, AbstractDcmContainer):
    """Immutable stack of sorted DICOM slices"""

    def __new__(cls, dcms: Union[DcmList, "DcmStack"]):
        if isinstance(dcms, DcmList):
            dcms = dcms.sort()
        elif not isinstance(dcms, DcmStack):
            raise TypeError("Invalid type for dcms parameter.")
        return super().__new__(cls, dcms)

    @property
    def n_slices(self):
        """Return the number of slices in the stack"""
        return len(self)

    @property
    def origin(self):
        """Description:
        Read only property

        Returns:
        --------
        origin: tuple
            x, y, z values for origin of first slice in DICOM sequence

        Raises:
        -------
        RuntimeError: if origin metadata is missing for first slice
        """
        elm = self[0].data_element(self.pos_tag)
        if elm is None:
            raise RuntimeError(f"{self.pos_tag} metadata is missing"
                               "for first slice in DICOM stack.")
        return tuple(elm.value)

    @property
    def direction_cosines(self):
        """Description:
        Read only property

        Returns:
        --------
        direction_cosines: DirectionCosines
            NamedTuple with row and col properties
            that contain numpy arrays with the cosine
            vectors of the DICOM stack

        Raises:
        -------
        RuntimeError: if orientation metadata is not present
        and constant for all slices of the DICOM sequence
        """
        orn = self.get_constant_metadata(self.orn_tag)
        if orn is None:
            raise RuntimeError("Direction cosines are not constant")
        return DirectionCosines(np.array(orn[3:]), np.array(orn[:3]))

    @property
    def r_vec(self) -> np.ndarray:
        """Description:
        Read only property

        Returns:
        --------
        r_vec: ndarry
            unit vector that is orthogonal to all the slices of
            the DICOM sequence with shape (3, 1)
        """
        cosines = self.direction_cosines
        r_vec = np.cross(cosines.row, cosines.col)
        return r_vec.reshape((3, 1))

    @property
    def slice_origins(self):
        """Description:
        Read only poperty

        Returns:
        --------
        origins: ndarry
            Array of slice origins shape is (# of slices in sequence, 3)
            with x, y, z values as rows in slice order

        Raises:
        ------
        RuntimeError if origin metadata is not present for all slices
        of the DICOM sequence
        """
        if not self.has_metadata(self.pos_tag):
            raise RuntimeError(
                f"{self.pos_tag} metadata is not present"
                "in all slices of the DICOM sequence"
            )
        origins = np.empty((len(self), 3))
        for k, dcm_slice in enumerate(self):
            origins[k] = np.array(dcm_slice.data_element(self.pos_tag).value)
        return origins

    @property
    def w_vec(self) -> np.ndarray:
        """Description:
        Read only property, the direction of the vector may not match r_vec

        Returns:
        --------
        w_vec: ndarry
            unit vector pointing along the DICOM sequence slice
            origins with shape (3, 1)

        Raises:
        ------
        AttributeError if the unit vector between adjecent
        slices is not constant to within tolerances

        Notes:
        ------
        Calculates the vector pointing along slice origins in the sequences.
        Performs three linear interpolations using the x, y, & z values of
        each slice origin vs. the distance from the stack origin.
        The slopes of these interpolations correspond to the x, y, & z
        components of the vector w. Normalize the slopes into a vector w
        and set any values that are less than eps to equal zero
        (filters numerical errors).
        Normalize the vectors between slice origins and compute the dot product
        of w and all these vectors. If any of the dot products indicates an
        angle greater than 0.1 degrees then the slices are not aligned.
        """
        eps = 10**-16
        w_check = np.cos(np.deg2rad(0.1))
        origins = self.slice_origins
        d_vecs = np.array(origins[1:]) - np.array(origins[:-1])
        slice_distances = np.linalg.norm(d_vecs, axis=1)
        d_vecs = d_vecs / slice_distances[:, np.newaxis]
        total_dist = np.cumsum(slice_distances)
        total_dist = np.insert(total_dist, 0, 0.0, axis=0)
        del slice_distances

        # calculate w vector
        w_vec = np.array(
            [linregress(total_dist, origins[:, i])[0]
             for i in range(3)]
        )

        # normalize and round off floating point errors
        w_vec = w_vec / np.linalg.norm(w_vec)
        w_vec[np.abs(w_vec) < eps] = 0

        # check if all vectors between slices are roughly aligned
        w_dots = np.sum(d_vecs * w_vec, axis=1)
        if np.any(w_dots < w_check):
            raise RuntimeError("Slice origins do not lie on a single line")
        return w_vec.reshape((3, 1))

    @property
    def slice_distances(self):
        """Description:
        Read only property

        Returns:
        --------
        slice_distances: tuple
            contains the distance between slices
        """
        if len(self) == 1:
            return (0.0,)
        origins = self.slice_origins
        d_vecs = np.array(origins[1:]) - np.array(origins[:-1])
        d_norms = np.linalg.norm(d_vecs, axis=1)
        return tuple(d_norms)

    @property
    def uniform_spacing(self) -> bool:
        """Description:
        Read only property

        Returns:
        --------
        uniform_spacing: bool
            True is distnace between slice origins is uniform and
            False otherwise

        Notes:
        -----
        When called finds the maximum, minimum, and mean distance
        between slice origins using self.slice_spacing.
        If the difference between maximum and minimum slice
        distance is less than or equal to 1% of the mean
        distance then say the sequence has uniform spacing.
        """
        eps = 0.01
        slice_distances = self.slice_distances
        max_dist = np.nanmax(slice_distances)
        min_dist = np.nanmin(slice_distances)
        mean_dist = np.nanmean(slice_distances)

        max_rel_diff = (max_dist - mean_dist) / mean_dist
        min_rel_diff = (mean_dist - min_dist) / min_dist
        if max_rel_diff <= eps and min_rel_diff <= eps:
            return True
        return False

    @property
    def orthognal_axes(self) -> bool:
        """Description:
        Read only property

        Returns:
        --------
        orthogonal_axes: bool
            True if the DICOM sequence image axes are orthogonal
            and False otherwise.

        Notes:
        ------
        Determined by taking dot product of the current r vector and w vector
        """
        r_vec = np.squeeze(self.r_vec)
        w_vec = np.squeeze(self.w_vec)
        if abs(abs(np.dot(r_vec, w_vec)) - 1) < 1e-3:
            return True
        return False

    @property
    def direction_matrix(self):
        """Description:
        Read only property

        Returns:
        --------
        direction_matrix: ndarray
            3x3 matrix containing the coordinate system axes for the image

        Raises:
        ------
        RuntimeError: if not an orthogonal sequence

        Notes:
        ------
        Could use w_vec and allow non-orthogonal axes
        """
        if not self.orthognal_axes:
            raise RuntimeError("Image Axes are not orthogonal")

        cosines = self.direction_cosines
        d_matrix = np.empty((3, 3))
        d_matrix[:, 0] = cosines.row
        d_matrix[:, 1] = cosines.col
        d_matrix[:, 2] = np.squeeze(self.r_vec)
        return d_matrix

    def get_stack_metadata(self, key: str):
        """Return tuple of metadata values for each slice in stack"""
        return tuple([self.get_slice_tag_value(k, key)
                      for k in range(self.n_slices)])

    # pylint: disable=invalid-name
    def set_stack_metadata(self, key: str, VR: str, value):
        """
        Set the value of or create a new element and
        add it to each Dataset in the stack.

        Parameters:
        -------------
        k: int
            slice id number
        key: str
            keyword for the DICOM tag
        VR : str
            The 2 character DICOM value representation
            (see DICOM Standard, Part 5, Section 6.2<part05/sect_6.2.html>).
        value:
            The value of the data element. One of the following:

        - a single string or number
        - a list or tuple with all strings or all numbers
        - a multi-value string with backslash separator
        - for a sequence element, an empty list or list of Dataset
        """
        for k, _ in enumerate(self):
            self.set_slice_tag_value(k, key, VR, value)

    def get_slice_tag_value(self, k: int, key: str):
        """
        Return the value of an element in the Dataset in slice k.

        Parameters:
        -------------
        k: int
            slice id number
        key: str
            keyword for the DICOM tag

        Returns:
        ------------
        value:
            The value of the data element. One of the following:

        - a single string or number
        - a list or tuple with all strings or all numbers
        - a multi-value string with backslash separator
        - for a sequence element, an empty list or list of Dataset
        - None if the element is not present
        """
        if self[k].dir(key):
            return self[k][key].value
        return None

    def set_slice_tag_value(self, k: int, key: str, VR: str, value):
        """
        Set the value of or create a new element and
        add it to the Dataset in slice k.

        Parameters:
        -------------
        k: int
            slice id number
        key: str
            keyword for the DICOM tag
        VR : str
            The 2 character DICOM value representation
            (see DICOM Standard, Part 5, Section 6.2<part05/sect_6.2.html>).
        value:
            The value of the data element. One of the following:

        - a single string or number
        - a list or tuple with all strings or all numbers
        - a multi-value string with backslash separator
        - for a sequence element, an empty list or list of Dataset
        """
        if self[k].dir(key):
            self[k][key].value = value
        else:
            self[k].add_new(key, VR, value)

    # pylint: disable = useless-super-delegation
    def __getitem__(self, index) -> Dataset:
        """For Type hinting"""
        return super().__getitem__(index)

    def __iter__(self):
        self._iter_num = 0  # pylint: disable=attribute-defined-outside-init
        return self

    def __next__(self):
        if self._iter_num < len(self):
            result = self[self._iter_num]
            self._iter_num += 1
            return result
        else:
            raise StopIteration

    def __str__(self) -> str:
        tag_dict = dict()
        for dcm in self:
            lines = str(dcm).splitlines()
            for elem in lines:
                if ":" in elem:
                    desc, val = elem.split(":", maxsplit=1)
                    if not desc.startswith(" "):
                        if desc not in tag_dict:
                            tag_dict.update({desc: [val]})
                        else:
                            tag_dict[desc].append(val)

        title = "<" + type(self).__name__ + "> Metadata Summary"
        strings = self._properties_summary(title)
        for key in sorted(tag_dict.keys()):
            values = tag_dict[key]
            if len(values) == len(self):
                if all(x == values[0] for x in values):
                    strings.append(key + ":" + values[0])
                else:
                    strings.append(key + ": VARIABLE")
            else:
                strings.append(key + ": NOT PRESENT IN ALL SLICES")
        strings.append("-"*2*len(title))
        return "\n".join(strings)

    def _properties_summary(self, title: str):
        """Creates header with class name and properties summarized"""
        pad = 30
        cosines = self.direction_cosines
        strings = [title,
                   "-"*2*len(title),
                   f"{'Orthogonal Axes:':{pad}} {self.orthognal_axes}",
                   f"{'Direction Cosine (Row):':{pad}} {cosines.row}",
                   f"{'Direction Cosine (Column):':{pad}} {cosines.col}",
                   f"{'Uniform Slice Spacing:':{pad}} {self.uniform_spacing}",
                   "-"*2*len(title)]
        return strings
