from os import PathLike
from typing import Tuple

from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from scipy.ndimage import binary_erosion
from pyvista import Plotter

# pylint: disable = no-name-in-module
from vtkmodules.all import vtkExtractCells

from .dcmpy import CtScan, read_ct_scan
from .adaptors import (
    ctscan_to_vtkimage,
    convert_array,
    convert_id_list
)


__all__ = ["surface_pvc"]


def surface_pvc(
    src_dcm_dir: PathLike,
    mask_dcm_dir: PathLike,
    background: int = None,
    plot_results: bool = True,
    verbose: bool = True,
    progress_bar: bool = True
):
    """Description:
    Algorithm to perform partial volume
    correction on the surface voxels of a CT scan.

    Parameters:
    ------------
    src_dcm: str | PathLike
        path to the DICOM sequence CT scan being corrected

    mask_dcm: str | PathLike
        path to the DICOM sequence containing a binary mask of the
        bone to undergo partial volume correction

    background: None | int
        if not None then set all values not included in the segmentation
        mask equal to the given background value, otherwise leave
        background values untouched

    plot_results: bool
        if True then generate histogram of corrections to surface voxels
        and visualize the segmented bone, otherwise do nothing

    verbose: bool
        if True then display metadata regarding the input DICOMs

    progress_bar: bool
        if True then display progress bar

    Returns:
    ------------
    pvc_dcm: CtScan | ImageData
        corrected DICOM sequence as a CtScan or ImageData object

        if it is an ImageData object then it contains the point data arrays:
            str(data_name),
            "PVC" + str(data_name),
            "Mask",
            "Surface",
            "Interior",

    Notes:
    ------
    Algorithm only corrects surface voxels that have adjacent interior
    voxels. Has a tendency to miss correcting very thin regions (if
    region is 1-2 voxels thick there are no interior voxels there) and
    small but highly convex features are also not corrected due to
    having no adjacent interior voxels.
    """
    # ------------------------------
    power = 2
    pad = 2
    # -------------------------------

    _check_flags_are_bools(plot_results, verbose, progress_bar)

    if (background is not None) and not isinstance(background, int):
        raise ValueError(
            "Background is non-integer value. Cannot be stored in DICOM."
        )

    src_dcm = read_ct_scan(src_dcm_dir, binary_data=False)
    mask_dcm = read_ct_scan(mask_dcm_dir, binary_data=True)

    if not src_dcm.orthognal_axes:
        raise ValueError("DICOM must have orthogonal image axes")

    if not src_dcm.uniform_spacing:
        raise ValueError("Non-uniform slice spacing not supported by pvcpy.")

    if not src_dcm.is_aligned(mask_dcm):
        raise ValueError("Expected source and mask DICOMs to be aligned.")

    hu_data = src_dcm.pixel_array
    binary_mask = mask_dcm.pixel_array

    # pad arrays to avoid issues with kernel near the edges
    hu_data = pad_image(hu_data, pad_width=pad, pad_val=0)
    binary_mask = pad_image(binary_mask, pad_width=pad, pad_val=0)

    # define surface and interior points
    interior: np.ndarray = binary_erosion(binary_mask).astype(dtype=np.int8)
    surface = np.array(binary_mask - interior)

    # define the interpolation kernel
    kernel = create_idw_kernel(np.asarray(src_dcm.spacing), power)

    # get array of surface point indexes to perform pvc on
    surf_ids = np.argwhere(surface == 1)

    # for each surface point apply the kernel to it's
    # neighborhood to compute the new value
    # multiply by that neighborhood in the interior mask first though
    # to set non-internal point weights to zero
    pvc_data = np.array(hu_data)  # copy hu_data
    with tqdm(surf_ids, desc="PVC") if progress_bar else surf_ids as pbar:
        for pt_loc in pbar:
            wgts = get_neighborhood(pt_loc, interior)*kernel  # mask
            if np.count_nonzero(wgts) != 0:
                wgts /= np.sum(wgts)  # normalize weights
                idw_val = np.sum(wgts * get_neighborhood(pt_loc, hu_data))
                i, j, k = pt_loc[0], pt_loc[1], pt_loc[2]
                pvc_data[i, j, k] = max(idw_val, pvc_data[i, j, k])

    if background is not None:  # set background voxel values
        pvc_data = np.where(binary_mask, pvc_data, background)

    print("Reduced Intensity:", np.count_nonzero(pvc_data - hu_data < 0))

    # crop arrays to remove padding
    pvc_data = remove_padding(pvc_data, pad_width=pad)
    src_dcm.pixel_array = pvc_data

    if plot_results:
        hu_data = remove_padding(hu_data, pad_width=pad)
        surface = remove_padding(surface, pad_width=pad)
        binary_mask = remove_padding(binary_mask, pad_width=pad)
        _plot_histo(hu_data, pvc_data, surface)
        _plot_meshes(src_dcm, hu_data, pvc_data, binary_mask)
        plt.show()  # hold figures open until they are closed.
    return src_dcm


def pad_image(img: NDArray, pad_width: int, pad_val: float = 0):
    """Pad all axes before and after to width.
    So pad_width of 2 expands a dimension by 4."""
    widths = tuple((pad_width, pad_width) for _ in range(img.ndim))
    return np.pad(img, widths, "constant", constant_values=pad_val)


def remove_padding(img: NDArray, pad_width: int):
    """Removes padding from beginning and end of each axis."""
    return img[pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width]


def get_neighborhood(loc: Tuple[int, int, int], img: NDArray):
    """Get the 3x3 cube of voxels centered on the point at loc."""
    # nbrs = np.empty((3,3,3))
    # for i in range(-1, 2, 1):
    #     for j in range(-1, 2, 1):
    #         for k in range(-1, 2, 1):
    #             ii, jj, kk = loc + np.array([i, j, k])
    #             try:
    #                 nbrs[i, j, k] = img[ii, jj, kk]
    #             except IndexError:
    #                 nbrs[ii, jj, kk] = 0
    # return nbrs
    idx = []
    for i, dim in enumerate(img.shape):
        start = max(0, loc[i]-1)
        stop = min(dim, loc[i]+2)
        idx.append(slice(start, stop))
    nbrs = img[idx[0], idx[1], idx[2]]
    if nbrs.shape == (3, 3, 3):
        return nbrs
    raise RuntimeError("Insufficient padding of image.")


def create_idw_kernel(spacing: NDArray, power: float):
    """Create IDW kernel for image grid."""
    ddi = spacing[0]
    ddj = spacing[1]
    ddk = spacing[2]
    ddij = np.linalg.norm([ddi, ddj])
    ddik = np.linalg.norm([ddi, ddk])
    ddjk = np.linalg.norm([ddj, ddk])
    ddijk = np.linalg.norm([ddi, ddj, ddk])

    kernel = np.empty((3, 3, 3))
    kernel[:, :, 0] = np.array(
        [[ddijk, ddik, ddijk],
         [ddjk, ddk, ddjk],
         [ddijk, ddik, ddijk]]
    )
    kernel[:, :, 1] = np.array(
        [[ddij, ddi, ddij],
         [ddj, 1, ddj],
         [ddij, ddi, ddij]]
    )
    kernel[:, :, 2] = kernel[:, :, 0]
    kernel = 1 / np.power(kernel, power)
    kernel = np.where(np.isnan(kernel), 0, kernel)  # center point to zero
    return kernel


def _plot_histo(
    hu_data: NDArray,
    pvc_data: NDArray,
    surface: NDArray
):
    """Description:
    Creates and plots a histogram of the magnitude of HU correction at
    each surface voxel
    """
    # get change in intensity values
    pvc_diff = pvc_data - hu_data
    pvc_diff = pvc_diff[surface == 1]

    max_diff = np.nanmax(pvc_diff)
    min_diff = np.nanmin(pvc_diff)
    n_bins = round(max_diff - min_diff)
    bin_range = (min_diff, max_diff)
    n_voxels = len(pvc_diff)

    vals, bin_edges = np.histogram(
        pvc_diff, bins=n_bins, range=bin_range, density=True)
    vals = vals * 100  # convert to percentage of voxels with this change

    # exclude points with zero change to avoid poor y-axis scale
    fig, axs = plt.subplots(2)
    fig.suptitle('PVC \u0394HU for Surface Voxels')  #
    axs[0].bar(bin_edges[1:-1], vals[1:], width=1)
    axs[0].set(ylabel='Percentage of Voxels')
    axs[1].bar(bin_edges[1:-1], vals[1:] * n_voxels / 100, width=1)
    axs[1].set(ylabel='Number of Voxels')
    axs[1].set(xlabel='|\u0394HU| Correction')
    plt.tight_layout()
    msg = (f"Voxels with no correction: {int(vals[0] * n_voxels / 100)}"
           f" or {vals[0]:0.2f}%")
    plt.figtext(0.45, 0.02, msg)

    back_end = plt.get_backend()
    fig_manager = plt.get_current_fig_manager()
    if back_end == "TkAgg":
        fig_manager.window.state('zoomed')
    elif back_end == "QT4Agg":
        fig_manager.window.showMaximized()
    elif back_end == "wxAgg":
        fig_manager.frame.Maximize(True)
    plt.show(block=False)


def _plot_meshes(
    src_dcm: CtScan,
    hu_data: NDArray,
    pvc_data: NDArray,
    mask: NDArray
):
    """Plot before and after visualization of segmented voxels"""
    mesh = ctscan_to_vtkimage(src_dcm, as_cells=True)
    mesh.GetCellData().AddArray(
        convert_array(hu_data.flatten("F"), "Raw HU-Intensity")
    )
    mesh.GetCellData().AddArray(
        convert_array(pvc_data.flatten("F"), "PVC HU-Intensity",)
    )
    mask = mask.flatten("F")
    mask = np.squeeze(np.argwhere(mask == 1))
    mask_ids = convert_id_list(mask)

    extractor = vtkExtractCells()
    extractor.SetInputData(mesh)
    extractor.SetCellList(mask_ids)
    extractor.Update()
    mesh = extractor.GetOutput()

    sargs = dict(fmt="%.0f")

    plotter = Plotter(shape=(1, 2))
    plotter.subplot(0, 0)
    plotter.add_title("Original Intensity")
    plotter.add_mesh(mesh, scalars="Raw HU-Intensity", scalar_bar_args=sargs)

    plotter.subplot(0, 1)
    plotter.add_title("PVC Intensity")
    plotter.add_mesh(mesh, scalars="PVC HU-Intensity", scalar_bar_args=sargs)
    plotter.show()


def _check_flags_are_bools(*flags):
    """Checks each value is a bool"""
    for flag in flags:
        if not isinstance(flag, bool):
            raise TypeError("Invalid type for boolean flag keyword.")
