from os import PathLike
from typing import Union
from warnings import warn

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import binary_erosion
from vtkmodules.all import (  # pylint: disable = no-name-in-module
    vtkShepardKernel,
    vtkStaticPointLocator,
    vtkDoubleArray,
    vtkImageData,
    vtkImageContinuousErode3D,
    vtkImageMathematics
)
import pyvista as pv

from .dcmpy import CtScan, read_ct_scan
from .adaptors import (
    convert_3d_img_to_vtkarray,
    convert_array,
    convert_id_list,
    convert_matrix,
    convert_transform_matrix,
    convert_points
)



__all__ = ["surface_pvc"]


def surface_pvc(
    src_dcm_dir: PathLike,
    mask_dcm_dir: PathLike,
    data_name: str = "Intensity",
    background: int = None,
    plot_results: bool = True,
    return_vtk: bool = False,
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

    return_vtk: bool
        If True then return the partial volume corrected image
        as an ImageData object, otherwise return it as a CtScan

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
    pvc_data_name = "PVC" + data_name
    # -------------------------------

    _check_flags_are_bools(plot_results, return_vtk, verbose, progress_bar)

    if (background is not None) and np.isnan(background) and (not return_vtk):
        background = None
        warn("NaN cannot be stored in a DICOM. Background set to None.")

    src_dcm = read_ct_scan(src_dcm_dir, binary_data=False)
    mask_dcm = read_ct_scan(mask_dcm_dir, binary_data=True)
    binary_mask = mask_dcm.pixel_array

    if not src_dcm.is_aligned(mask_dcm):
        raise ValueError("Expected source and mask DICOMs to be aligned.")

    src_img = ctscan_to_vtkimage(src_dcm, data_name)

    # add segmentation mask to source image
    src_img.GetPointData().AddArray(
        convert_3d_img_to_vtkarray(binary_mask.astype(dtype=np.uint8), "Mask")
    )

    # define surface and interior points
    interior: np.ndarray = binary_erosion(binary_mask).astype(dtype=np.uint8)
    surface = np.array(binary_mask - interior, dtype=np.uint8)

    src_img.GetPointData().AddArray(
        convert_3d_img_to_vtkarray(
            interior.astype(dtype=np.uint8), "Interior"
        )
    )

    src_img.GetPointData().AddArray(
        convert_3d_img_to_vtkarray(
            surface.astype(dtype=np.uint8), "Surface"
        )
    )

    # get array of surface point ids to perform pvc on
    surface = surface.flatten("F")
    surf_pt_ids = np.squeeze(np.argwhere(surface == 1))

    # get set of interior values to interpolate from (set -> faster search)
    interior = interior.flatten("F")
    interior_pt_ids = set(np.squeeze(np.argwhere(interior == 1)))

    # get array of intensity data
    intensity: np.ndarray = convert_array(
        src_img.GetPointData().GetAbstractArray(data_name)
    )

    # use static point locator
    locator = vtkStaticPointLocator()
    locator.SetDataSet(src_img)
    locator.Update()

    # create interpolation kernel
    idw_kernel = vtkShepardKernel()
    idw_kernel.SetPowerParameter(power)
    idw_kernel.NormalizeWeightsOn()
    idw_kernel.Initialize(locator,
                          src_img,
                          src_img.GetPointData())
    idw_kernel.RequiresInitializationOff()  # manually intialized

    with tqdm(surf_pt_ids, desc="PVC") if progress_bar else surf_pt_ids as pbar:
        for pt_id in pbar:
            # get connected interior points
            connected_pt_ids = src_img.get_connected_points(pt_id)
            basis_pt_ids = convert_id_list(
                [idx for idx in connected_pt_ids if idx in interior_pt_ids]
            )

            # interpolate from connected interior points
            if len(basis_pt_ids) > 0:
                point = src_img.get_point(pt_id)
                wgts = vtkDoubleArray()
                idw_kernel.ComputeWeights(
                    point,          # point to interpolate onto
                    basis_pt_ids,   # points to interpolate from
                    None,           # set probability function to 1's
                    wgts            # interpolation weights
                )
                vals = intensity[convert_id_list(basis_pt_ids)]
                idw_val = round(np.sum(vals * wgts))  # DICOMs require ints
                if idw_val > intensity[pt_id]:        # update val if greater
                    intensity[pt_id] = idw_val

    src_img.point_data[pvc_data_name] = intensity

    # apply background value is supplied one
    if background is not None:
        if not isinstance(background, int) and not np.isnan(background):
            background = round(background)
        mask = src_img.point_data["Mask"]
        pvc_intensity = src_img.point_data[pvc_data_name]
        pvc_intensity = np.where(mask == 0, background, pvc_intensity)
        src_img.point_data[pvc_data_name] = pvc_intensity

    if plot_results:
        _plot_histo(src_img, data_name, pvc_data_name)
        _plot_meshes(src_img, data_name, pvc_data_name)

    # return result
    if not return_vtk:
        src_dcm.pixel_array = src_img.get_3d_point_data(pvc_data_name)
        return src_dcm

    return src_img

def ctscan_to_vtkimage(ctscan: CtScan, data_name: str):
    """Convert a CtScan instance into a vtkImageData instance."""
    vtkarr = convert_3d_img_to_vtkarray(ctscan.pixel_array, data_name)

    img = vtkImageData()
    img.SetOrigin(ctscan.origin)
    img.SetDirectionMatrix(convert_matrix(ctscan.direction_matrix))
    img.SetDimensions(ctscan.dimensions)
    img.SetSpacing(ctscan.spacing)
    img.GetPointData().AddArray(vtkarr)
    return img


# def erode_vtkimage(img: vtkImageData, arr_name: str) -> vtkImageData:
#     """Erode the point data array specified by arr_name."""
#     img.GetPointData().SetActiveScalars(arr_name)

#     alg = vtkImageContinuousErode3D()
#     alg.SetInputData(img)
#     alg.SetKernelSize(3, 3, 3)
#     alg.Update()
#     return alg.GetOutput()



def _plot_histo(src_img: vtkImageData, data_name: str, pvc_data_name: str):
    """Description:
    Creates and plots a histogram of the magnitude of HU correction at
    each surface voxel
    """
    surface = src_img.point_data["Surface"]
    intensity = src_img.point_data[data_name]
    pvc_intensity = src_img.point_data[pvc_data_name]

    # get change in intensity values
    pvc_diff = pvc_intensity - intensity
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
    plt.show()


def _plot_meshes(src_img: vtkImageData, data_name: str, pvc_data_name: str):
    """Plot before and after visualizaiton of segmented voxels"""
    src_mesh = src_img.points_to_cells()
    bone_mesh = src_mesh.extract_cells(src_mesh.cell_data["Mask"])

    sargs = dict(fmt="%.0f")

    plotter = pv.Plotter(shape=(1, 2))
    plotter.subplot(0, 0)
    plotter.add_title("Original Intensity")
    plotter.add_mesh(bone_mesh, scalars=data_name, scalar_bar_args=sargs)

    plotter.subplot(0, 1)
    plotter.add_title("PVC Intensity")
    plotter.add_mesh(bone_mesh, scalars=pvc_data_name, scalar_bar_args=sargs)
    plotter.show()


def _check_flags_are_bools(*flags):
    """Checks each value is a bool"""
    for flag in flags:
        if not isinstance(flag, bool):
            raise TypeError("Invalid type for boolean flag keyword.")
