"""Run tests using pytest."""
import pytest

if __name__ == "__main__":
    import pvcpy
    src_dcm = r"E:\1_Masters\!Data\GSA\GSA_11\GSA_11_dicom_scap_raw"
    mask_dcm = r"E:\1_Masters\!Data\GSA\GSA_11\GSA_11_dicom_scap_mask"
    dcm = pvcpy.surface_pvc(src_dcm_dir=src_dcm, mask_dcm_dir=mask_dcm)
    print(dcm)

    exit()
    retcode = pytest.main()
