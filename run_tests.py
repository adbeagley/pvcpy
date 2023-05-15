"""Run tests using pytest."""
import pytest

if __name__ == "__main__":
    import pvcpy
    src_dcm = r"Z:\Aren\Porcine PVC\Specimen 3\DICOMS\S3_Raw"
    mask_dcm = r"Z:\Aren\Porcine PVC\Specimen 3\DICOMS\S3_Mask"
    dcm = pvcpy.surface_pvc(src_dcm_dir=src_dcm, mask_dcm_dir=mask_dcm)
    print(dcm)

    exit()
    retcode = pytest.main()
