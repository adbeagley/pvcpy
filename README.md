# pvcpy
Python code to take an input dicom and a mask dicom to determine the voxels making up a bone and then perform a partial volume correction and ouput a corrected dicom.

Make sure that the dicoms are stack data with 1 slice per file.

For details on the algorithm and its validation see the working paper "Effects of Material Mapping Agnostic Partial Volume Correction for Subject Specific Finite Elements Simulations" available on [arXiv](https://arxiv.org/)
