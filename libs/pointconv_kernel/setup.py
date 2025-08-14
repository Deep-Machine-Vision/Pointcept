# PCF CUDA Kernel:
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
#
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the absolute path to the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

src_files = [
    'pcf_cuda.cpp',
    'src/pcf.cu',
    'src/common.cu',
    'src/knn.cu',
    'src/pconv_ops.cu',
]

setup(
    name='PCFcuda',
    version='1.1',
    author='Logeswaran Sivakumar, Stefan Lee,  Pritesh Verma, Skand Peri, Li Fuxin',
    author_email='loges.siva14@gmail.com',
    description='PointConvFormer CUDA Kernel',
    ext_modules=[
        CUDAExtension('pcf_cuda', src_files,
        include_dirs=[
            os.path.join(project_root, 'cpp_wrappers/cpp_pcf_kernel/include'),
            os.path.join(project_root, 'cutlass/include'),
            os.path.join(project_root, 'cutlass/tools/util/include'),
            ],
        extra_compile_args={'nvcc': ['-L/usr/local/cuda/lib64 -lcudadevrt -lcudart']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
