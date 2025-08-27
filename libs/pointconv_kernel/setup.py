# PCF CUDA Kernel:
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Oregon State University. All Rights Reserved.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

src_files = [
    'pcf_cuda.cpp',
    'src/pcf.cu',
    'src/common.cu',
    'src/knn.cu',
    'src/pconv_ops.cu',
]

class BuildExtWithCutlass(BuildExtension):
    # Add a new user option: --cutlass-dir=...
    user_options = BuildExtension.user_options + [
        ('cutlass-dir=', None, 'Path to CUTLASS root directory'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.cutlass_dir = None  # populated from CLI/env/default

    def finalize_options(self):
        super().finalize_options()
        print(f"[setup] CUTLASS_DIR parsed: {self.cutlass_dir!r}")
        # Priority: CLI > ENV > default submodule
        if not self.cutlass_dir:
            self.cutlass_dir = os.environ.get(
                'CUTLASS_DIR',
                os.path.join(current_dir, 'third_party', 'cutlass')
            )

        # Validate early with a helpful message
        if not os.path.exists(self.cutlass_dir):
            raise FileNotFoundError(
                f"CUTLASS directory not found at: {self.cutlass_dir}\n"
                "Please either:\n"
                "  • pass --cutlass-dir=/path/to/cutlass\n"
                "  • set environment variable CUTLASS_DIR=/path/to/cutlass\n"
                "  • or place CUTLASS in third_party/cutlass relative to this setup.py"
            )

    def build_extensions(self):
        # Inject CUTLASS include dirs dynamically
        for ext in self.extensions:
            ext.include_dirs = list(ext.include_dirs) + [
                os.path.join(self.cutlass_dir, 'include'),
                os.path.join(self.cutlass_dir, 'tools', 'util', 'include'),
            ]
        super().build_extensions()

setup(
    name='PCFcuda',
    version='2.0',
    author='Logeswaran Sivakumar, Stefan Lee,  Pritesh Verma, Skand Peri, Li Fuxin',
    author_email='loges.siva14@gmail.com',
    description='PointConv CUDA Kernel',
    ext_modules=[
        CUDAExtension(
            'pcf_cuda',
            src_files,
            include_dirs=[os.path.join(current_dir, 'include')],
            extra_compile_args={'nvcc': ['-L/usr/local/cuda/lib64', '-lcudadevrt', '-lcudart']},
        )
    ],
    cmdclass={'build_ext': BuildExtWithCutlass},
)
