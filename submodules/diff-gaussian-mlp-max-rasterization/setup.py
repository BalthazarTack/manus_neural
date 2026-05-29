#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

DEBUG = os.environ.get("DEBUG_CUDA", "0") == "1"

os.path.dirname(os.path.abspath(__file__))

extra_nvcc_flags = [
    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")
]

if DEBUG:
    print("Compiling with CUDA debug flags (-G -lineinfo, TORCH_USE_CUDA_DSA)")
    extra_nvcc_flags += ["-G", "-lineinfo", "-DTORCH_USE_CUDA_DSA"]

setup(
    name="diff_gaussian_mlp_max_rasterization",
    packages=["diff_gaussian_mlp_max_rasterization"],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_mlp_max_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_mlp_impl.cu",
                "cuda_rasterizer/forward_mlp.cu",
                "cuda_rasterizer/backward_mlp.cu",
                "cuda_rasterizer/mlp_kernels.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            # extra_compile_args={
            #     "nvcc": [
            #         "-I"
            #         + os.path.join(
            #             os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"
            #         )
            #     ]
            # },
            extra_compile_args={"nvcc": extra_nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
