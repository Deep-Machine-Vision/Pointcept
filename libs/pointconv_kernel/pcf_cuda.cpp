//
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025 DMV Lab
//

#include <torch/extension.h>
#include "pcf.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pconv_forward", &pcf::pconv_forward, "PointConv forward (CUDA)");
    m.def("pconv_linear_forward", &pcf::pconv_linear_forward, "Fused PointConv+Linear forward (CUDA)");
    m.def("pconv_backward", &pcf::pconv_backward, "PointConv backward (CUDA)");
    m.def("pconv_linear_backward", &pcf::pconv_linear_backward, "Fused PointConv+Linear backward (CUDA)");
    m.def("pconv_linear_opt_backward", &pcf::pconv_linear_opt_backward, "Optimized Fused PointConv+Linear backward (CUDA)");
    m.def("compute_knn_inverse", &pcf::compute_knn_inverse, "Compute KNN inverse mapping (CUDA)");
    m.def("pconv_linear_cutlass_forward", &pcf::pconv_linear_cutlass, "PConv Linear forward (CUTLASS)");
}
