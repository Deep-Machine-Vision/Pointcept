//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
//
#pragma once

#include <torch/extension.h>
#include <vector>

namespace pcf {
namespace pconv_ops {

/**
 * @brief Kernel: Forward pass for Point Convolution (PConv)
 * 
 * @param input Input features [B x M x C_in]
 *        B = batch size, M = number of points in the original point cloud, C_in = input channels
 * @param neighbor_inds Neighbor indices [B x N x K]
 *        K = number of neighbors per point
 * @param weights Weight [B x N x K x C_mid]
 *        C_mid = mid channels
 * @param additional_features Additional features [B x N x K x C_add]
 *        C_add = additional features that do not require indexing
 * @param output Output [B x N x C_mid * (C_in + C_add)]
 */
template <typename scalar_t>
__global__ void pconv_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ input,
    const torch::PackedTensorAccessor32<long, 3, torch::RestrictPtrTraits> __restrict__ neighbor_inds,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ weights,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ additional_features,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ output
);

/**
 * @brief Kernel: Forward pass for fused Point Convolution and Linear layer
 * 
 * @param input Input features [B x M x C_in]
 *        B = batch size, M = number of points in the original point cloud, C_in = input channels
 * @param neighbor_inds Neighbor indices [B x N x K]
 *        K = number of neighbors per point
 * @param weights Weight [B x N x K x C_mid]
 *        C_mid = mid channels
 * @param additional_features Additional features [B x N x K x C_add]
 *        C_add = additional features that do not require indexing
 * @param linear_weights Linear layer weights [C_out x (C_mid*(C_in + C_add))]
 * @param linear_bias Linear layer bias [C_out]
 * @param final_output Final output [B x N x C_out]
 * @param pconv_output PConv output [B x N x C_mid * (C_in + C_add)]
 */
template <typename scalar_t>
__global__ void pconv_linear_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ input,
    const torch::PackedTensorAccessor32<long, 3, torch::RestrictPtrTraits> __restrict__ neighbor_inds,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ weights,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ additional_features,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> __restrict__ linear_weights,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> __restrict__ linear_bias,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ final_output,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ pconv_output
);

/**
 * @brief Kernel: Backward pass for Point Convolution
 * 
 * @param grad_output Gradient of output [B x N x (C_mid * C_in)]
 *        B = batch size, N = number of points, C_in = input channels, C_mid = mid channels
 * @param input Input features [B x N x C_in]
 *        B = batch size, N = number of points, C_in = input channels
 * @param neighbor_inds Neighbor indices [B x N x K]
 *        K = number of neighbors per point
 * @param weights Weight [B x N x K x C_mid]
 *        C_mid = mid channels
 * @param additional_features Additional features [B x N x K x C_add]
 *        C_add = additional features that do not need gather
 * @param grad_input [B x N x C_in]
 * @param grad_weights [B x N x K x C_mid]
 * @param grad_additional [B x N x K x C_add]
 */
template <typename scalar_t>
__global__ void pconv_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ grad_output,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ input,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> __restrict__ neighbor_inds,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ weights,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ additional_features,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ grad_input,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ grad_weights,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ grad_additional
);

/**
 * @brief Kernel: Backward pass for Fused Point Convolution + Linear Layer
 * 
 * @param grad_output Gradient of output [B x N x (C_mid * C_in)]
 *        B = batch size, N = number of points, C_in = input channels, C_mid = mid channels
 * @param input Input features [B x N x C_in]
 *        B = batch size, N = number of points, C_in = input channels
 * @param neighbor_inds Neighbor indices [B x N x K]
 *        K = number of neighbors per point
 * @param weights Weight [B x N x K x C_mid]
 *        C_mid = mid channels
 * @param additional_features Additional features [B x N x K x C_add]
 *        C_add = additional features that do not need gather
 * @param linear_weights Linear layer weights [C_out x (C_mid*(C_in + C_add))]
 * @param pconv_output PConv output tensor [B x N x C_mid * (C_in + C_add)]
 * @param grad_input [B x N x C_in]
 * @param grad_weights [B x N x K x C_mid]
 * @param grad_additional [B x N x K x C_add]
 * @param grad_linear_weights [C_out x (C_mid*(C_in + C_add))]
 * @param grad_linear_bias [C_out]
 */
template <typename scalar_t>
__global__ void pconv_linear_cuda_backward_kernel(
        const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ grad_output,
        const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ input,
        const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> __restrict__ neighbor_inds,
        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ weights,
        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ additional_features,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> __restrict__ linear_weights,
        const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ pconv_output,
        torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ grad_input,
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ grad_weights,
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ grad_additional,
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> __restrict__ grad_linear_weights,
        torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> __restrict__ grad_linear_bias
);

/**
 * @brief Kernel: Backward pass for Fused Point Convolution + Linear Layer with Inverse Indices
 * 
 * @param grad_output
 * @param input Input features [B x M x C_in]
 * @param inverse_neighbor [B, (N * K)]
 * @param inverse_neighbor_k [B, (N * K)]
 * @param inverse_neighbor_idx [B, (total_points + 1)]
 * @param neighbor_inds Neighbor indices [B x N x K]
 * @param weights Weight [B x N x K x C_mid]
 * @param additional_features Additional features [B x N x K x C_add]
 * @param linear_weights Linear layer weights [C_out x (C_mid*(C_in+C_add))]
 * @param pconv_output PConv output [B x N x (C_mid*(C_in+C_add))]
 * @param grad_input [B x N x C_in]
 * @param grad_weights [B x N x K x C_mid]
 * @param grad_additional [B x N x K x C_add]
 * @param grad_linear_weights [C_out x (C_mid*(C_in + C_add))]
 * @param grad_linear_bias [C_out]
 */
template <typename scalar_t>
__global__ void pconv_linear_fused_cuda_backward_kernel_opt(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_output,
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input,
        const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> inverse_neighbor,
        const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> inverse_neighbor_k,
        const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> inverse_neighbor_idx,
        const torch::PackedTensorAccessor32<long, 3, torch::RestrictPtrTraits> neighbor_inds,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> weights,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> additional_features,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> linear_weights,
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> pconv_output,
        torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_input,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_weights,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_additional,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_linear_weights,
        torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> grad_linear_bias
);

/**
 * @brief Kernel: (Input Only points) Backward pass for Fused Point Convolution + Linear Layer with Inverse Indices
 * 
 * @param grad_output
 * @param input Input features [B x M x C_in]
 * @param inverse_neighbor [B, (N * K)]
 * @param inverse_neighbor_k [B, (N * K)]
 * @param inverse_neighbor_idx [B, (total_points + 1)]
 * @param weights Weight [B x N x K x C_mid]
 * @param linear_weights Linear layer weights [C_out x (C_mid*(C_in+C_add))]
 * @param grad_input [B x N x C_in]
 */
template <typename scalar_t>
__global__ void input_only_backward_kernel(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_output,
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input,
        const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> inverse_neighbor,
        const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> inverse_neighbor_k,
        const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> inverse_neighbor_idx,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> weights,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> linear_weights,
        torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_input,
        const int N, const int Nout, const int C_in, const int C_mid, const int C_add, const int C_out, const int input_only_points
);

/**
 * @brief Kernel: Scattered-Gather using Neighbor Indices around Input points
 * 
 * @param input Input features [B x M x C_in]
 * @param additional_features Additional features [B x N x K x C_add]
 * @param neighbor_inds Neighbor indices [B x N x K]
 * @param concatenated_output
 */
__global__ void gather_kernel(
    const float* input,
    const float* additional_features,
    const int64_t* neighbor_inds,
    float* concatenated_output,
    int B, int M, int Nout, int K, int C_in, int C_add
);

torch::Tensor pconv_cuda_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features
);

std::vector<torch::Tensor> pconv_linear_cuda_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor linear_bias
);

std::vector<torch::Tensor> pconv_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features
);

std::vector<torch::Tensor> pconv_linear_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor pconv_output
);

std::vector<torch::Tensor> pconv_linear_opt_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor inverse_neighbor,
    torch::Tensor inverse_neighbor_k,
    torch::Tensor inverse_neighbor_idx,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor pconv_output
);

/**
 * @brief Forward pass for PConv and Linear using CUTLASS
 * 
 * @param input Input features [B x M x C_in]
 *        B = batch size, M = number of points in the original point cloud, C_in = input channels
 * @param neighbor_inds Neighbor indices [B x N x K]
 *        K = number of neighbors per point
 * @param weights Weight [B x N x K x C_mid]
 *        C_mid = mid channels
 * @param additional_features Additional features [B x N x K x C_add]
 *        C_add = additional features that do not require indexing
 * @param linear_weights Linear layer weights [C_out x (C_mid*(C_in + C_add))]
 * @param linear_bias Linear layer bias [C_out]
 * @return Vector of output tensors
 *         - final_output: [B x Nout x C_out] Output after applying linear layer
 *         - pconv_output: [B x N x C_mid * (C_in + C_add)] Intermediate output after PConv, saved for gradient computation
 */
std::vector<torch::Tensor> pconv_linear_cutlass_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor linear_bias
);

} // namespace pconv_ops
} // namespace pcf
