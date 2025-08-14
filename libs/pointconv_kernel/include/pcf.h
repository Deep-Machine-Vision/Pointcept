//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
//
#pragma once

#include <torch/extension.h>

namespace pcf {

/**
 * @brief Check if tensor is on CUDA device
 */
#define CHECK_CUDA(x) {TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")}

/**
 * @brief Check if tensor is contiguous
 */
#define CHECK_CONTIGUOUS(x) {TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")}

/**
 * @brief Check if tensor is on CUDA device and contiguous
 */
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/**
 * @brief Wrapper: Forward pass for Point Convolution (PConv)
 * 
 * @param input Input features [B x M x C_in]
 *        B = batch size, M = number of points in the original point cloud, C_in = input channels
 * @param neighbor_inds Neighbor indices [B x N x K]
 *        K = number of neighbors per point
 * @param weights Weight [B x N x K x C_mid]
 *        C_mid = mid channels
 * @param additional_features Additional features [B x N x K x C_add]
 *        C_add = additional features that do not require indexing
 * @return Output [B x N x C_mid * (C_in + C_add)]
 */
torch::Tensor pconv_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features
);

/**
 * @brief Wrapper: Backward pass for Point Convolution (PConv)
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
 * @return Vector of gradient
 *         - grad_input: same shape as input [B x N x C_in]
 *         - grad_weights: same shape as weights [B x N x K x C_mid]
 *         - grad_additional: same shape as additional_features [B x N x K x C_add]
 */
std::vector<torch::Tensor> pconv_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features
);

/**
 * @brief Wrapper: Forward pass for fused PConv and Linear operation
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
std::vector<torch::Tensor> pconv_linear_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor linear_bias
);

/**
 * @brief Wrapper: Backward pass for fused PConv and Linear operation
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
 * @return Vector of gradient tensors
 *         - grad_input: same shape as input [B x N x C_in]
 *         - grad_weights: same shape as weights [B x N x K x C_mid]
 *         - grad_additional: same shape as additional_features [B x N x K x C_add]
 *         - grad_linear_weights: same shape as linear_weights [C_out x (C_mid*(C_in + C_add))]
 *         - grad_linear_bias: same shape as linear_bias [C_out]
 */
std::vector<torch::Tensor> pconv_linear_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor pconv_output
);

/**
 * @brief Wrapper: Compute KNN inverse mapping
 * 
 * @param neighbor_inds Neighbor indices [B x N x K]
 *        B = batch size, N = number of points, K = number of neighbors per point
 * @param total_points Total number of points
 * @return Vector of output tensors
 *         - inv_neighbors: List of points that reference each target point [B, (N * K)]
 *         - inv_k: Corresponding k-index in original neighbor_inds tensor [B, (N * K)]
 *         - inv_idx: Prefix sum indicating start/end positions in inv_neighbors per point [B, (total_points + 1)]
 */
std::vector<torch::Tensor> compute_knn_inverse(
    torch::Tensor neighbor_inds,
    const int total_points
);

/**
 * @brief Wrapper: Optimized backward pass for fused PConv and Linear operation
 * 
 * @param grad_output Gradient of output [B x N x (C_mid * C_in)]
 *        B = batch size, N = number of points, C_in = input channels, C_mid = mid channels
 * @param input Input features [B x N x C_in]
 *        B = batch size, N = number of points, C_in = input channels
 * @param inverse_neighbor Inverse neighbor [B, (N * K)]
 * @param inverse_neighbor_k Inverse neighbor k [B, (N * K)]
 * @param inverse_neighbor_idx Inverse neighbor index [B, (total_points + 1)]
 * @param neighbor_inds Neighbor indices [B x N x K]
 *        K = number of neighbors per point
 * @param weights Weight [B x N x K x C_mid]
 *        C_mid = mid channels
 * @param additional_features Additional features [B x N x K x C_add]
 *        C_add = additional features that do not need gather
 * @param linear_weights Linear layer weights [C_out x (C_mid*(C_in + C_add))]
 * @param pconv_output PConv output tensor [B x N x C_mid * (C_in + C_add)]
 * @return Vector of gradient tensors
 *         - grad_input: same shape as input [B x N x C_in]
 *         - grad_weights: same shape as weights [B x N x K x C_mid]
 *         - grad_additional: same shape as additional_features [B x N x K x C_add]
 *         - grad_linear_weights: same shape as linear_weights [C_out x (C_mid*(C_in + C_add))]
 *         - grad_linear_bias: same shape as linear_bias [C_out]
 */
std::vector<torch::Tensor> pconv_linear_opt_backward(
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
 * @brief Wrapper: Forward pass for PConv and Linear using CUTLASS
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
std::vector<torch::Tensor> pconv_linear_cutlass(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor linear_bias
);

} // namespace pcf
