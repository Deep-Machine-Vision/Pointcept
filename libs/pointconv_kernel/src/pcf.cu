//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
//

#include "pcf.h"
#include "knn.h"
#include "pcf_ops.h"
#include "pconv_ops.h"

namespace pcf {

torch::Tensor pconv_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features)
{
    CHECK_INPUT(input);
    CHECK_INPUT(neighbor_inds);
    CHECK_INPUT(weights);
    CHECK_INPUT(additional_features);

    return pconv_ops::pconv_cuda_forward(input, neighbor_inds, weights, additional_features);
}

std::vector<torch::Tensor> pconv_linear_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor linear_bias)
{
    CHECK_INPUT(input);
    CHECK_INPUT(neighbor_inds);
    CHECK_INPUT(weights);
    CHECK_INPUT(additional_features);
    CHECK_INPUT(linear_weights);
    CHECK_INPUT(linear_bias);

    return pconv_ops::pconv_linear_cuda_forward(input, neighbor_inds, weights, additional_features, linear_weights, linear_bias);
}

std::vector<torch::Tensor> pconv_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features)
{
    CHECK_INPUT(grad_output);
    CHECK_INPUT(neighbor_inds);
    CHECK_INPUT(weights);
    CHECK_INPUT(additional_features);
    return pconv_ops::pconv_cuda_backward(grad_output, input, neighbor_inds, weights, additional_features);
}

std::vector<torch::Tensor> pconv_linear_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor pconv_output)
{
    CHECK_INPUT(grad_output);
    CHECK_INPUT(neighbor_inds);
    CHECK_INPUT(weights);
    CHECK_INPUT(additional_features);
    CHECK_INPUT(linear_weights);
    CHECK_INPUT(pconv_output);
    return pconv_ops::pconv_linear_cuda_backward(grad_output, input, neighbor_inds, weights, additional_features,
                linear_weights, pconv_output);
}

std::vector<torch::Tensor> compute_knn_inverse(
    torch::Tensor neighbor_inds,
    const int total_points)
{
    CHECK_CUDA(neighbor_inds);
    CHECK_CONTIGUOUS(neighbor_inds);

    return knn::knn_inverse_cuda_forward(neighbor_inds, total_points);
}

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
    torch::Tensor pconv_output)
{
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(inverse_neighbor);
    CHECK_INPUT(inverse_neighbor_k);
    CHECK_INPUT(inverse_neighbor_idx);
    CHECK_INPUT(neighbor_inds);
    CHECK_INPUT(weights);
    CHECK_INPUT(additional_features);
    CHECK_INPUT(linear_weights);
    CHECK_INPUT(pconv_output);

    return pconv_ops::pconv_linear_opt_cuda_backward(
        grad_output, input, inverse_neighbor, inverse_neighbor_k,
        inverse_neighbor_idx, neighbor_inds, weights, additional_features,
        linear_weights, pconv_output);
}

std::vector<torch::Tensor> pconv_linear_cutlass(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor linear_bias)
{
    CHECK_INPUT(input);
    CHECK_INPUT(neighbor_inds);
    CHECK_INPUT(weights);
    CHECK_INPUT(additional_features);
    CHECK_INPUT(linear_weights);
    CHECK_INPUT(linear_bias);

    return pconv_ops::pconv_linear_cutlass_forward(input, neighbor_inds, weights, additional_features, linear_weights, linear_bias);
}

} // namespace pcf
