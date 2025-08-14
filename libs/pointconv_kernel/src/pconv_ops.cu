//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
//

#include "common.h"
#include "pconv_ops.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/half.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/gemm/device/gemm.h>
#include <c10/cuda/CUDAStream.h>

using namespace nvcuda;

namespace pcf {
namespace pconv_ops {

/** 
 * input: B x M x C_in tensor, B = batch size, M = number of points in the original point cloud, C_in = number of channels, input features
 * neighbor_inds: B x N x K tensor, K = neighborhood size, indices of the neighborhood of each point
 * weights: B x N x K x C_mid, C_mid = number of middle channels
 * additional_features: B x N x K x C_add, additional features that do not require indexing
 * output: B x N x (C_mid*C_in), final output of the PCF layer
 * 
 * This implements a fused layer of:
 *      concat(sum_{i} input[b,n][neighbor_inds[i]][j] * weights[b,n,k,i], sum_{i} additional_features[b,n][i][j] * weights[b,n,k,i]
 * 
 * It outputs a tensor of shape B x N x (C_mid * (C_in + C_add)
 * It avoids serializing the input hence faster than naive pyTorch implementation
 */
template <typename scalar_t>
__global__ void pconv_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ input,
    const torch::PackedTensorAccessor32<long, 3, torch::RestrictPtrTraits> __restrict__ neighbor_inds,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ weights,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ additional_features,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ output)
{

    extern __shared__ unsigned char memory[];
    scalar_t* shared_mem = reinterpret_cast<scalar_t*>(memory);

    const int B = input.size(0);
    const int Nout = neighbor_inds.size(1);
    const int C_in = input.size(2);
    const int K = neighbor_inds.size(2);
    const int C_mid = weights.size(3);
    const int C_add = additional_features.size(3);

    // Each thread block handles one point
    const int batch_idx = blockIdx.x / Nout;
    const int point_idx = blockIdx.x % Nout;
    const int tid = threadIdx.x;

    // Shared memory
    scalar_t* shared_input = shared_mem;
    scalar_t* shared_additional = shared_input + K * C_in;

    // Load input features
    for (int k = 0; k < K; k++) {
            const int n_idx = neighbor_inds[batch_idx][point_idx][k];
            for (int c = tid; c < C_in; c += blockDim.x) {
                    shared_input[k * C_in + c] = input[batch_idx][n_idx][c];
            }
    }

    // Load additional features
    if (C_add > 0) {
            for (int k = 0; k < K; k++) {
                    for (int c = tid; c < C_add; c += blockDim.x) {
                            shared_additional[k * C_add + c] = additional_features[batch_idx][point_idx][k][c];
                    }
            }
    }

    __syncthreads();

    // PConv
    const int total_channels = C_mid * (C_in + C_add);
    for (int c = tid; c < total_channels; c += blockDim.x) {
            const int mid_idx = c % C_mid;
            const int in_idx = c / C_mid;

            scalar_t sum = 0;
            for (int k = 0; k < K; k++) {
                    if (in_idx < C_in) {
                            sum += shared_input[k * C_in + in_idx] * weights[batch_idx][point_idx][k][mid_idx];
                    } else {
                            sum += shared_additional[k * C_add + (in_idx - C_in)] * weights[batch_idx][point_idx][k][mid_idx];
                    }
            }

            output[batch_idx][point_idx][c] = sum;
    }
}

/**
 * This kernel implements a fused Point Convolution (PConv) and Linear layer.
 * 
 * Input Tensors:
 * - input: [B x M x C_in] Input features for each point
 *      B = batch size, M = number of points, C_in = input channels
 * - neighbor_inds: [B x N x K] Indices of K neighbors for each point
 *      N = number of output points, K = neighborhood size
 * - weights: [B x N x K x C_mid] Weight matrix for point convolution
 *      C_mid = middle/intermediate channels
 * - additional_features: [B x N x K x C_add] Additional point features
 *      C_add = number of additional feature channels
 * - linear_weights: [C_out x (C_mid*(C_in + C_add))] Linear layer weights
 * - linear_bias: [C_out] Linear layer bias
 * 
 * Output:
 * - final_output: [B x N x C_out] Final output after PConv and Linear layer
 * 
 * Steps:
 * 1. PConv: For each point, aggregate features from its neighbors using weights
 * 2. Concatenate results from input features and additional features
 * 3. Apply linear transformation to the concatenated features
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
        torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ pconv_output)
{
    extern __shared__ unsigned char memory[];
    scalar_t* shared_mem = reinterpret_cast<scalar_t*>(memory);

    const int K = neighbor_inds.size(2);
    const int C_in = input.size(2);
    const int C_mid = weights.size(3);
    const int C_add = additional_features.size(3);
    const int C_out = linear_weights.size(0);

    const int Nout = neighbor_inds.size(1);
    const int total_blocks = gridDim.x;
    const int iter = blockIdx.x;
    const int batch_idx = iter / Nout;              // Current batch
    const int point_idx = iter % Nout;              // Current point in point cloud
    const int tid = threadIdx.x;                    // Thread ID within block
    const int warp_id = tid / 32;                   // Warp ID (32 threads per warp)
    const int lane_id = tid % 32;                   // Lane ID within the warp

    // Shared memory
    scalar_t* shared_input = shared_mem;
    scalar_t* shared_additional = shared_input + (K * C_in);
    scalar_t* shared_weights = shared_additional + (K * C_add);
    scalar_t* shared_intermediate = shared_weights + (K * C_mid);   // For PConv output

    // Load input features
    #pragma unroll
    for (int i = tid; i < K * C_in; i += blockDim.x) {
            const int k = i / C_in;         // Neighbor index
            const int c = i % C_in;         // Channel index
            const int n_idx = neighbor_inds[batch_idx][point_idx][k];
            shared_input[i] = input[batch_idx][n_idx][c];
    }

    // Load additional features
    #pragma unroll
    for (int i = tid; i < K * C_add; i += blockDim.x) {
            const int k = i / C_add;
            const int c = i % C_add;
            shared_additional[i] = additional_features[batch_idx][point_idx][k][c];
    }

    // Load weights
    #pragma unroll
    for (int i = tid; i < K * C_mid; i += blockDim.x) {
            const int k = i / C_mid;
            const int c = i % C_mid;
            shared_weights[i] = weights[batch_idx][point_idx][k][c];
    }

    __syncthreads();

    // PConv -> parallelize across output channels
    const int total_channels = C_mid * (C_in + C_add);      // Total channels after concat

    #pragma unroll
    for (int c = tid; c < total_channels; c += blockDim.x) {
            const int mid_idx = c % C_mid;
            const int in_idx = c / C_mid;

            scalar_t sum = 0;
            #pragma unroll
            for (int k = 0; k < K; k++) {
                    if (in_idx < C_in) {
                            sum += shared_input[k * C_in + in_idx] * shared_weights[k * C_mid + mid_idx];
                    } else {
                            sum += shared_additional[k * C_add + (in_idx - C_in)] * shared_weights[k * C_mid + mid_idx];
                    }
            }

            shared_intermediate[c] = sum;
            pconv_output[batch_idx][point_idx][c] = sum;  // Store PConv output
    }

    __syncthreads();

    // Linear layer -> each thread handles one output channel
    if (tid < C_out) {
            scalar_t sum = 0;

            #pragma unroll
            for (int c = 0; c < total_channels; c++) {
                    sum += shared_intermediate[c] * linear_weights[tid][c];
            }

            final_output[batch_idx][point_idx][tid] = sum + linear_bias[tid];
    }
}

/**
 * grad_output: B x N x (C_mid * C_in), the gradient derived from above
 * input: B x N x C_in tensor, B = batch size, N = number of points, C_in = number of channels, input features
 * neighbor_inds: B x N x K tensor, K = neighborhood size, indices of the neighborhood of each point
 * weights: B x N x K x C_mid, C_mid = number of middle channels
 * additional_features: B x N x K x C_add, additional features that do not need gather
 * grad_input: same shape as input, gradient of input
 * grad_weights: same shape as weights, gradient of weights
 *
 * Forward is:
 *      sum_{i} input[b,n][neighbor_inds[i]][j] * weights[b,n,k,i]
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
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ grad_additional)
{

    int i,j,k, ii, kk,iter0;
    const int B = input.size(0);
    const int N = input.size(1);
    const int Nout = neighbor_inds.size(1);
    const int C_in = input.size(2);
    const int K = neighbor_inds.size(2);
    const int C_mid = weights.size(3);
	const int C_add = additional_features.size(3);
    const int increment = blockDim.x / C_mid;
    const int cur_mid = threadIdx.x / increment;
    if (cur_mid >= C_mid) return;

    // Supposedly blockIdx.x should go up to B * N
    for (iter0 = blockIdx.x; iter0< B * Nout; iter0+= gridDim.x) {
            ii = iter0 / Nout;  // ii is the index on the batch dimension
            i = iter0 % Nout;   // i is the index on the point dimension
            k = threadIdx.x % K;
            scalar_t weight_grad_temp = 0.0;
            scalar_t cur_compute;

            #pragma unroll
            for (kk=0;kk<C_in;kk++) {
                    long cur_channel = cur_mid * (C_in + C_add) + kk;
                    cur_compute = grad_output[ii][i][cur_channel] * weights[ii][i][k][cur_mid];
                    weight_grad_temp += grad_output[ii][i][cur_channel] * input[ii][neighbor_inds[ii][i][k]][kk];

                    // It would be quite expensive to store this in shared memory so use this naive approach for now, using atomicAdd to avoid racing conditions
                    atomicAdd(&grad_input[ii][neighbor_inds[ii][i][k]][kk], cur_compute);
            }

            for (kk=0;kk<C_add;kk++) {
                    long cur_channel = cur_mid * (C_in + C_add) + kk + C_in;
                    cur_compute = grad_output[ii][i][cur_channel] * weights[ii][i][k][cur_mid];
                    weight_grad_temp += grad_output[ii][i][cur_channel] * additional_features[ii][i][k][kk];
                    grad_additional[ii][i][k][kk] = cur_compute;
            }

            grad_weights[ii][i][k][cur_mid] = weight_grad_temp;
	}
}

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
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> __restrict__ grad_linear_bias)
{

    extern __shared__ unsigned char memory[];
    scalar_t* shared_grad = reinterpret_cast<scalar_t*>(memory);

    const int B = input.size(0);
    const int Nout = neighbor_inds.size(1);
    const int C_in = input.size(2);
    const int K = neighbor_inds.size(2);
    const int C_mid = weights.size(3);
    const int C_add = additional_features.size(3);
    const int C_out = grad_output.size(2);
    const int total_channels = C_mid * (C_in + C_add);

    const int total_blocks = gridDim.x;
    const int iter = blockIdx.x;
    const int batch_idx = iter / Nout;
    const int point_idx = iter % Nout;
    const int tid = threadIdx.x;
    const int increment = blockDim.x / C_mid;

    // grad_x = grad_output * W^T for intermediate gradient
    for (int c = tid; c < total_channels; c += blockDim.x) {
            scalar_t sum = 0;
            for (int c_out = 0; c_out < C_out; c_out++) {
                    sum += grad_output[batch_idx][point_idx][c_out] * linear_weights[c_out][c];
            }
            shared_grad[c] = sum;
    }

    __syncthreads();

    // additional features gradient - each thread processes one (k, c_mid, c_add) combination
    if (tid < K * C_mid * C_add) {
            const int k = tid % K;
            const int mid_idx = (tid / K) % C_mid;
            const int add_idx = tid / (K * C_mid);

            const int grad_idx = mid_idx * (C_in + C_add) + C_in + add_idx;
            grad_additional[batch_idx][point_idx][k][add_idx] = 
            shared_grad[grad_idx] * weights[batch_idx][point_idx][k][mid_idx];
    }

    __syncthreads();

    // input features gradient and weight gradient
    if (tid < K * C_mid) {
            const int k = tid % K;
            const int c_mid = tid / K;
            scalar_t weight_grad = 0;

            // Input features gradient
            for (int c = 0; c < C_in; c++) {
                    const int grad_idx = c_mid * (C_in + C_add) + c;
                    const scalar_t grad_val = shared_grad[grad_idx];
                    atomicAdd(&grad_input[batch_idx][neighbor_inds[batch_idx][point_idx][k]][c],
                            grad_val * weights[batch_idx][point_idx][k][c_mid]);
                    weight_grad += grad_val * input[batch_idx][neighbor_inds[batch_idx][point_idx][k]][c];
            }

            // Additional features contribution to weight gradient
            for (int c = 0; c < C_add; c++) {
                    const int grad_idx = c_mid * (C_in + C_add) + C_in + c;
                    weight_grad += shared_grad[grad_idx] * additional_features[batch_idx][point_idx][k][c];
            }

            grad_weights[batch_idx][point_idx][k][c_mid] = weight_grad;
    }

    // Linear layer gradients
    if (tid < C_out) {
            scalar_t bias_grad = grad_output[batch_idx][point_idx][tid];

            // Linear weights gradient
            for (int c = 0; c < total_channels; c++) {
                    atomicAdd(&grad_linear_weights[tid][c],
                            bias_grad * pconv_output[batch_idx][point_idx][c]);
            }

            // Linear bias gradient
            atomicAdd(&grad_linear_bias[tid], bias_grad);
    }
}

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
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> grad_linear_bias) 
{
    extern __shared__ unsigned char shared_memory[];
    scalar_t* shared_grad_intermediate = reinterpret_cast<scalar_t*>(shared_memory);

    const int B = input.size(0);
    const int N = input.size(1);
    const int Nout = neighbor_inds.size(1);
    const int C_in = input.size(2);
    const int K = neighbor_inds.size(2);
    const int C_mid = weights.size(3);
    const int C_add = additional_features.size(3);
    const int C_out = grad_output.size(2);
    const int total_channels = C_mid * (C_in + C_add);

    const int total_blocks = gridDim.x;
    const int iter = blockIdx.x;
    const int batch_idx = iter / Nout;
    const int point_idx = iter % Nout;
    const int tid = threadIdx.x;

    if (batch_idx >= B || point_idx >= Nout) return;

    // Process output points
    if (tid < C_out) {
            atomicAdd(&grad_linear_bias[tid], grad_output[batch_idx][point_idx][tid]);
    }

    for (int c = tid; c < total_channels; c += blockDim.x) {
            scalar_t sum = 0;
            for (int co = 0; co < C_out; co++) {
                    sum += grad_output[batch_idx][point_idx][co] * linear_weights[co][c];
            }
            shared_grad_intermediate[c] = sum;
    }

    __syncthreads();

    // Process weight gradients
    for (int idx = tid; idx < K * C_mid; idx += blockDim.x) {
            const int k = idx % K;
            const int mid_idx = idx / K;
            scalar_t weight_grad = 0;

            const long n_idx_long = neighbor_inds[batch_idx][point_idx][k];
            const int32_t n_idx = static_cast<int32_t>(n_idx_long);

            if (n_idx >= 0 && n_idx < N && n_idx < static_cast<int32_t>(input.size(1))) {
                    for (int c_in = 0; c_in < C_in; c_in++) {
                            const int grad_idx = mid_idx * (C_in + C_add) + c_in;
                            if (grad_idx < total_channels) {
                                    weight_grad += shared_grad_intermediate[grad_idx] * input[batch_idx][n_idx][c_in];
                            }
                    }
            }

            // Additional features contribution
            for (int c_add = 0; c_add < C_add; c_add++) {
                    const int grad_idx = mid_idx * (C_in + C_add) + C_in + c_add;
                    if (grad_idx < total_channels) {
                            weight_grad += shared_grad_intermediate[grad_idx] * 
                                        additional_features[batch_idx][point_idx][k][c_add];
                    }
            }

            grad_weights[batch_idx][point_idx][k][mid_idx] = weight_grad;
    }

    // Process additional feature gradients
    for (int idx = tid; idx < K * C_add; idx += blockDim.x) {
            const int k = idx % K;
            const int c_add = idx / K;
            scalar_t add_grad = 0;

            for (int mid_idx = 0; mid_idx < C_mid; mid_idx++) {
                    const int grad_idx = mid_idx * (C_in + C_add) + C_in + c_add;
                    if (grad_idx < total_channels) {
                            add_grad += shared_grad_intermediate[grad_idx] * weights[batch_idx][point_idx][k][mid_idx];
                    }
            }

            grad_additional[batch_idx][point_idx][k][c_add] = add_grad;
    }

    for (int k = 0; k < K; k++) {
            const long n_idx_long = neighbor_inds[batch_idx][point_idx][k];
            const int32_t n_idx = static_cast<int32_t>(n_idx_long);

            if (n_idx >= 0 && 
                n_idx < N && 
                n_idx < static_cast<int32_t>(input.size(1)) && 
                n_idx < static_cast<int32_t>(grad_input.size(1))) {

                    for (int c_in = tid; c_in < C_in; c_in += blockDim.x) {
                            scalar_t input_grad = 0;

                            for (int mid_idx = 0; mid_idx < C_mid; mid_idx++) {
                                    const int grad_idx = mid_idx * (C_in + C_add) + c_in;
                                    if (grad_idx < total_channels) {
                                            input_grad += shared_grad_intermediate[grad_idx] * weights[batch_idx][point_idx][k][mid_idx];
                                    }
                            }

                            if (c_in < C_in && c_in < static_cast<int>(grad_input.size(2))) {
                                    atomicAdd(&grad_input[batch_idx][n_idx][c_in], input_grad);
                            }
                    }
            }
    }

    // Linear weights gradient computation
    const int linear_chunk_size = (total_channels + blockDim.x - 1) / blockDim.x;
    const int start_idx = tid * linear_chunk_size;
    const int end_idx = min(start_idx + linear_chunk_size, total_channels);

    for (int c_out = 0; c_out < C_out; c_out++) {
            const scalar_t grad_out = grad_output[batch_idx][point_idx][c_out];

            for (int c = start_idx; c < end_idx; c++) {
                    if (c < total_channels && 
                            c_out < C_out && 
                            c < static_cast<int>(grad_linear_weights.size(1)) &&
                            c_out < static_cast<int>(grad_linear_weights.size(0))) {
                            atomicAdd(&grad_linear_weights[c_out][c], 
                                        grad_out * pconv_output[batch_idx][point_idx][c]);
                    }
            }
    }

    // For input-only points, we move to a separate kernel to reduce branch divergence and simplify control flow
}

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
    const int N, const int Nout, const int C_in, const int C_mid, const int C_add, const int C_out, const int input_only_points)
{
    const int iter = blockIdx.x;
    const int batch_idx = iter / input_only_points;
    const int local_point_idx = iter % input_only_points;
    const int point_idx = Nout + local_point_idx;
    const int tid = threadIdx.x;

    if (batch_idx >= static_cast<int>(grad_output.size(0))) return;
    if (point_idx >= N) return;
    if (point_idx >= static_cast<int>(inverse_neighbor_idx.size(1))) return;
    if (local_point_idx >= input_only_points) return;

    const int start_idx = inverse_neighbor_idx[batch_idx][point_idx];
    if (start_idx < 0 || start_idx == -1) return;  // No neighbors reference this point

    int end_idx;
    if (point_idx < N - 1 && point_idx + 1 < static_cast<int>(inverse_neighbor_idx.size(1))) {
            int next_idx = point_idx + 1;
            while (next_idx < N && 
                    next_idx < static_cast<int>(inverse_neighbor_idx.size(1)) && 
                    inverse_neighbor_idx[batch_idx][next_idx] == -1) {
                    next_idx++;
            }
            end_idx = (next_idx < N && next_idx < static_cast<int>(inverse_neighbor_idx.size(1))) ? 
                        inverse_neighbor_idx[batch_idx][next_idx] : static_cast<int>(inverse_neighbor.size(1));
    } else {
            end_idx = static_cast<int>(inverse_neighbor.size(1));
    }

    end_idx = min(end_idx, static_cast<int>(inverse_neighbor.size(1)));

    for (int c_in = tid; c_in < C_in; c_in += blockDim.x) {
            scalar_t grad_sum = 0;

            for (int inv_idx = start_idx; inv_idx < end_idx; inv_idx++) {
                    if (inv_idx >= static_cast<int>(inverse_neighbor.size(1))) break;

                    const int n_out = inverse_neighbor[batch_idx][inv_idx];
                    const int k_idx = inverse_neighbor_k[batch_idx][inv_idx];

                    if (n_out < 0 || n_out >= Nout) continue;
                    if (k_idx < 0 || k_idx >= static_cast<int>(weights.size(2))) continue;
                    if (n_out >= static_cast<int>(weights.size(1))) continue;

                    for (int mid_idx = 0; mid_idx < C_mid; mid_idx++) {
                            const int pconv_idx = mid_idx * (C_in + C_add) + c_in;

                            if (pconv_idx >= static_cast<int>(linear_weights.size(1))) continue;

                            scalar_t grad_through_linear = 0;

                            for (int c_out_idx = 0; c_out_idx < C_out; c_out_idx++) {
                                    if (c_out_idx >= static_cast<int>(grad_output.size(2))) break;
                                    if (n_out >= static_cast<int>(grad_output.size(1))) break;

                                    grad_through_linear += grad_output[batch_idx][n_out][c_out_idx] * 
                                                            linear_weights[c_out_idx][pconv_idx];
                            }

                            if (mid_idx < static_cast<int>(weights.size(3))) {
                                    grad_sum += grad_through_linear * weights[batch_idx][n_out][k_idx][mid_idx];
                            }
                    }
            }

            if (c_in < static_cast<int>(grad_input.size(2)) && 
                    point_idx < static_cast<int>(grad_input.size(1))) {
                    grad_input[batch_idx][point_idx][c_in] = grad_sum;
            }
    }
}

__global__ void gather_kernel(
    const float* input,
    const float* additional_features,
    const int64_t* neighbor_inds,
    float* concatenated_output,
    int B, int M, int Nout, int K, int C_in, int C_add)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * Nout * K * (C_in + C_add);
    if (idx >= total) return;

    const int b = idx / (Nout * K * (C_in + C_add));
    int rem = idx % (Nout * K * (C_in + C_add));
    const int n = rem / (K * (C_in + C_add));
    rem %= (K * (C_in + C_add));
    const int k = rem / (C_in + C_add);
    const int c = rem % (C_in + C_add);

    const int64_t neighbor_idx = neighbor_inds[b * Nout * K + n * K + k];
    if (c < C_in) {
            concatenated_output[idx] = input[b * M * C_in + neighbor_idx * C_in + c];
    } else {
            int c_add = c - C_in;
            concatenated_output[idx] = additional_features[b * Nout * K * C_add + n * K * C_add + k * C_add + c_add];
    }
}

torch::Tensor pconv_cuda_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features)
{
    const int B = input.size(0);
    const int N = input.size(1);
    const int Nout = neighbor_inds.size(1);
    const int C_in = input.size(2);
    const int C_add = additional_features.size(3);
    const int C_mid =  weights.size(3);
    const int numBlocks = B * Nout;
    const int numThreads = C_mid * C_in > 256 ? 256 : C_mid * C_in;

    const int K = neighbor_inds.size(2);
    const int shared_mem_size = (K * (C_in + C_add)) * sizeof(float);   // shared memory size for input and additional features

    auto output = torch::zeros({B, Nout, C_mid * (C_in + C_add)}, input.type());

    AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "pconv_cuda_forward_kernel", ([&] {
    pconv_cuda_forward_kernel<scalar_t><<<numBlocks, numThreads, shared_mem_size>>>(
                input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                neighbor_inds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
                weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                additional_features.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
    }));

    return output;
}

// This function does a fused Point Convolution (PConv) followed by a Linear layer.
//
// 1. PConv Operation:
//    - For each output point, gather input features from K neighbors using indices from neighbor_inds
//    - Concatenate these gathered features with additional per-point features (additional_features)
//    - Apply learned weights to compute intermediate features (C_mid channels)
//
// 2. Linear Layer:
//    - Take the PConv output (C_mid*(C_in+C_add) channels)
//    - Apply linear transformation to get final output (C_out channels)
//
std::vector<torch::Tensor> pconv_linear_cuda_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor linear_bias)
{
    const int B = input.size(0);
    const int Nout = neighbor_inds.size(1);
    const int C_in = input.size(2);
    const int K = neighbor_inds.size(2);
    const int C_mid = weights.size(3);
    const int C_add = additional_features.size(3);
    const int C_out = linear_weights.size(0);

    auto pconv_output = torch::zeros({B, Nout, C_mid * (C_in + C_add)}, input.options());
    auto final_output = torch::zeros({B, Nout, C_out}, input.options());

    const int total_work = std::max({K * C_in, K * C_add, K * C_mid, C_out});
    const int thread_count = std::min(256, nextPowerOf2(total_work));

    const int total_blocks = B * Nout;
    dim3 grid(total_blocks);

    const int shared_mem_size = (
                                        (K * C_in) +                    // shared_input
                                        (K * C_add) +                   // shared_additional  
                                        (K * C_mid) +                   // shared_weights
                                        (C_mid * (C_in + C_add))        // shared_intermediate
                                ) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "pconv_linear_cuda_forward_kernel", ([&] {
                pconv_linear_cuda_forward_kernel<scalar_t><<<grid, thread_count, shared_mem_size>>>(
                input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                neighbor_inds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
                weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                additional_features.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                linear_weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                linear_bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                final_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                pconv_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
    }));

    return {final_output, pconv_output};
}

// Computes gradients for the Point Convolution (PConv) layer using a custom CUDA kernel.
// Each thread processes a specific (k, c_mid) pair and:
// - Computes partial input gradients via atomicAdd
// - Computes per-neighbor additional feature gradients
// - Accumulates weight gradients from input and additional contributions
//
std::vector<torch::Tensor> pconv_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features)
{
    const int B = input.size(0);
    const int N = input.size(1);
    const int Nout = neighbor_inds.size(1);
    const int C_in = input.size(2);
    const int C_mid =  weights.size(3);
    const int K = neighbor_inds.size(2);
    const int numBlocks = B * Nout;
    const int numThreads = C_mid * K;
    auto grad_input = torch::zeros_like(input);
    auto grad_weights = torch::zeros_like(weights);
    auto grad_additional = torch::zeros_like(additional_features);

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "pconv_cuda_backward_kernel", ([&] {
                pconv_cuda_backward_kernel<scalar_t><<<numBlocks,numThreads>>>(
                grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                neighbor_inds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
                weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                additional_features.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                grad_input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                grad_weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                grad_additional.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
    }));

    return {grad_input, grad_weights, grad_additional};
}

// This function computes gradients for a fused Point Convolution (PConv) + Linear layer
// using a single optimized CUDA kernel pconv_linear_cuda_backward_kernel.
//
// 1. Compute Intermediate Gradient (grad_x)
// - Multiply grad_output by linear_weights^T to backpropagate through the linear layer
// - Store result in shared memory to reuse in downstream computations
//
// 2. Additional Features Gradient
// - Use grad_x and PConv weights to compute gradients w.r.t. additional features
// - Write directly to grad_additional
//
// 3. Input Features and PConv Weights Gradients
// - Compute input gradients by multiplying grad_x with weights and scattering to input positions
// - Compute PConv weight gradients using both input and additional_features
//
// 4. Linear Layer Gradients
// - For each output channel, computes weight gradients using grad_output and pconv_output
// - Compute bias gradients by summing grad_output
//
std::vector<torch::Tensor> pconv_linear_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor pconv_output)
{
    const int B = input.size(0);
    const int Nout = neighbor_inds.size(1);
    const int C_in = input.size(2);
    const int K = neighbor_inds.size(2);
    const int C_mid = weights.size(3);
    const int C_add = additional_features.size(3);
    const int C_out = grad_output.size(2);

    auto grad_input = torch::zeros_like(input);
    auto grad_weights = torch::zeros_like(weights);
    auto grad_additional = torch::zeros_like(additional_features);
    auto grad_linear_weights = torch::zeros_like(linear_weights);
    auto grad_linear_bias = torch::zeros({C_out}, input.options());

    const int total_work = std::max({K * C_in, K * C_add, K * C_mid, C_out});
    const int thread_count = std::min(256, nextPowerOf2(total_work));

    const int total_blocks = B * Nout;
    dim3 grid(total_blocks);

    const int shared_mem_size = (C_mid * (C_in + C_add)) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "pconv_linear_cuda_backward_kernel", ([&] {
    pconv_linear_cuda_backward_kernel<scalar_t><<<grid, thread_count, shared_mem_size>>>(
                grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                neighbor_inds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
                weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                additional_features.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                linear_weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                pconv_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                grad_input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                grad_weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                grad_additional.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                grad_linear_weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                grad_linear_bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
    }));

    return {grad_input, grad_weights, grad_additional, grad_linear_weights, grad_linear_bias};
}

// This function computes gradients for a fused Point Convolution (PConv) + Linear layer
// using two optimized CUDA kernels: pconv_linear_fused_cuda_backward_kernel_opt (for output points) and
// input_only_backward_kernel (for input-only points).
//
// 1. pconv_linear_fused_cuda_backward_kernel_opt:
//    - Cache intermediate gradients to reduce global memory accesses
//    - Gradient Computation:
//      - Compute PConv weight and additional feature gradients using neighbor indices
//      - Aggregate input gradients
//      - Compute linear layer gradients in parallel across output channels
//
// 2. input_only_backward_kernel:
//    - Handle input points referenced by neighbors but not processed in forward pass
//    - Use precomputed indices to access referencing output points
//    - We use dedicated kernel for input-only points to prevent branch divergence
//
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
    torch::Tensor pconv_output)
{
    const int B = input.size(0);
    const int N = input.size(1);
    const int Nout = neighbor_inds.size(1);
    const int C_in = input.size(2);
    const int K = neighbor_inds.size(2);
    const int C_mid = weights.size(3);
    const int C_add = additional_features.size(3);
    const int C_out = grad_output.size(2);

//     TORCH_CHECK(N >= Nout, "Input points N must be >= output points Nout");
    TORCH_CHECK(inverse_neighbor_idx.size(1) >= N, "inverse_neighbor_idx size must be >= N");

    TORCH_CHECK(neighbor_inds.max().item<long>() < N, 
                "neighbor_inds contains out-of-bounds indices: max=" + 
                std::to_string(neighbor_inds.max().item<long>()) + ", N=" + std::to_string(N));

    auto grad_input = torch::zeros_like(input);
    auto grad_weights = torch::zeros_like(weights);
    auto grad_additional = torch::zeros_like(additional_features);
    auto grad_linear_weights = torch::zeros_like(linear_weights);
    auto grad_linear_bias = torch::zeros({C_out}, input.options());

    const int max_threads_per_block = 512;
    const int total_work = std::max({C_in, K * C_mid, K * C_add, C_out});
    const int thread_count = std::min(max_threads_per_block, nextPowerOf2(total_work));

    const int shared_mem_size = (C_mid * (C_in + C_add)) * input.element_size();

    // Main kernel for output points (0 to Nout-1)
    {
            const int total_blocks_main = B * Nout;
            dim3 grid(total_blocks_main);
            AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "pconv_linear_fused_cuda_backward_kernel_opt", ([&] {
            pconv_linear_fused_cuda_backward_kernel_opt<scalar_t><<<grid, thread_count, shared_mem_size>>>(
                            grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                            input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                            inverse_neighbor.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                            inverse_neighbor_k.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
                            inverse_neighbor_idx.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                            neighbor_inds.packed_accessor32<long, 3, torch::RestrictPtrTraits>(),
                            weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                            additional_features.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                            linear_weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                            pconv_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                            grad_input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                            grad_weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                            grad_additional.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                            grad_linear_weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                            grad_linear_bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
            }));
    }

    // We launch a separate kernel for input-only points (Nout to N-1)
    int input_only_points = N - Nout;
    if (input_only_points > 0) {
            const int total_blocks_input = B * input_only_points;
            dim3 grid(total_blocks_input);

            AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "input_only_backward_kernel", ([&] {
            input_only_backward_kernel<scalar_t><<<grid, thread_count>>>(
                            grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                            input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                            inverse_neighbor.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                            inverse_neighbor_k.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
                            inverse_neighbor_idx.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                            weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                            linear_weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                            grad_input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                            N, Nout, C_in, C_mid, C_add, C_out, input_only_points);
            }));
    }

    return {grad_input, grad_weights, grad_additional, grad_linear_weights, grad_linear_bias};
}

// This function performs the forward pass for a fused Point Convolution (PConv) + Linear layer using CUTLASS-based GEMM kernels.
//
// 1. Gather Phase (custom CUDA gather_kernel):
// - For each output point, gather features of its K neighbors from both input and additional_features
// - Concatenate to form a [B, Nout, K, C_concat] tensor
//
// 2. PConv (Batch GEMM using CUTLASS):
// - Each output point performs a batched GEMM: [K, C_concat] × [K, C_mid] → [C_concat, C_mid]
// - Implemented via CUTLASS GemmBatched
//
// 3. Linear Projection (GEMM using CUTLASS):
// - Flatten PConv output per point into [C_concat * C_mid]
// - Apply linear transformation with CUTLASS Gemm kernel to compute final output
// - Add bias
//
// 4. Batching Strategy:
// - If Nout is large, input is split into manageable batches to reduce memory usage
// - Both PConv and Linear GEMM are executed in batches accordingly
//
std::vector<torch::Tensor> pconv_linear_cutlass_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor linear_bias)
{
    const int B = input.size(0);
    const int M = input.size(1);
    const int Nout = neighbor_inds.size(1);
    const int C_in = input.size(2);
    const int K = neighbor_inds.size(2);
    const int C_mid = weights.size(3);
    const int C_add = additional_features.size(3);
    const int C_out = linear_weights.size(0);
    const int C_concat = C_in + C_add;

    const int batch_size = 10000;
    const bool use_batched = Nout > batch_size;

    auto final_output = torch::zeros({B, Nout, C_out}, input.options());
    auto pconv_output = torch::zeros({B, Nout, C_concat * C_mid}, input.options());

    if (use_batched) {
            const int num_batches = (Nout + batch_size - 1) / batch_size;

            for (int batch = 0; batch < num_batches; batch++) {
                    const int start_idx = batch * batch_size;
                    const int end_idx = std::min(start_idx + batch_size, Nout);
                    const int batch_points = end_idx - start_idx;

                    if (batch_points <= 0) continue;

                    auto neighbor_inds_batch = neighbor_inds.slice(1, start_idx, end_idx);
                    auto additional_features_batch = additional_features.slice(1, start_idx, end_idx);
                    auto weights_batch = weights.slice(1, start_idx, end_idx);

                    auto concatenated_batch = torch::zeros({B, batch_points, K, C_concat}, input.options());

                    const int threads = 256;
                    const int blocks = (B * batch_points * K * C_concat + threads - 1) / threads;

                    gather_kernel<<<blocks, threads>>>(
                                    input.data_ptr<float>(),
                                    additional_features_batch.data_ptr<float>(),
                                    neighbor_inds_batch.data_ptr<int64_t>(),
                                    concatenated_batch.data_ptr<float>(),
                                    B, M, batch_points, K, C_in, C_add
                    );

                    auto features = concatenated_batch.view({B * batch_points, K, C_concat});
                    auto weights_reshaped = weights_batch.view({B * batch_points, K, C_mid}).contiguous();

                    auto pconv_result = torch::zeros({B * batch_points, C_concat, C_mid}, input.options());

                    using LayoutA = cutlass::layout::ColumnMajor;
                    using LayoutB = cutlass::layout::RowMajor;
                    using LayoutC = cutlass::layout::RowMajor;

                    using Gemm = cutlass::gemm::device::GemmBatched<
                                    float, LayoutA,
                                    float, LayoutB,
                                    float, LayoutC,
                                    float,
                                    cutlass::arch::OpClassSimt,
                                    cutlass::arch::Sm90,
                                    cutlass::gemm::GemmShape<64, 64, 8>,
                                    cutlass::gemm::GemmShape<32, 32, 8>,
                                    cutlass::gemm::GemmShape<1, 1, 1>
                    >;

                    Gemm gemm_op;
                    cutlass::Status status;

                    const int batch_count = B * batch_points;
                    const cutlass::gemm::GemmCoord problem_size(C_concat, C_mid, K);

                    cutlass::TensorRef<float const, LayoutA> A_ref(features.data_ptr<float>(), C_concat);
                    cutlass::TensorRef<float const, LayoutB> B_ref(weights_reshaped.data_ptr<float>(), C_mid);
                    cutlass::TensorRef<float, LayoutC> C_ref(pconv_result.data_ptr<float>(), C_mid);

                    const int64_t stride_A = K * C_concat;
                    const int64_t stride_B = K * C_mid;
                    const int64_t stride_C = C_concat * C_mid;

                    typename Gemm::EpilogueOutputOp::Params epilogue_op(1.0f, 0.0f);

                    typename Gemm::Arguments args(
                                    problem_size,
                                    A_ref, stride_A,
                                    B_ref, stride_B,
                                    C_ref, stride_C,
                                    C_ref, stride_C,
                                    epilogue_op,
                                    batch_count
                    );

                    const size_t workspace_size = Gemm::get_workspace_size(args);
                    auto workspace = torch::empty({static_cast<int64_t>(workspace_size)}, input.options().dtype(torch::kUInt8));

                    status = gemm_op.initialize(args, workspace.data_ptr());
                    if (status != cutlass::Status::kSuccess) {
                            throw std::runtime_error("CUTLASS GEMM initialization failed");
                    }

                    status = gemm_op();
                    if (status != cutlass::Status::kSuccess) {
                            throw std::runtime_error("CUTLASS GEMM execution failed");
                    }

                    // shape [B*batch_points, C_concat, C_mid]
                    auto pconv_batch_output = pconv_result.view({B, batch_points, C_concat * C_mid});

                    // Copy slice
                    pconv_output.slice(1, start_idx, end_idx).copy_(pconv_batch_output);

                    int K_linear = C_concat * C_mid;
                    auto X_mat = pconv_batch_output.view({B * batch_points, K_linear});
                    auto W_t = linear_weights.t().contiguous();

                    auto Y_batch = torch::zeros({B * batch_points, C_out}, input.options());

                    using GemmLinear = cutlass::gemm::device::Gemm<
                                    float, cutlass::layout::RowMajor,
                                    float, cutlass::layout::RowMajor,
                                    float, cutlass::layout::RowMajor,
                                    float,
                                    cutlass::arch::OpClassSimt,
                                    cutlass::arch::Sm90,
                                    cutlass::gemm::GemmShape<64, 64, 8>,
                                    cutlass::gemm::GemmShape<32, 32, 8>,
                                    cutlass::gemm::GemmShape<1, 1, 1>
                    >;

                    GemmLinear gemm_linear;
                    cutlass::gemm::GemmCoord problem_size_linear(B * batch_points, C_out, K_linear);

                    cutlass::TensorRef<float const, cutlass::layout::RowMajor> X_ref(X_mat.data_ptr<float>(), K_linear);
                    cutlass::TensorRef<float const, cutlass::layout::RowMajor> W_ref_linear(W_t.data_ptr<float>(), C_out);
                    cutlass::TensorRef<float, cutlass::layout::RowMajor> Y_ref(Y_batch.data_ptr<float>(), C_out);

                    typename GemmLinear::EpilogueOutputOp::Params epilogue_op_linear(1.0f, 0.0f);
                    typename GemmLinear::Arguments args_linear(
                                    problem_size_linear,
                                    X_ref,
                                    W_ref_linear,
                                    Y_ref,
                                    Y_ref,
                                    epilogue_op_linear
                    );

                    size_t workspace_size_linear = GemmLinear::get_workspace_size(args_linear);
                    auto workspace_linear = torch::empty({static_cast<int64_t>(workspace_size_linear)}, input.options().dtype(torch::kUInt8));

                    auto status_linear = gemm_linear.initialize(args_linear, workspace_linear.data_ptr());
                    if (status_linear != cutlass::Status::kSuccess) {
                            throw std::runtime_error("CUTLASS GEMM (Linear) initialization failed");
                    }
                    status_linear = gemm_linear();
                    if (status_linear != cutlass::Status::kSuccess) {
                            throw std::runtime_error("CUTLASS GEMM (Linear) execution failed");
                    }

                    Y_batch = Y_batch + linear_bias.view({1, C_out});

                    // Copy slice
                    final_output.slice(1, start_idx, end_idx).copy_(Y_batch.view({B, batch_points, C_out}));
            }
    } else {
            // For small point clouds, we process all at once
            auto concatenated = torch::zeros({B, Nout, K, C_concat}, input.options());

            const int threads = 256;
            const int blocks = (B * Nout * K * C_concat + threads - 1) / threads;

            gather_kernel<<<blocks, threads>>>(
                            input.data_ptr<float>(),
                            additional_features.data_ptr<float>(),
                            neighbor_inds.data_ptr<int64_t>(),
                            concatenated.data_ptr<float>(),
                            B, M, Nout, K, C_in, C_add
            );

            auto features = concatenated.view({B * Nout, K, C_concat});
            auto weights_reshaped = weights.view({B * Nout, K, C_mid}).contiguous();

            auto pconv_result = torch::zeros({B * Nout, C_concat, C_mid}, input.options());

            using LayoutA = cutlass::layout::ColumnMajor;
            using LayoutB = cutlass::layout::RowMajor;
            using LayoutC = cutlass::layout::RowMajor;

            using Gemm = cutlass::gemm::device::GemmBatched<
                            float, LayoutA,
                            float, LayoutB,
                            float, LayoutC,
                            float,
                            cutlass::arch::OpClassSimt,
                            cutlass::arch::Sm90,
                            cutlass::gemm::GemmShape<64, 64, 8>,
                            cutlass::gemm::GemmShape<32, 32, 8>,
                            cutlass::gemm::GemmShape<1, 1, 1>
            >;

            Gemm gemm_op;
            cutlass::Status status;

            const int batch_count = B * Nout;
            const cutlass::gemm::GemmCoord problem_size(C_concat, C_mid, K);

            cutlass::TensorRef<float const, LayoutA> A_ref(features.data_ptr<float>(), C_concat);
            cutlass::TensorRef<float const, LayoutB> B_ref(weights_reshaped.data_ptr<float>(), C_mid);
            cutlass::TensorRef<float, LayoutC> C_ref(pconv_result.data_ptr<float>(), C_mid);

            const int64_t stride_A = K * C_concat;
            const int64_t stride_B = K * C_mid;
            const int64_t stride_C = C_concat * C_mid;

            typename Gemm::EpilogueOutputOp::Params epilogue_op(1.0f, 0.0f);

            typename Gemm::Arguments args(
                            problem_size,
                            A_ref, stride_A,
                            B_ref, stride_B,
                            C_ref, stride_C,
                            C_ref, stride_C,
                            epilogue_op,
                            batch_count
            );

            const size_t workspace_size = Gemm::get_workspace_size(args);
            auto workspace = torch::empty({static_cast<int64_t>(workspace_size)}, input.options().dtype(torch::kUInt8));

            status = gemm_op.initialize(args, workspace.data_ptr());
            if (status != cutlass::Status::kSuccess) {
                    throw std::runtime_error("CUTLASS GEMM initialization failed");
            }

            status = gemm_op();
            if (status != cutlass::Status::kSuccess) {
                    throw std::runtime_error("CUTLASS GEMM execution failed");
            }

            // shape [B*Nout, C_concat, C_mid]
            pconv_output = pconv_result.view({B, Nout, C_concat * C_mid});

            int K_linear = C_concat * C_mid;
            auto X_mat = pconv_output.view({B * Nout, K_linear});
            auto W_t = linear_weights.t().contiguous();

            auto Y = torch::zeros({B * Nout, C_out}, input.options());

            using GemmLinear = cutlass::gemm::device::Gemm<
                            float, cutlass::layout::RowMajor,
                            float, cutlass::layout::RowMajor,
                            float, cutlass::layout::RowMajor,
                            float,
                            cutlass::arch::OpClassSimt,
                            cutlass::arch::Sm90,
                            cutlass::gemm::GemmShape<64, 64, 8>,
                            cutlass::gemm::GemmShape<32, 32, 8>,
                            cutlass::gemm::GemmShape<1, 1, 1>
            >;

            GemmLinear gemm_linear;
            cutlass::gemm::GemmCoord problem_size_linear(B * Nout, C_out, K_linear);

            cutlass::TensorRef<float const, cutlass::layout::RowMajor> X_ref(X_mat.data_ptr<float>(), K_linear);
            cutlass::TensorRef<float const, cutlass::layout::RowMajor> W_ref_linear(W_t.data_ptr<float>(), C_out);
            cutlass::TensorRef<float, cutlass::layout::RowMajor> Y_ref(Y.data_ptr<float>(), C_out);

            typename GemmLinear::EpilogueOutputOp::Params epilogue_op_linear(1.0f, 0.0f);
            typename GemmLinear::Arguments args_linear(
                            problem_size_linear,
                            X_ref,
                            W_ref_linear,
                            Y_ref,
                            Y_ref,
                            epilogue_op_linear
            );

            size_t workspace_size_linear = GemmLinear::get_workspace_size(args_linear);
            auto workspace_linear = torch::empty({static_cast<int64_t>(workspace_size_linear)}, input.options().dtype(torch::kUInt8));

            auto status_linear = gemm_linear.initialize(args_linear, workspace_linear.data_ptr());
            if (status_linear != cutlass::Status::kSuccess) {
                    throw std::runtime_error("CUTLASS GEMM (Linear) initialization failed");
            }
            status_linear = gemm_linear();
            if (status_linear != cutlass::Status::kSuccess) {
                    throw std::runtime_error("CUTLASS GEMM (Linear) execution failed");
            }

            Y = Y + linear_bias.view({1, C_out});

            final_output = Y.view({B, Nout, C_out});
    }

    return {final_output, pconv_output};
}

} // namespace pconv_ops
} // namespace pcf
