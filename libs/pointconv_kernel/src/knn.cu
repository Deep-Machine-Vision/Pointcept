//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
//

#include "knn.h"

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

namespace pcf {
namespace knn {

__global__ void count_neighbors_kernel(
    const torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> neighbor_inds,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> counts,
    const int total_points,
    const int start_point,
    const int batch_idx)
{
    const int local_point_idx = blockIdx.y;
    const int point_idx = start_point + local_point_idx;
    const int tid = threadIdx.x;
    const int K = neighbor_inds.size(2);

    for (int k = tid; k < K; k += blockDim.x) {
            const int64_t neighbor = neighbor_inds[batch_idx][point_idx][k];
            if (neighbor >= 0 && neighbor < total_points) {
                    atomicAdd(&counts[static_cast<int32_t>(neighbor)], 1);
            }
    }
}

__global__ void compute_inv_idx_kernel(
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> counts,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> inv_idx,
    const int total_points,
    const int batch_idx)
{
    int32_t sum = 0;
    inv_idx[batch_idx][0] = 0;
    for (int i = 0; i < total_points; i++) {
            sum += counts[i];
            inv_idx[batch_idx][i + 1] = sum;
    }
}

__global__ void fill_inverse_kernel(
    const torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> neighbor_inds,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> inv_neighbors,
    torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> inv_k,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> running_counts,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> inv_idx,
    const int total_points,
    const int start_point,
    const int batch_idx)
{
    const int local_point_idx = blockIdx.y;
    const int point_idx = start_point + local_point_idx;
    const int tid = threadIdx.x;
    const int K = neighbor_inds.size(2);

    for (int k = tid; k < K; k += blockDim.x) {
            const int64_t neighbor = neighbor_inds[batch_idx][point_idx][k];
            if (neighbor >= 0 && neighbor < total_points) {
                    const int32_t pos = atomicAdd(&running_counts[static_cast<int32_t>(neighbor)], 1);
                    const int32_t idx = inv_idx[batch_idx][static_cast<int32_t>(neighbor)] + pos;

                    if (idx < inv_neighbors.size(1)) {
                            inv_neighbors[batch_idx][idx] = static_cast<int32_t>(point_idx);
                            inv_k[batch_idx][idx] = static_cast<uint8_t>(k);
                    }
            }
    }
}

/**
 * This function computes inverse neighborhood relationships for KNN indices:
 *  inv_neighbors: List of points that reference each target point (B x total_references)
 *  inv_k: Corresponding k-index in original neighbor_inds tensor (B x total_references)
 *  inv_idx: Prefix sum indicating start/end positions in inv_neighbors per point (B x (total_points+1))
 *
 * 1. count_neighbors_kernel:
 *      - Build histogram of how many times each point is referenced as a neighbor
 *      - Process points in segments of 'points_this_grid' to handle large point clouds
 *
 * 2. compute_inv_idx_kernel:
 *      - Convert counts to prefix sum for indexing into inv_neighbors
 *
 * 3. fill_inverse_kernel:
 *      - Populate inverse mappings using precomputed indices
 *      - Reuse segmented processing from count_neighbors_kernel
 */
std::vector<torch::Tensor> knn_inverse_cuda_forward(
    torch::Tensor neighbor_inds,
    const int total_points)
{
    const int B = neighbor_inds.size(0);
    const int N = neighbor_inds.size(1);
    const int K = neighbor_inds.size(2);

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(neighbor_inds.device());
    auto inv_neighbors = torch::zeros({B, N * K}, options);
    auto inv_k = torch::zeros({B, N * K}, options.dtype(torch::kUInt8));
    auto inv_idx = torch::zeros({B, total_points + 1}, options);

    const int threads_per_block = std::min(512, K);
    const int points_per_grid = 65535;
    const int num_y_grids = (N + points_per_grid - 1) / points_per_grid;

    for (int b = 0; b < B; b++) {
            auto counts = torch::zeros({total_points}, options);

            for (int grid_idx = 0; grid_idx < num_y_grids; grid_idx++) {
                    const int start_point = grid_idx * points_per_grid;
                    const int points_this_grid = std::min(points_per_grid, N - start_point);
                    const dim3 block(threads_per_block);
                    const dim3 grid(1, points_this_grid);

                    count_neighbors_kernel<<<grid, block>>>(
                            neighbor_inds.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
                            counts.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                            total_points,
                            start_point,
                            b
                    );
            }

            compute_inv_idx_kernel<<<1, 1>>>(
                    counts.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                    inv_idx.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                    total_points,
                    b
            );

            counts.zero_();

            for (int grid_idx = 0; grid_idx < num_y_grids; grid_idx++) {
                    const int start_point = grid_idx * points_per_grid;
                    const int points_this_grid = std::min(points_per_grid, N - start_point);
                    const dim3 block(threads_per_block);
                    const dim3 grid(1, points_this_grid);

                    fill_inverse_kernel<<<grid, block>>>(
                            neighbor_inds.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
                            inv_neighbors.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                            inv_k.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
                            counts.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                            inv_idx.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                            total_points,
                            start_point,
                            b
                    );
            }
    }

    return {inv_neighbors, inv_k, inv_idx};
}

} // namespace knn
} // namespace pcf
