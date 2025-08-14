//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
//
#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace pcf {

/**
 * @brief Calculate the next power of 2 for a given integer
 * 
 * @param n Input integer
 * @return Next power of 2 greater than or equal to n
 */
__host__ __device__ inline int nextPowerOf2(int n) {
    n--;           // Decrement n to handle the case when n is already a power of 2
    n |= n >> 1;   // Set all bits after the highest set bit to 1
    n |= n >> 2;   // Continue setting bits
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;  // Add 1 to get the next power of 2
}

} // namespace pcf
