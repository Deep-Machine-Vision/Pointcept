import gc
import time
import numpy as np
import torch
import pcf_cuda
from pykeops.torch import LazyTensor

from test_kernels import (
    knn, 
    PConv, 
    PConvLinear, 
    PConvLinearOpt,
    PConvLinearCutlass,
    create_inverse_python
)


def memory_usage_stats():
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)    # MB
    return allocated, reserved

def print_memory_usage(title, before, after):
    alloc_diff = after[0] - before[0]
    res_diff = after[1] - before[1]
    print(f"{title}: Allocated +{alloc_diff:.2f} MB, Reserved +{res_diff:.2f} MB")

def profile_knn_inverse_memory(n_points=100000, K=64):
    """
    Profile memory usage of KNN inverse mapping kernel
    """
    device = torch.device("cuda")
    B = 1  # batch size

    print(f"\n===== Memory profiling for KNN inverse with {n_points} points, K={K} =====")

    # Reset
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    # Create random pc
    torch.manual_seed(42)
    before = memory_usage_stats()
    input_points = torch.randn(B, n_points, 3, device=device, requires_grad=False)
    after = memory_usage_stats()
    print_memory_usage("Point cloud creation", before, after)

    # Compute KNN indices
    print("\nComputing KNN indices...")
    before = memory_usage_stats()
    neighbor_inds = knn(input_points, input_points, K).contiguous()
    after = memory_usage_stats()
    print_memory_usage("KNN computation", before, after)
    total_points = n_points

    # Compute KNN inverse
    print("\nComputing KNN inverse mapping...")
    before = memory_usage_stats()
    peak_before = torch.cuda.max_memory_allocated() / (1024 * 1024)

    outputs = pcf_cuda.compute_knn_inverse(neighbor_inds, total_points)
    torch.cuda.synchronize()

    after = memory_usage_stats()
    peak_after = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print_memory_usage("KNN inverse computation", before, after)
    print(f"Peak memory during KNN inverse: {peak_after:.2f} MB (+{peak_after - peak_before:.2f} MB)")

    print("\nMemory usage of output tensors:")
    for i, tensor in enumerate(outputs):
        print(f"  Output tensor {i}: {tensor.element_size() * tensor.numel() / (1024 * 1024):.2f} MB, shape: {tensor.shape}")
    
    return outputs, neighbor_inds, total_points


def profile_pconv_memory(n_points=100000, K=64):
    """
    Profile memory usage of PConv implementations
    """
    device = torch.device("cuda")
    B = 1  # batch size

    # Params
    C_in = 16       # input feature channels
    C_add = 16      # additional feature channels
    C_mid = 16      # mid feature channels - weight dim
    C_out = 64      # output channels for linear layer

    print(f"\n===== Memory profiling for PConv with {n_points} points, K={K} =====")

    # Reset
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    print("Creating input tensors...")
    torch.manual_seed(42)

    before = memory_usage_stats()

    input_points = torch.randn(B, n_points, 3, device=device, requires_grad=True)
    input_features = torch.randn(B, n_points, C_in, device=device, requires_grad=True)
    neighbor_inds = knn(input_points, input_points, K).contiguous()
    inverse_neighbors, inverse_k, inverse_idx = pcf_cuda.compute_knn_inverse(neighbor_inds, input_features.shape[1])
    weights = torch.randn(B, n_points, K, C_mid, device=device, requires_grad=True)
    additional_features = torch.randn(B, n_points, K, C_add, device=device, requires_grad=True)
    linear_weights = torch.randn(C_out, (C_in + C_add) * C_mid, device=device, requires_grad=True)
    linear_bias = torch.randn(C_out, device=device, requires_grad=True)

    after = memory_usage_stats()
    print_memory_usage("Input tensor creation total", before, after)

    print("\nIndividual tensor sizes:")
    print(f"  Input features: {input_features.element_size() * input_features.numel() / (1024 * 1024):.2f} MB, shape: {input_features.shape}")
    print(f"  Neighbor indices: {neighbor_inds.element_size() * neighbor_inds.numel() / (1024 * 1024):.2f} MB, shape: {neighbor_inds.shape}")
    print(f"  Inverse neighbors: {inverse_neighbors.element_size() * inverse_neighbors.numel() / (1024 * 1024):.2f} MB, shape: {inverse_neighbors.shape}")
    print(f"  Weights: {weights.element_size() * weights.numel() / (1024 * 1024):.2f} MB, shape: {weights.shape}")
    print(f"  Additional features: {additional_features.element_size() * additional_features.numel() / (1024 * 1024):.2f} MB, shape: {additional_features.shape}")

    # Init
    pconv = PConv()
    linear_layer = torch.nn.Linear((C_in + C_add) * C_mid, C_out).to(device)
    linear_layer.weight.data.copy_(linear_weights)
    linear_layer.bias.data.copy_(linear_bias)

    pconv_linear = PConvLinear((C_in + C_add) * C_mid, C_out)
    pconv_linear.linear.weight.data.copy_(linear_weights)
    pconv_linear.linear.bias.data.copy_(linear_bias)
    pconv_linear = pconv_linear.cuda()

    pconv_linear_opt = PConvLinearOpt((C_in + C_add) * C_mid, C_out)
    pconv_linear_opt.linear.weight.data.copy_(linear_weights)
    pconv_linear_opt.linear.bias.data.copy_(linear_bias)
    pconv_linear_opt = pconv_linear_opt.cuda()

    pconv_linear_cutlass = PConvLinearCutlass((C_in + C_add) * C_mid, C_out)
    pconv_linear_cutlass.linear.weight.data.copy_(linear_weights)
    pconv_linear_cutlass.linear.bias.data.copy_(linear_bias)
    pconv_linear_cutlass = pconv_linear_cutlass.cuda()

    # Dummy gradient
    grad_output = torch.randn(B, n_points, C_out, device=device)

    implementations = [
        {
            "name": "Unfused (PConv + Linear)",
            "forward": lambda: (pconv(input_features, neighbor_inds, weights, additional_features), linear_layer),
            "backward": lambda out_tuple: linear_layer(out_tuple[0]).backward(grad_output)
        },
        {
            "name": "Fused (PConvLinear)",
            "forward": lambda: pconv_linear(input_features, neighbor_inds, weights, additional_features),
            "backward": lambda out: out.backward(grad_output)
        },
        {
            "name": "Optimized Fused (PConvLinearOpt)",
            "forward": lambda: pconv_linear_opt(input_features, neighbor_inds, inverse_neighbors, inverse_k, inverse_idx, weights, additional_features),
            "backward": lambda out: out.backward(grad_output)
        },
        {
            "name": "Cutlass (PConvLinearCutlass)",
            "forward": lambda: pconv_linear_cutlass(input_features, neighbor_inds, inverse_neighbors, inverse_k, inverse_idx, weights, additional_features),
            "backward": lambda out: out.backward(grad_output)
        }
    ]

    # Profile
    for impl in implementations:
        print(f"\n----- {impl['name']} -----")
    
        # Reset
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        input_features.grad = None
        weights.grad = None
        additional_features.grad = None
        if "linear_layer" in locals():
            linear_layer.weight.grad = None
            linear_layer.bias.grad = None
        if "pconv_linear" in locals():
            pconv_linear.linear.weight.grad = None
            pconv_linear.linear.bias.grad = None
        if "pconv_linear_opt" in locals():
            pconv_linear_opt.linear.weight.grad = None
            pconv_linear_opt.linear.bias.grad = None
        if "pconv_linear_cutlass" in locals():
            pconv_linear_cutlass.linear.weight.grad = None
            pconv_linear_cutlass.linear.bias.grad = None

        # Forward pass
        before_forward = memory_usage_stats()
        peak_before_forward = torch.cuda.max_memory_allocated() / (1024 * 1024)

        torch.cuda.nvtx.range_push(f"{impl['name']}_FORWARD")
        out = impl["forward"]()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        after_forward = memory_usage_stats()
        peak_after_forward = torch.cuda.max_memory_allocated() / (1024 * 1024)

        print_memory_usage("Forward pass", before_forward, after_forward)
        print(f"Peak memory during forward: {peak_after_forward:.2f} MB (+{peak_after_forward - peak_before_forward:.2f} MB)")

        # Backward pass
        before_backward = memory_usage_stats()
        peak_before_backward = torch.cuda.max_memory_allocated() / (1024 * 1024)

        torch.cuda.nvtx.range_push(f"{impl['name']}_BACKWARD")
        impl["backward"](out)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        after_backward = memory_usage_stats()
        peak_after_backward = torch.cuda.max_memory_allocated() / (1024 * 1024)

        print_memory_usage("Backward pass", before_backward, after_backward)
        print(f"Peak memory during backward: {peak_after_backward:.2f} MB (+{peak_after_backward - peak_before_backward:.2f} MB)")
        print(f"Total peak memory: {peak_after_backward:.2f} MB")

        del out


def main():
    import sys
    if len(sys.argv) > 1:
        n_points = int(sys.argv[1])
    else:
        n_points = 100000

    if len(sys.argv) > 2:
        K = int(sys.argv[2])
    else:
        K = 64

    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Configuration: {n_points} points, K={K}")

    profile_knn_inverse_memory(n_points, K)
    profile_pconv_memory(n_points, K)

    print("\nMemory profiling completed successfully")


if __name__ == "__main__":
    main()