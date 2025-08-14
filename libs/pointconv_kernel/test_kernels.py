import gc
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import pcf_cuda
import torch.cuda.profiler as profiler
from pykeops.torch import LazyTensor


class PConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_feat, neighbor_inds, weightnet, additional_features):
        neighbor_inds.requires_grad = False
        output = pcf_cuda.pconv_forward(input_feat, neighbor_inds, weightnet, additional_features)
        ctx.save_for_backward(input_feat, neighbor_inds, weightnet, additional_features)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_additional = pcf_cuda.pconv_backward(
            grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, None, grad_weight, grad_additional

class PConvLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_feat, neighbor_inds, weightnet, additional_features, linear_weights, linear_bias):
        neighbor_inds.requires_grad = False
        output, pconv_output = pcf_cuda.pconv_linear_forward(input_feat, neighbor_inds, weightnet, additional_features, 
                                             linear_weights, linear_bias)
        
        ctx.save_for_backward(input_feat, neighbor_inds, weightnet, additional_features, 
                            linear_weights, pconv_output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        input_feat, neighbor_inds, weightnet, additional_features, linear_weights, pconv_output = saved
        
        grads = pcf_cuda.pconv_linear_backward(
            grad_output.contiguous(), input_feat, neighbor_inds, weightnet,
            additional_features, linear_weights, pconv_output)
        
        return grads[0], None, grads[1], grads[2], grads[3], grads[4]

class PConv(torch.nn.Module):
    def __init__(self):
        super(PConv, self).__init__()

    @staticmethod
    def forward(input_features, neighbor_inds, weightnet, additional_features=None):
        if additional_features is None:
            additional_features = torch.zeros(input_features.shape[0], input_features.shape[1], 
                                           neighbor_inds.shape[2], 0, device=input_features.device)
        return PConvFunction.apply(input_features, neighbor_inds, weightnet, additional_features)

class PConvLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(PConvLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, input_features, neighbor_inds, weightnet, additional_features=None):
        if additional_features is None:
            additional_features = torch.zeros(input_features.shape[0], input_features.shape[1], 
                                           neighbor_inds.shape[2], 0, device=input_features.device)
        return PConvLinearFunction.apply(input_features, neighbor_inds, weightnet, additional_features,
                                       self.linear.weight, self.linear.bias)

class PConvLinearOptFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_feat, neighbor_inds, inverse_neighbors, inverse_k, inverse_idx,
                        weightnet, additional_features, linear_weights, linear_bias):
        neighbor_inds.requires_grad = False

        output, pconv_output = pcf_cuda.pconv_linear_forward(
            input_feat, neighbor_inds, weightnet, additional_features, 
            linear_weights, linear_bias)

        ctx.save_for_backward(input_feat, inverse_neighbors, inverse_k, inverse_idx, 
                            neighbor_inds, weightnet, additional_features, 
                            linear_weights, pconv_output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        input_feat, inverse_neighbors, inverse_k, inverse_idx, neighbor_inds, \
        weightnet, additional_features, linear_weights, pconv_output = saved

        grad_output = grad_output.contiguous()

        grads = pcf_cuda.pconv_linear_opt_backward(
            grad_output, input_feat, inverse_neighbors, inverse_k, 
            inverse_idx, neighbor_inds, weightnet, additional_features,
            linear_weights, pconv_output)

        return grads[0], None, None, None, None, grads[1], grads[2], grads[3], grads[4]

class PConvLinearOpt(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(PConvLinearOpt, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, input_features, neighbor_inds, inverse_neighbors, inverse_k, inverse_idx, weightnet, additional_features=None):
        if additional_features is None:
            additional_features = torch.zeros(input_features.shape[0], input_features.shape[1], 
                                          neighbor_inds.shape[2], 0, device=input_features.device)
        return PConvLinearOptFunction.apply(input_features, neighbor_inds, inverse_neighbors, inverse_k, inverse_idx,
                                                weightnet, additional_features, self.linear.weight, self.linear.bias)


class PConvLinearCutlassFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_feat, neighbor_inds, inverse_neighbors, inverse_k, inverse_idx,
                        weightnet, additional_features, linear_weights, linear_bias):
        neighbor_inds.requires_grad = False

        output, pconv_output = pcf_cuda.pconv_linear_cutlass_forward(
            input_feat, neighbor_inds, weightnet, additional_features, 
            linear_weights, linear_bias)

        ctx.save_for_backward(input_feat, inverse_neighbors, inverse_k, inverse_idx, 
                            neighbor_inds, weightnet, additional_features, 
                            linear_weights, pconv_output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        input_feat, inverse_neighbors, inverse_k, inverse_idx, neighbor_inds, \
        weightnet, additional_features, linear_weights, pconv_output = saved

        grad_output = grad_output.contiguous()

        grads = pcf_cuda.pconv_linear_opt_backward(
            grad_output, input_feat, inverse_neighbors, inverse_k, 
            inverse_idx, neighbor_inds, weightnet, additional_features,
            linear_weights, pconv_output)

        return grads[0], None, None, None, None, grads[1], grads[2], grads[3], grads[4]

class PConvLinearCutlass(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(PConvLinearCutlass, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, input_features, neighbor_inds, inverse_neighbors, inverse_k, inverse_idx, weightnet, additional_features=None):
        if additional_features is None:
            additional_features = torch.zeros(input_features.shape[0], input_features.shape[1], 
                                          neighbor_inds.shape[2], 0, device=input_features.device)
        return PConvLinearCutlassFunction.apply(input_features, neighbor_inds, inverse_neighbors, inverse_k, inverse_idx,
                                                weightnet, additional_features, self.linear.weight, self.linear.bias)


def knn(pts1, pts2, k):
    """
    Find the k-nearest neighbors for each point in pts1 from pts2

    Args:
        pts1: (B, S, C) tensor of query points
        pts2: (B, N, C) tensor of reference points
        k: int, number of neighbors to find

    Returns:
        (B, S, k) tensor of neighbor indices
    """
    B, S, C = pts1.shape
    _, N, _ = pts2.shape
    x_i = LazyTensor(pts1.view(B, S, 1, C))
    y_j = LazyTensor(pts2.view(B, 1, N, C))
    D_ij = ((x_i - y_j)**2).sum(-1)**0.5
    _, indices_i = D_ij.Kmin_argKmin(k, dim=2)
    return indices_i.long()


def create_inverse_python(neighbor_mat, total_points):
        """
        Create inverse mapping for KNN

        Args:
            neighbor_mat: shape [num_points, K] containing neighbor indices
            total_points: Total number of points in the point cloud (including potential points to add)
        """

        K = neighbor_mat.shape[1]
        neigh = [[] for n in range(total_points)]
        inv_k = [[] for n in range(total_points)]

        # Neighbor lists
        for r in range(neighbor_mat.shape[0]):
            for c in range(K):
                neigh[neighbor_mat[r,c]].append(r)
                inv_k[neighbor_mat[r,c]].append(c)

        # Get index array
        idx = [len(x) for x in neigh]
        cum_sum = 0
        idx_array = []
        for i in range(total_points):
            if idx[i] == 0:
                idx_array.append(cum_sum)
            else:
                idx_array.append(cum_sum)
                cum_sum = idx[i] + cum_sum
        idx_array.append(neighbor_mat.shape[0] * K)  # total size of inverse arrays

        neighbors = np.hstack([np.array(x) for x in neigh if len(x) > 0]).flatten()
        inv_k = np.hstack([np.array(x) for x in inv_k if len(x) > 0]).flatten()

        return (torch.from_numpy(neighbors).cuda().long(), 
                torch.from_numpy(inv_k).cuda().long(), 
                torch.from_numpy(np.array(idx_array)).cuda().long())


def test_pconv_linear():
    path = "/nfs/stak/users/sivakuml/hpc-memory/cutlass/data/500000pts"
    device = torch.device("cuda")
    
    feats_x = torch.load(f"{path}/feat_x.pt").to(device).contiguous()
    nei_inds = torch.load(f"{path}/nei_inds.pt").to(device).contiguous()
    weights = torch.load(f"{path}/weights.pt").to(device).contiguous()
    feat_pe = torch.load(f"{path}/feat_pe.pt").to(device).contiguous()
    linear_weights = torch.load(f"{path}/linear_weights.pt").to(device).contiguous()
    linear_bias = torch.load(f"{path}/linear_bias.pt").to(device).contiguous()

    # Params
    out_channel = 64
    last_ch = 16
    weightnet_dim = 16

    pconv = PConv()
    linear_layer = torch.nn.Linear((out_channel // 4 + last_ch) * weightnet_dim, out_channel // 2).to(device)
    linear_layer.weight.data.copy_(linear_weights)
    linear_layer.bias.data.copy_(linear_bias)

    pconv_linear = PConvLinear((out_channel // 4 + last_ch) * weightnet_dim, out_channel // 2)
    pconv_linear.linear.weight.data.copy_(linear_weights)
    pconv_linear.linear.bias.data.copy_(linear_bias)
    pconv_linear = pconv_linear.cuda()

    # Create dummy gradient for backward pass
    grad_output = torch.randn_like(linear_layer(torch.empty(feats_x.shape[0], feats_x.shape[1], 
                                              (out_channel // 4 + last_ch) * weightnet_dim, device=device)))

    # Forward & Backward (Unfused)
    pconv_out = pconv(feats_x, nei_inds, weights, feat_pe)
    out_unfused = linear_layer(pconv_out)
    out_unfused.backward(grad_output)

    # Store unfused gradients
    unfused_grads = {
        "input": feats_x.grad.clone(),
        "weightnet": weights.grad.clone(),
        "additional": feat_pe.grad.clone(),
        "linear_weight": linear_layer.weight.grad.clone(),
        "linear_bias": linear_layer.bias.grad.clone()
    }

    # Clear gradients
    feats_x.grad = None
    weights.grad = None
    feat_pe.grad = None
    linear_layer.weight.grad = None
    linear_layer.bias.grad = None

    # Forward & Backward (Fused)
    out_fused = pconv_linear(feats_x, nei_inds, weights, feat_pe)
    out_fused.backward(grad_output)

    torch.cuda.synchronize()

    # Store fused gradients
    fused_grads = {
        "input": feats_x.grad.clone(),
        "weightnet": weights.grad.clone(),
        "additional": feat_pe.grad.clone(),
        "linear_weight": pconv_linear.linear.weight.grad.clone(),
        "linear_bias": pconv_linear.linear.bias.grad.clone()
    }
    
    # Compare outputs
    print("\nForward output max difference:", 
          torch.max(torch.abs(out_unfused - out_fused)).item())
    
    # Compare gradients
    print("\nGradient differences (max absolute error):")
    for key in unfused_grads:
        diff = torch.max(torch.abs(unfused_grads[key] - fused_grads[key])).item()
        print(f"{key:13} grad diff: {diff:.6f}")
        print(f"{key:13} stats:")
        print(f"  Unfused - min: {unfused_grads[key].min().item():.6f}, max: {unfused_grads[key].max().item():.6f}, mean: {unfused_grads[key].mean().item():.6f}")
        print(f"  Fused   - min: {fused_grads[key].min().item():.6f}, max: {fused_grads[key].max().item():.6f}, mean: {fused_grads[key].mean().item():.6f}")
        print()
    
    # Timing comparison
    num_runs = 100
    torch.cuda.synchronize()
    
    # Time unfused
    total_time = 0
    for _ in range(num_runs):
        pconv_out = pconv(feats_x, nei_inds, weights, feat_pe)
        out_unfused = linear_layer(pconv_out)
        
        feats_x.grad = None
        weights.grad = None
        feat_pe.grad = None
        linear_layer.weight.grad = None
        linear_layer.bias.grad = None
        
        torch.cuda.synchronize()
        start = time.time()
        out_unfused.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
        total_time += (time.time() - start)
    print(f"\n(Unfused) Average Time: {(total_time / num_runs) * 1000} ms")
    
    # Time fused
    total_time = 0
    for _ in range(num_runs):
        out_fused = pconv_linear(feats_x, nei_inds, weights, feat_pe)
        
        feats_x.grad = None
        weights.grad = None
        feat_pe.grad = None
        pconv_linear.linear.weight.grad = None
        pconv_linear.linear.bias.grad = None
        
        torch.cuda.synchronize()
        start = time.time()
        out_fused.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
        total_time += (time.time() - start)
    print(f"(Fused) Average Time: {(total_time / num_runs) * 1000} ms")


def test_pconv_linear_with_memory(num_runs=100):
    path = "/nfs/stak/users/sivakuml/hpc-memory/cutlass/data/500000pts"
    device = torch.device("cuda")

    # Load data
    feats_x = torch.load(f"{path}/feat_x.pt").to(device).contiguous()
    nei_inds = torch.load(f"{path}/nei_inds.pt").to(device).contiguous()
    weights = torch.load(f"{path}/weights.pt").to(device).contiguous()
    feat_pe = torch.load(f"{path}/feat_pe.pt").to(device).contiguous()
    linear_weights = torch.load(f"{path}/linear_weights.pt").to(device).contiguous()
    linear_bias = torch.load(f"{path}/linear_bias.pt").to(device).contiguous()

    # Params
    out_channel = 64
    last_ch = 16
    weightnet_dim = 16

    unfused_stats = {
        "after_pconv": [],
        "after_linear": [],
        "peak_memory": []
    }

    fused_stats = {
        "after_forward": [],
        "peak_memory": []
    }

    # Initialize models
    pconv = PConv()
    linear_layer = torch.nn.Linear((out_channel // 4 + last_ch) * weightnet_dim, out_channel // 2).to(device)
    linear_layer.weight.data.copy_(linear_weights)
    linear_layer.bias.data.copy_(linear_bias)

    pconv_linear = PConvLinear((out_channel // 4 + last_ch) * weightnet_dim, out_channel // 2)
    pconv_linear.linear.weight.data.copy_(linear_weights)
    pconv_linear.linear.bias.data.copy_(linear_bias)
    pconv_linear = pconv_linear.cuda()

    # Dummy gradient for backward pass
    grad_output = torch.randn_like(linear_layer(torch.empty(feats_x.shape[0], feats_x.shape[1], 
                                              (out_channel // 4 + last_ch) * weightnet_dim, device=device)))

    print(f"\nRunning memory test for {num_runs} iterations...")

    for i in range(num_runs):
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1} runs...")

        # Test Unfused Version
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        # Forward Pass (Unfused)
        pconv_out = pconv(feats_x, nei_inds, weights, feat_pe)
        torch.cuda.synchronize()
        unfused_stats["after_pconv"].append(torch.cuda.memory_allocated() - mem_before)

        out_unfused = linear_layer(pconv_out)
        torch.cuda.synchronize()
        unfused_stats["after_linear"].append(torch.cuda.memory_allocated() - mem_before)

        # Backward Pass (Unfused)
        out_unfused.backward(grad_output)
        torch.cuda.synchronize()
        unfused_stats["peak_memory"].append(torch.cuda.max_memory_allocated())

        # Clear memory
        del pconv_out, out_unfused
        feats_x.grad = None
        weights.grad = None
        feat_pe.grad = None
        linear_layer.weight.grad = None
        linear_layer.bias.grad = None
        torch.cuda.empty_cache()

        # Test Fused Version
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        # Forward & Backward (Fused)
        out_fused = pconv_linear(feats_x, nei_inds, weights, feat_pe)
        torch.cuda.synchronize()
        fused_stats["after_forward"].append(torch.cuda.memory_allocated() - mem_before)

        out_fused.backward(grad_output)
        torch.cuda.synchronize()
        fused_stats["peak_memory"].append(torch.cuda.max_memory_allocated())

        # Clear memory
        del out_fused
        feats_x.grad = None
        weights.grad = None
        feat_pe.grad = None
        pconv_linear.linear.weight.grad = None
        pconv_linear.linear.bias.grad = None
        torch.cuda.empty_cache()

    def calculate_stats(values):
        values = torch.tensor(values, dtype=torch.float64)
        return {
            "mean": values.mean().item(),
            "std": values.std().item(),
            "min": values.min().item(),
            "max": values.max().item()
        }

    print("\nUnfused Version Statistics (in MB):")
    for key, values in unfused_stats.items():
        stats = calculate_stats(values)
        print(f"\n{key}:")
        print(f"  Mean: {stats['mean'] / 1024**2:.2f} ± {stats['std'] / 1024**2:.2f}")
        print(f"  Min: {stats['min'] / 1024**2:.2f}")
        print(f"  Max: {stats['max'] / 1024**2:.2f}")

    print("\nFused Version Statistics (in MB):")
    for key, values in fused_stats.items():
        stats = calculate_stats(values)
        print(f"\n{key}:")
        print(f"  Mean: {stats['mean'] / 1024**2:.2f} ± {stats['std'] / 1024**2:.2f}")
        print(f"  Min: {stats['min'] / 1024**2:.2f}")
        print(f"  Max: {stats['max'] / 1024**2:.2f}")

    # Average memory savings
    avg_memory_saved = (torch.tensor(unfused_stats["peak_memory"], dtype=torch.float64).mean() - 
                       torch.tensor(fused_stats["peak_memory"], dtype=torch.float64).mean())
    print(f"\nAverage Memory Saved by Fusion: {avg_memory_saved / 1024**2:.2f} MB")

    print("\nTensor Sizes:")
    print(f"Input features: {feats_x.shape}")
    print(f"Neighbor indices: {nei_inds.shape}")
    print(f"Weights: {weights.shape}")
    print(f"Additional features: {feat_pe.shape}")
    print(f"Linear weights: {linear_weights.shape}")


def test_knn_inv():
    def compare_outputs_by_segments(cuda_neighbors, cuda_k, py_neighbors, py_k, inv_idx):
        """
        Compare outputs by sorting pairs within each segment defined by inv_idx
        """
        cuda_neighbors = cuda_neighbors.cpu().numpy()
        cuda_k = cuda_k.to(torch.int32).cpu().numpy()
        py_neighbors = py_neighbors.cpu().numpy()
        py_k = py_k.cpu().numpy()
        inv_idx = inv_idx.cpu().numpy()

        total_diff = 0
        for batch in range(cuda_neighbors.shape[0]):
            for i in range(len(inv_idx[batch]) - 1):
                start, end = inv_idx[batch][i], inv_idx[batch][i+1]
                if start == end:  # if empty segment
                    continue

                # Get segments and create pairs
                cuda_pairs = list(zip(cuda_neighbors[batch, start:end], 
                                    cuda_k[batch, start:end]))
                py_pairs = list(zip(py_neighbors[batch, start:end], 
                                  py_k[batch, start:end]))

                # Sort pairs within segment
                cuda_pairs.sort()
                py_pairs.sort()

                # Compare
                if cuda_pairs != py_pairs:
                    total_diff += 1
                    if total_diff <= 5:  # Show only first 5 diffs
                        print(f"\nMismatch in batch {batch}, point {i}:")
                        print(f"  CUDA pairs:   {cuda_pairs}")
                        print(f"  Python pairs: {py_pairs}")

        return total_diff


    B = 1               # batch size (meh!!)
    N = 200000          # number of points 
    C_in = 16           # number of inp channels
    K = 64              # number of neighbors
    total_points = N
    num_runs = 3

    torch.manual_seed(42)
    input_points = torch.randn(B, N, 3, device="cuda", requires_grad=False)
    input_points_ds = torch.randn(B, N//2, 3, device="cuda", requires_grad=False)

    neighbor_inds = knn(input_points, input_points, K).contiguous()

    print("\nNeighbor indices shape:", neighbor_inds.shape)
    print("Neighbor indices range:", neighbor_inds.min().item(), "to", neighbor_inds.max().item())

    print("\nPerformance Benchmarking:")
    print(f"Running {num_runs} iterations for each implementation...")

    # Warmup
    print("\nPerforming warmup runs...")
    for _ in range(5):
        _ = pcf_cuda.compute_knn_inverse(neighbor_inds, total_points)
        torch.cuda.synchronize()
        for b in range(B):
            _ = create_inverse_python(neighbor_inds[b].cpu().numpy(), total_points)
        torch.cuda.synchronize()

    # Benchmark
    print("\nBenchmarking CUDA implementation...")
    cuda_times = []
    cuda_memory = []

    for i in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        start_time = time.time()

        cuda_outputs = pcf_cuda.compute_knn_inverse(neighbor_inds, total_points)

        torch.cuda.synchronize()
        end_time = time.time()

        cuda_times.append(end_time - start_time)
        cuda_memory.append(torch.cuda.max_memory_allocated() - mem_before)

        del cuda_outputs
        torch.cuda.empty_cache()

    # Benchmark
    print("Benchmarking Python implementation...")
    python_times = []
    python_memory = []

    for i in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        start_time = time.time()

        python_outputs = [], [], []
        for b in range(B):
            n_out, k_out, idx_out = create_inverse_python(neighbor_inds[b].cpu().numpy(), total_points)
            python_outputs[0].append(n_out)
            python_outputs[1].append(k_out)
            python_outputs[2].append(idx_out)
        python_outputs = [torch.stack(x, dim=0) for x in python_outputs]

        torch.cuda.synchronize()
        end_time = time.time()

        python_times.append(end_time - start_time)
        python_memory.append(torch.cuda.max_memory_allocated() - mem_before)

        del python_outputs
        torch.cuda.empty_cache()

    cuda_times = np.array(cuda_times) * 1000 
    python_times = np.array(python_times) * 1000
    cuda_memory = np.array(cuda_memory) / (1024 * 1024)  # to MB
    python_memory = np.array(python_memory) / (1024 * 1024)  # to MB

    print("\nPerformance Results:")
    print("\nCUDA Implementation:")
    print(f"Runtime (ms):")
    print(f"  Mean: {np.mean(cuda_times):.2f} ± {np.std(cuda_times):.2f}")
    print(f"  Min:  {np.min(cuda_times):.2f}")
    print(f"  Max:  {np.max(cuda_times):.2f}")
    print(f"Memory Usage (MB):")
    print(f"  Mean: {np.mean(cuda_memory):.2f} ± {np.std(cuda_memory):.2f}")
    print(f"  Min:  {np.min(cuda_memory):.2f}")
    print(f"  Max:  {np.max(cuda_memory):.2f}")

    print("\nPython Implementation:")
    print(f"Runtime (ms):")
    print(f"  Mean: {np.mean(python_times):.2f} ± {np.std(python_times):.2f}")
    print(f"  Min:  {np.min(python_times):.2f}")
    print(f"  Max:  {np.max(python_times):.2f}")
    print(f"Memory Usage (MB):")
    print(f"  Mean: {np.mean(python_memory):.2f} ± {np.std(python_memory):.2f}")
    print(f"  Min:  {np.min(python_memory):.2f}")
    print(f"  Max:  {np.max(python_memory):.2f}")

    print("\nSpeedup: {:.2f}x".format(np.mean(python_times) / np.mean(cuda_times)))
    print("Memory Reduction: {:.2f}x".format(np.mean(python_memory) / np.mean(cuda_memory)))

    # Compute Inverse Mapping
    print("\nVerifying correctness...")

    # Cuda Kernel
    cuda_outputs = pcf_cuda.compute_knn_inverse(neighbor_inds, total_points)

    # Python - batch by batch
    python_outputs = [], [], []
    for b in range(B):
        n_out, k_out, idx_out = create_inverse_python(neighbor_inds[b].cpu().numpy(), total_points)
        python_outputs[0].append(n_out)
        python_outputs[1].append(k_out)
        python_outputs[2].append(idx_out)
    
    python_outputs = [torch.stack(x, dim=0) for x in python_outputs]

    print("\nComparing CUDA and Python (native) implementations:")
    print("\nShape comparison:")
    for cuda_out, py_out, name in zip(cuda_outputs, python_outputs, 
                                    ["inv_neighbors", "inv_k", "inv_idx"]):
        print(f"{name}:")
        print(f"  CUDA shape: {cuda_out.shape}")
        print(f"  Python shape: {py_out.shape}")

    print("\ninv_idx comparison:")
    inv_idx_diff = (cuda_outputs[2] != python_outputs[2]).sum().item()
    if inv_idx_diff > 0:
        print(f"WARNING: inv_idx don't match :(  Number of differences: {inv_idx_diff}")
        # Show only first few diffs
        mismatch = (cuda_outputs[2] != python_outputs[2]).nonzero(as_tuple=True)
        for i in range(min(5, len(mismatch[0]))):
            idx = tuple(d[i] for d in mismatch)
            print(f"  Position {idx}:")
            print(f"    CUDA:   {cuda_outputs[2][idx].item()}")
            print(f"    Python: {python_outputs[2][idx].item()}")
    else:
        print("inv_idx sums match exactly :)")

    print("\nNeighbor-K pairs comparison:")
    num_diff = compare_outputs_by_segments(
        cuda_outputs[0], cuda_outputs[1],
        python_outputs[0], python_outputs[1],
        python_outputs[2]
    )

    print(f"\nNumber of segments with mismatches: {num_diff}")
    if num_diff == 0:
        print("All segments match after sorting :)")


def benchmark_knn_inv(point_sizes, k_values, num_runs=3):
    """
    Benchmark the KNN inverse computation with various point sizes and K values
    """
    results = {
        'point_sizes': point_sizes,
        'k_values': k_values,
        'cuda': {
            'runtime': np.zeros((len(point_sizes), len(k_values))),
            'memory': np.zeros((len(point_sizes), len(k_values)))
        },
        'python': {
            'runtime': np.zeros((len(point_sizes), len(k_values))),
            'memory': np.zeros((len(point_sizes), len(k_values)))
        }
    }

    device = torch.device("cuda")
    B = 1  # batch size

    for i, n_points in enumerate(point_sizes):
        print(f"\n===== Testing with {n_points} points =====")

        for j, K in enumerate(k_values):
            print(f"\n----- K = {K} -----")

            torch.manual_seed(42)
            input_points = torch.randn(B, n_points, 3, device=device, requires_grad=False)

            print("Computing KNN indices...")
            neighbor_inds = knn(input_points, input_points, K).contiguous()
            total_points = n_points

            print(f"Neighbor indices shape: {neighbor_inds.shape}")

            print("Performing warmup runs...")
            for _ in range(2):
                _ = pcf_cuda.compute_knn_inverse(neighbor_inds, total_points)
                torch.cuda.synchronize()

                for b in range(B):
                    _ = create_inverse_python(neighbor_inds[b].cpu().numpy(), total_points)
                torch.cuda.synchronize()

            cuda_times = []
            cuda_memory = []
            python_times = []
            python_memory = []

            print(f"Running {num_runs} iterations for each implementation...")

            for r in range(num_runs):
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated()
                start_time = time.time()

                cuda_outputs = pcf_cuda.compute_knn_inverse(neighbor_inds, total_points)

                torch.cuda.synchronize()
                end_time = time.time()

                cuda_times.append(end_time - start_time)
                cuda_memory.append(torch.cuda.max_memory_allocated() - mem_before)

                del cuda_outputs
                torch.cuda.empty_cache()

                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated()
                start_time = time.time()

                python_outputs = [], [], []
                for b in range(B):
                    n_out, k_out, idx_out = create_inverse_python(neighbor_inds[b].cpu().numpy(), total_points)
                    python_outputs[0].append(n_out)
                    python_outputs[1].append(k_out)
                    python_outputs[2].append(idx_out)

                python_outputs = [torch.stack(x, dim=0) if x else None for x in python_outputs]

                torch.cuda.synchronize()
                end_time = time.time()

                python_times.append(end_time - start_time)
                python_memory.append(torch.cuda.max_memory_allocated() - mem_before)

                del python_outputs
                torch.cuda.empty_cache()

            cuda_times_ms = np.array(cuda_times) * 1000
            python_times_ms = np.array(python_times) * 1000
            cuda_memory_mb = np.array(cuda_memory) / (1024 * 1024)
            python_memory_mb = np.array(python_memory) / (1024 * 1024)

            results['cuda']['runtime'][i, j] = np.mean(cuda_times_ms)
            results['cuda']['memory'][i, j] = np.mean(cuda_memory_mb)
            results['python']['runtime'][i, j] = np.mean(python_times_ms)
            results['python']['memory'][i, j] = np.mean(python_memory_mb)

            print(f"\nCUDA Implementation:")
            print(f"Runtime (ms): {np.mean(cuda_times_ms):.2f} ± {np.std(cuda_times_ms):.2f}")
            print(f"Memory (MB): {np.mean(cuda_memory_mb):.2f} ± {np.std(cuda_memory_mb):.2f}")

            print(f"\nPython Implementation:")
            print(f"Runtime (ms): {np.mean(python_times_ms):.2f} ± {np.std(python_times_ms):.2f}")
            print(f"Memory (MB): {np.mean(python_memory_mb):.2f} ± {np.std(python_memory_mb):.2f}")

            print(f"\nSpeedup: {np.mean(python_times_ms)/np.mean(cuda_times_ms):.2f}x")
            print(f"Memory Reduction: {np.mean(python_memory_mb)/np.mean(cuda_memory_mb):.2f}x")

    return results


def test_pconv_linear_opt():
    path = "/nfs/stak/users/sivakuml/hpc-memory/cutlass/data/500000pts"
    device = torch.device("cuda")
    
    feats_x = torch.load(f"{path}/feat_x.pt").to(device).contiguous()
    nei_inds = torch.load(f"{path}/nei_inds.pt").to(device).contiguous()
    weights = torch.load(f"{path}/weights.pt").to(device).contiguous()
    feat_pe = torch.load(f"{path}/feat_pe.pt").to(device).contiguous()
    linear_weights = torch.load(f"{path}/linear_weights.pt").to(device).contiguous()
    linear_bias = torch.load(f"{path}/linear_bias.pt").to(device).contiguous()

    inverse_neighbors, inverse_k, inverse_idx = pcf_cuda.compute_knn_inverse(nei_inds, feats_x.shape[1])

    out_channel = 64
    last_ch = 16
    weightnet_dim = 16

    # Unfused
    pconv = PConv()
    linear_layer = torch.nn.Linear((out_channel // 4 + last_ch) * weightnet_dim, out_channel // 2).to(device)
    linear_layer.weight.data.copy_(linear_weights)
    linear_layer.bias.data.copy_(linear_bias)

    # PConvLinear
    pconv_linear = PConvLinear((out_channel // 4 + last_ch) * weightnet_dim, out_channel // 2)
    pconv_linear.linear.weight.data.copy_(linear_weights)
    pconv_linear.linear.bias.data.copy_(linear_bias)
    pconv_linear = pconv_linear.cuda()

    # Optimized PConvLinear
    pconv_linear_opt = PConvLinearOpt((out_channel // 4 + last_ch) * weightnet_dim, out_channel // 2)
    pconv_linear_opt.linear.weight.data.copy_(linear_weights)
    pconv_linear_opt.linear.bias.data.copy_(linear_bias)
    pconv_linear_opt = pconv_linear_opt.cuda()

    # dummy gradient
    grad_output = torch.randn_like(pconv_linear.linear(torch.empty(feats_x.shape[0], feats_x.shape[1], 
                                              (out_channel // 4 + last_ch) * weightnet_dim, device=device)))

    # -------------- Correctness Check ---------------
    # Forward & Backward (Unfused)
    pconv_out = pconv(feats_x, nei_inds, weights, feat_pe)
    out_unfused = linear_layer(pconv_out)
    out_unfused.backward(grad_output)

    torch.cuda.synchronize()

    unfused_grads = {
        "input": feats_x.grad.clone(),
        "weightnet": weights.grad.clone(),
        "additional": feat_pe.grad.clone(),
        "linear_weight": linear_layer.weight.grad.clone(),
        "linear_bias": linear_layer.bias.grad.clone()
    }

    feats_x.grad = None
    weights.grad = None
    feat_pe.grad = None
    linear_layer.weight.grad = None
    linear_layer.bias.grad = None

    # Forward & Backward (Regular Fused)
    out_fused = pconv_linear(feats_x, nei_inds, weights, feat_pe)
    out_fused.backward(grad_output)

    torch.cuda.synchronize()

    regular_fused_grads = {
        "input": feats_x.grad.clone(),
        "weightnet": weights.grad.clone(),
        "additional": feat_pe.grad.clone(),
        "linear_weight": pconv_linear.linear.weight.grad.clone(),
        "linear_bias": pconv_linear.linear.bias.grad.clone()
    }

    feats_x.grad = None
    weights.grad = None
    feat_pe.grad = None
    pconv_linear.linear.weight.grad = None
    pconv_linear.linear.bias.grad = None

    # Forward & Backward (Optimized Fused)
    out_opt_fused = pconv_linear_opt(feats_x, nei_inds, inverse_neighbors, inverse_k, inverse_idx, weights, feat_pe)
    out_opt_fused.backward(grad_output)

    torch.cuda.synchronize()

    opt_fused_grads = {
        "input": feats_x.grad.clone(),
        "weightnet": weights.grad.clone(),
        "additional": feat_pe.grad.clone(),
        "linear_weight": pconv_linear_opt.linear.weight.grad.clone(),
        "linear_bias": pconv_linear_opt.linear.bias.grad.clone()
    }


    print("\n---------- Correctness Check ----------")
    print("Forward output differences:")
    print("  Unfused vs Regular Fused:", torch.max(torch.abs(out_unfused - out_fused)).item())
    print("  Unfused vs Optimized Fused:", torch.max(torch.abs(out_unfused - out_opt_fused)).item())
    print("  Regular Fused vs Optimized Fused:", torch.max(torch.abs(out_fused - out_opt_fused)).item())

    print("\nGradient differences (max absolute error):")
    for key in unfused_grads:
        diff_unfused_reg = torch.max(torch.abs(unfused_grads[key] - regular_fused_grads[key])).item()
        diff_unfused_opt = torch.max(torch.abs(unfused_grads[key] - opt_fused_grads[key])).item()
        diff_reg_opt = torch.max(torch.abs(regular_fused_grads[key] - opt_fused_grads[key])).item()

        print(f"\n{key:13} grad differences:")
        print(f"  Unfused vs Regular Fused: {diff_unfused_reg:.6f}")
        print(f"  Unfused vs Optimized Fused: {diff_unfused_opt:.6f}")
        print(f"  Regular Fused vs Optimized Fused: {diff_reg_opt:.6f}")

        print(f"{key:13} stats:")
        print(f"  Unfused    - min: {unfused_grads[key].min().item():.6f}, max: {unfused_grads[key].max().item():.6f}, mean: {unfused_grads[key].mean().item():.6f}")
        print(f"  Reg Fused  - min: {regular_fused_grads[key].min().item():.6f}, max: {regular_fused_grads[key].max().item():.6f}, mean: {regular_fused_grads[key].mean().item():.6f}")
        print(f"  Opt Fused  - min: {opt_fused_grads[key].min().item():.6f}, max: {opt_fused_grads[key].max().item():.6f}, mean: {opt_fused_grads[key].mean().item():.6f}")


    print("\n---------- Performance Benchmarking ----------")
    num_runs = 100
    torch.cuda.synchronize()

    # Unfused
    total_time_forward = 0
    total_time_backward = 0
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        pconv_out = pconv(feats_x, nei_inds, weights, feat_pe)
        out_unfused = linear_layer(pconv_out)
        torch.cuda.synchronize()
        total_time_forward += (time.time() - start)

        feats_x.grad = None
        weights.grad = None
        feat_pe.grad = None
        linear_layer.weight.grad = None
        linear_layer.bias.grad = None

        torch.cuda.synchronize()
        start = time.time()
        out_unfused.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
        total_time_backward += (time.time() - start)

    print(f"(Unfused) Average Forward Time: {(total_time_forward / num_runs) * 1000:.2f} ms")
    print(f"(Unfused) Average Backward Time: {(total_time_backward / num_runs) * 1000:.2f} ms")
    print(f"(Unfused) Average Total Time: {((total_time_forward + total_time_backward) / num_runs) * 1000:.2f} ms")

    # Regular fused
    total_time_forward = 0
    total_time_backward = 0
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        out_fused = pconv_linear(feats_x, nei_inds, weights, feat_pe)
        torch.cuda.synchronize()
        total_time_forward += (time.time() - start)

        feats_x.grad = None
        weights.grad = None
        feat_pe.grad = None
        pconv_linear.linear.weight.grad = None
        pconv_linear.linear.bias.grad = None

        torch.cuda.synchronize()
        start = time.time()
        out_fused.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
        total_time_backward += (time.time() - start)

    print(f"(Regular Fused) Average Forward Time: {(total_time_forward / num_runs) * 1000:.2f} ms")
    print(f"(Regular Fused) Average Backward Time: {(total_time_backward / num_runs) * 1000:.2f} ms")
    print(f"(Regular Fused) Average Total Time: {((total_time_forward + total_time_backward) / num_runs) * 1000:.2f} ms")

    # Optimized fused
    total_time_forward = 0
    total_time_backward = 0
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        out_opt_fused = pconv_linear_opt(feats_x, nei_inds, inverse_neighbors, inverse_k, inverse_idx, weights, feat_pe)
        torch.cuda.synchronize()
        total_time_forward += (time.time() - start)

        feats_x.grad = None
        weights.grad = None
        feat_pe.grad = None
        pconv_linear_opt.linear.weight.grad = None
        pconv_linear_opt.linear.bias.grad = None

        torch.cuda.synchronize()
        start = time.time()
        out_opt_fused.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
        total_time_backward += (time.time() - start)

    print(f"(Optimized Fused) Average Forward Time: {(total_time_forward / num_runs) * 1000:.2f} ms")
    print(f"(Optimized Fused) Average Backward Time: {(total_time_backward / num_runs) * 1000:.2f} ms")
    print(f"(Optimized Fused) Average Total Time: {((total_time_forward + total_time_backward) / num_runs) * 1000:.2f} ms")


    print("\n---------- Memory Usage Test ----------")
    # Unfused
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()

    # Forward
    pconv_out = pconv(feats_x, nei_inds, weights, feat_pe)
    torch.cuda.synchronize()
    mem_after_pconv = torch.cuda.memory_allocated()

    out_unfused = linear_layer(pconv_out)
    torch.cuda.synchronize()
    mem_after_linear = torch.cuda.memory_allocated()

    # Backward
    out_unfused.backward(grad_output)
    torch.cuda.synchronize()
    unfused_peak = torch.cuda.max_memory_allocated()

    print(f"\nUnfused Memory Usage (MB):")
    print(f"  PConv allocation: {(mem_after_pconv - mem_before) / 1024**2:.2f}")
    print(f"  Total forward allocation: {(mem_after_linear - mem_before) / 1024**2:.2f}")
    print(f"  Peak memory: {unfused_peak / 1024**2:.2f}")
    print(f"  Additional memory for backward: {(unfused_peak - mem_after_linear) / 1024**2:.2f}")

    feats_x.grad = None
    weights.grad = None
    feat_pe.grad = None
    linear_layer.weight.grad = None
    linear_layer.bias.grad = None
    torch.cuda.empty_cache()

    def test_memory_usage(model, name, inverse_neighbors=None, inverse_k=None, inverse_idx=None):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        # Forward
        if inverse_neighbors is not None:
            out = model(feats_x, nei_inds, inverse_neighbors, inverse_k, inverse_idx, weights, feat_pe)
        else:
            out = model(feats_x, nei_inds, weights, feat_pe)
        torch.cuda.synchronize()
        mem_after_forward = torch.cuda.memory_allocated()

        # Backward
        out.backward(grad_output)
        torch.cuda.synchronize()
        mem_peak = torch.cuda.max_memory_allocated()

        print(f"\n{name} Memory Usage (MB):")
        print(f"  Forward allocation: {(mem_after_forward - mem_before) / 1024**2:.2f}")
        print(f"  Peak memory: {mem_peak / 1024**2:.2f}")
        print(f"  Additional memory for backward: {(mem_peak - mem_after_forward) / 1024**2:.2f}")

        feats_x.grad = None
        weights.grad = None
        feat_pe.grad = None
        model.linear.weight.grad = None
        model.linear.bias.grad = None
        torch.cuda.empty_cache()

        return mem_peak

    reg_peak = test_memory_usage(pconv_linear, "Regular Fused")
    opt_peak = test_memory_usage(pconv_linear_opt, "Optimized Fused", inverse_neighbors, inverse_k, inverse_idx)

    print("\n---------- Memory Reduction ----------")
    print(f"Unfused vs Regular Fused: {(unfused_peak - reg_peak) / 1024**2:.2f} MB ({100 * (1 - reg_peak/unfused_peak):.2f}%)")
    print(f"Unfused vs Optimized Fused: {(unfused_peak - opt_peak) / 1024**2:.2f} MB ({100 * (1 - opt_peak/unfused_peak):.2f}%)")
    print(f"Regular Fused vs Optimized Fused: {(reg_peak - opt_peak) / 1024**2:.2f} MB ({100 * (1 - opt_peak/reg_peak):.2f}%)")

    print("\nTensor Shapes:")
    print(f"  Input features: {feats_x.shape}")
    print(f"  Neighbor indices: {nei_inds.shape}")
    print(f"  Weights: {weights.shape}")
    print(f"  Additional features: {feat_pe.shape}")
    print(f"  Linear weights: {linear_weights.shape}")


def test_pconv_linear_opt_random(point_sizes=[50000, 100000, 200000], K=64, num_runs=5):
    """
    Test PConv + Linear optimization with random data

    Args:
        point_sizes: Number of points
        K: Number of neighbors
        num_runs: Number of runs for each point size

    Returns:
        Dict of results
    """

    device = torch.device("cuda")
    torch.manual_seed(42)
    
    # Params
    B = 1           # batch size
    C_in = 16       # input feature channels 
    C_add = 16      # additional feature channels
    C_mid = 16      # mid feature channels - weight dim
    C_out = 64      # output channels for linear layer

    results = {
        'point_sizes': point_sizes,
        'unfused': {'forward': [], 'backward': [], 'total': [], 'memory': []},
        'fused': {'forward': [], 'backward': [], 'total': [], 'memory': []},
        'opt_fused': {'forward': [], 'backward': [], 'total': [], 'memory': []}
    }

    for n_points in point_sizes:
        print(f"\n\n===== Testing with {n_points} points =====")

        input_points = torch.randn(B, n_points, 3, device=device, requires_grad=True)
        input_features = torch.randn(B, n_points, C_in, device=device, requires_grad=True)

        print("Calculating KNN indices and Inverse indices...")
        neighbor_inds = knn(input_points, input_points, K).contiguous()
        inverse_neighbors, inverse_k, inverse_idx = pcf_cuda.compute_knn_inverse(neighbor_inds, input_features.shape[1])
        torch.cuda.synchronize()
        print(f"KNN indices shape: {neighbor_inds.shape}")
        print(f"KNN inverse indices shape: inv nei-{inverse_neighbors.shape}, inv k-{inverse_k.shape}, inv idx-{inverse_idx.shape}")

        weights = torch.randn(B, n_points, K, C_mid, device=device, requires_grad=True)
        additional_features = torch.randn(B, n_points, K, C_add, device=device, requires_grad=True)
        linear_weights = torch.randn(C_out, (C_in + C_add) * C_mid, device=device, requires_grad=True)
        linear_bias = torch.randn(C_out, device=device, requires_grad=True)

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

        # Dummy gradient for backward pass
        grad_output = torch.randn(B, n_points, C_out, device=device)

        print("Performing warm-up runs...")
        for _ in range(2):
            # Unfused
            pconv_out = pconv(input_features, neighbor_inds, weights, additional_features)
            out_unfused = linear_layer(pconv_out)
            out_unfused.backward(grad_output)

            input_features.grad = None
            weights.grad = None
            additional_features.grad = None
            linear_layer.weight.grad = None
            linear_layer.bias.grad = None

            # Fused
            out_fused = pconv_linear(input_features, neighbor_inds, weights, additional_features)
            out_fused.backward(grad_output)

            input_features.grad = None
            weights.grad = None
            additional_features.grad = None
            pconv_linear.linear.weight.grad = None
            pconv_linear.linear.bias.grad = None

            # Optimized fused
            out_opt_fused = pconv_linear_opt(input_features, neighbor_inds, inverse_neighbors, inverse_k, inverse_idx,
                                                weights, additional_features)
            out_opt_fused.backward(grad_output)

            input_features.grad = None
            weights.grad = None
            additional_features.grad = None
            pconv_linear_opt.linear.weight.grad = None
            pconv_linear_opt.linear.bias.grad = None


        unfused_times = {'forward': [], 'backward': [], 'memory': []}
        fused_times = {'forward': [], 'backward': [], 'memory': []}
        opt_fused_times = {'forward': [], 'backward': [], 'memory': []}

        print(f"Running benchmark with {num_runs} iterations for each implementation...")
        for r in range(num_runs):
            print(f"Run {r+1}/{num_runs}")

            # ----- Unfused -----
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()

            # Forward
            torch.cuda.synchronize()
            start = time.time()
            pconv_out = pconv(input_features, neighbor_inds, weights, additional_features)
            out_unfused = linear_layer(pconv_out)
            torch.cuda.synchronize()
            forward_time = time.time() - start

            # Backward
            input_features.grad = None
            weights.grad = None
            additional_features.grad = None
            linear_layer.weight.grad = None
            linear_layer.bias.grad = None

            torch.cuda.synchronize()
            start = time.time()
            out_unfused.backward(grad_output)
            torch.cuda.synchronize()
            backward_time = time.time() - start

            memory_used = torch.cuda.max_memory_allocated() - mem_before

            unfused_times['forward'].append(forward_time * 1000)        # ms
            unfused_times['backward'].append(backward_time * 1000)      # ms
            unfused_times['memory'].append(memory_used / (1024 ** 2))   # MB

            # ----- Fused -----
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()

            # Forward
            torch.cuda.synchronize()
            start = time.time()
            out_fused = pconv_linear(input_features, neighbor_inds, weights, additional_features)
            torch.cuda.synchronize()
            forward_time = time.time() - start

            # Backward
            input_features.grad = None
            weights.grad = None
            additional_features.grad = None
            pconv_linear.linear.weight.grad = None
            pconv_linear.linear.bias.grad = None

            torch.cuda.synchronize()
            start = time.time()
            out_fused.backward(grad_output)
            torch.cuda.synchronize()
            backward_time = time.time() - start

            memory_used = torch.cuda.max_memory_allocated() - mem_before

            fused_times['forward'].append(forward_time * 1000)          # ms
            fused_times['backward'].append(backward_time * 1000)        # ms
            fused_times['memory'].append(memory_used / (1024 ** 2))     # MB

            # ----- Optimized Fused -----
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()

            # Forward
            torch.cuda.synchronize()
            start = time.time()
            out_opt_fused = pconv_linear_opt(input_features, neighbor_inds, inverse_neighbors, inverse_k, inverse_idx, weights, additional_features)
            torch.cuda.synchronize()
            forward_time = time.time() - start

            # Backward
            input_features.grad = None
            weights.grad = None
            additional_features.grad = None
            pconv_linear_opt.linear.weight.grad = None
            pconv_linear_opt.linear.bias.grad = None

            torch.cuda.synchronize()
            start = time.time()
            out_opt_fused.backward(grad_output)
            torch.cuda.synchronize()
            backward_time = time.time() - start

            memory_used = torch.cuda.max_memory_allocated() - mem_before

            opt_fused_times['forward'].append(forward_time * 1000)          # ms
            opt_fused_times['backward'].append(backward_time * 1000)        # ms
            opt_fused_times['memory'].append(memory_used / (1024 ** 2))     # MB


        for impl, times in [('unfused', unfused_times), 
                            ('fused', fused_times), 
                            ('opt_fused', opt_fused_times)]:
            results[impl]['forward'].append(np.mean(times['forward']))
            results[impl]['backward'].append(np.mean(times['backward']))
            results[impl]['total'].append(np.mean(times['forward']) + np.mean(times['backward']))
            results[impl]['memory'].append(np.mean(times['memory']))

            print(f"\n{impl.upper()} results for {n_points} points:")
            print(f"  Forward time: {np.mean(times['forward']):.2f} ± {np.std(times['forward']):.2f} ms")
            print(f"  Backward time: {np.mean(times['backward']):.2f} ± {np.std(times['backward']):.2f} ms")
            print(f"  Total time: {np.mean(times['forward']) + np.mean(times['backward']):.2f} ms")
            print(f"  Memory usage: {np.mean(times['memory']):.2f} ± {np.std(times['memory']):.2f} MB")

    return results


def test_pconv_linear_cutlass_random(point_sizes=[50000, 100000, 200000], K=64, num_runs=5):
    """
    Test PConv + Linear optimization with random data

    Args:
        point_sizes: Number of points
        K: Number of neighbors
        num_runs: Number of runs for each point size

    Returns:
        Dict of results
    """

    device = torch.device("cuda")
    torch.manual_seed(42)
    
    # Params
    B = 1           # batch size
    C_in = 16       # input feature channels 
    C_add = 16      # additional feature channels
    C_mid = 16      # mid feature channels - weight dim
    C_out = 64      # output channels for linear layer

    results = {
        'point_sizes': point_sizes,
        'unfused': {'forward': [], 'backward': [], 'total': [], 'memory': []},
        'fused': {'forward': [], 'backward': [], 'total': [], 'memory': []},
        'opt_fused': {'forward': [], 'backward': [], 'total': [], 'memory': []}
    }

    for n_points in point_sizes:
        print(f"\n\n===== Testing with {n_points} points =====")

        input_points = torch.randn(B, n_points, 3, device=device, requires_grad=True)
        input_features = torch.randn(B, n_points, C_in, device=device, requires_grad=True)

        print("Calculating KNN indices and Inverse indices...")
        neighbor_inds = knn(input_points, input_points, K).contiguous()
        inverse_neighbors, inverse_k, inverse_idx = pcf_cuda.compute_knn_inverse(neighbor_inds, input_features.shape[1])
        torch.cuda.synchronize()
        print(f"KNN indices shape: {neighbor_inds.shape}")
        print(f"KNN inverse indices shape: inv nei-{inverse_neighbors.shape}, inv k-{inverse_k.shape}, inv idx-{inverse_idx.shape}")

        weights = torch.randn(B, n_points, K, C_mid, device=device, requires_grad=True)
        additional_features = torch.randn(B, n_points, K, C_add, device=device, requires_grad=True)
        linear_weights = torch.randn(C_out, (C_in + C_add) * C_mid, device=device, requires_grad=True)
        linear_bias = torch.randn(C_out, device=device, requires_grad=True)

        # Init
        pconv = PConv()
        linear_layer = torch.nn.Linear((C_in + C_add) * C_mid, C_out).to(device)
        linear_layer.weight.data.copy_(linear_weights)
        linear_layer.bias.data.copy_(linear_bias)

        pconv_linear = PConvLinear((C_in + C_add) * C_mid, C_out)
        pconv_linear.linear.weight.data.copy_(linear_weights)
        pconv_linear.linear.bias.data.copy_(linear_bias)
        pconv_linear = pconv_linear.cuda()

        pconv_linear_opt = PConvLinearCutlass((C_in + C_add) * C_mid, C_out)
        pconv_linear_opt.linear.weight.data.copy_(linear_weights)
        pconv_linear_opt.linear.bias.data.copy_(linear_bias)
        pconv_linear_opt = pconv_linear_opt.cuda()

        # Dummy gradient for backward pass
        grad_output = torch.randn(B, n_points, C_out, device=device)

        print("Performing warm-up runs...")
        for _ in range(2):
            # Unfused
            pconv_out = pconv(input_features, neighbor_inds, weights, additional_features)
            out_unfused = linear_layer(pconv_out)
            out_unfused.backward(grad_output)

            input_features.grad = None
            weights.grad = None
            additional_features.grad = None
            linear_layer.weight.grad = None
            linear_layer.bias.grad = None

            # Fused
            out_fused = pconv_linear(input_features, neighbor_inds, weights, additional_features)
            out_fused.backward(grad_output)

            input_features.grad = None
            weights.grad = None
            additional_features.grad = None
            pconv_linear.linear.weight.grad = None
            pconv_linear.linear.bias.grad = None

            # Optimized fused
            out_opt_fused = pconv_linear_opt(input_features, neighbor_inds, inverse_neighbors, inverse_k, inverse_idx,
                                                weights, additional_features)
            out_opt_fused.backward(grad_output)

            input_features.grad = None
            weights.grad = None
            additional_features.grad = None
            pconv_linear_opt.linear.weight.grad = None
            pconv_linear_opt.linear.bias.grad = None


        unfused_times = {'forward': [], 'backward': [], 'memory': []}
        fused_times = {'forward': [], 'backward': [], 'memory': []}
        opt_fused_times = {'forward': [], 'backward': [], 'memory': []}

        print(f"Running benchmark with {num_runs} iterations for each implementation...")
        for r in range(num_runs):
            print(f"Run {r+1}/{num_runs}")

            # ----- Unfused -----
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()

            # Forward
            torch.cuda.synchronize()
            start = time.time()
            pconv_out = pconv(input_features, neighbor_inds, weights, additional_features)
            out_unfused = linear_layer(pconv_out)
            torch.cuda.synchronize()
            forward_time = time.time() - start

            # Backward
            input_features.grad = None
            weights.grad = None
            additional_features.grad = None
            linear_layer.weight.grad = None
            linear_layer.bias.grad = None

            torch.cuda.synchronize()
            start = time.time()
            out_unfused.backward(grad_output)
            torch.cuda.synchronize()
            backward_time = time.time() - start

            memory_used = torch.cuda.max_memory_allocated() - mem_before

            unfused_times['forward'].append(forward_time * 1000)        # ms
            unfused_times['backward'].append(backward_time * 1000)      # ms
            unfused_times['memory'].append(memory_used / (1024 ** 2))   # MB

            # ----- Fused -----
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()

            # Forward
            torch.cuda.synchronize()
            start = time.time()
            out_fused = pconv_linear(input_features, neighbor_inds, weights, additional_features)
            torch.cuda.synchronize()
            forward_time = time.time() - start

            # Backward
            input_features.grad = None
            weights.grad = None
            additional_features.grad = None
            pconv_linear.linear.weight.grad = None
            pconv_linear.linear.bias.grad = None

            torch.cuda.synchronize()
            start = time.time()
            out_fused.backward(grad_output)
            torch.cuda.synchronize()
            backward_time = time.time() - start

            memory_used = torch.cuda.max_memory_allocated() - mem_before

            fused_times['forward'].append(forward_time * 1000)          # ms
            fused_times['backward'].append(backward_time * 1000)        # ms
            fused_times['memory'].append(memory_used / (1024 ** 2))     # MB

            # ----- Optimized Fused -----
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()

            # Forward
            torch.cuda.synchronize()
            start = time.time()
            out_opt_fused = pconv_linear_opt(input_features, neighbor_inds, inverse_neighbors, inverse_k, inverse_idx, weights, additional_features)
            torch.cuda.synchronize()
            forward_time = time.time() - start

            # Backward
            input_features.grad = None
            weights.grad = None
            additional_features.grad = None
            pconv_linear_opt.linear.weight.grad = None
            pconv_linear_opt.linear.bias.grad = None

            torch.cuda.synchronize()
            start = time.time()
            out_opt_fused.backward(grad_output)
            torch.cuda.synchronize()
            backward_time = time.time() - start

            memory_used = torch.cuda.max_memory_allocated() - mem_before

            opt_fused_times['forward'].append(forward_time * 1000)          # ms
            opt_fused_times['backward'].append(backward_time * 1000)        # ms
            opt_fused_times['memory'].append(memory_used / (1024 ** 2))     # MB


        for impl, times in [('unfused', unfused_times), 
                            ('fused', fused_times), 
                            ('opt_fused', opt_fused_times)]:
            results[impl]['forward'].append(np.mean(times['forward']))
            results[impl]['backward'].append(np.mean(times['backward']))
            results[impl]['total'].append(np.mean(times['forward']) + np.mean(times['backward']))
            results[impl]['memory'].append(np.mean(times['memory']))

            print(f"\n{impl.upper()} results for {n_points} points:")
            print(f"  Forward time: {np.mean(times['forward']):.2f} ± {np.std(times['forward']):.2f} ms")
            print(f"  Backward time: {np.mean(times['backward']):.2f} ± {np.std(times['backward']):.2f} ms")
            print(f"  Total time: {np.mean(times['forward']) + np.mean(times['backward']):.2f} ms")
            print(f"  Memory usage: {np.mean(times['memory']):.2f} ± {np.std(times['memory']):.2f} MB")

    return results


def plot_knn_inv_benchmark(results):
    point_sizes = results['point_sizes']
    k_values = results['k_values']

    plt.figure(figsize=(20, 12))

    plt.subplot(2, 2, 1)
    for j, K in enumerate(k_values):
        cuda_runtimes = results['cuda']['runtime'][:, j]
        python_runtimes = results['python']['runtime'][:, j]

        plt.plot(point_sizes, cuda_runtimes, 'o-', label=f'CUDA (K={K})')
        plt.plot(point_sizes, python_runtimes, 's--', label=f'Python (K={K})')

    plt.title('Runtime Comparison')
    plt.xlabel('Number of Points')
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(2, 2, 2)
    for j, K in enumerate(k_values):
        cuda_memory = results['cuda']['memory'][:, j]
        python_memory = results['python']['memory'][:, j]

        plt.plot(point_sizes, cuda_memory, 'o-', label=f'CUDA (K={K})')
        plt.plot(point_sizes, python_memory, 's--', label=f'Python (K={K})')

    plt.title('Memory Usage Comparison')
    plt.xlabel('Number of Points')
    plt.ylabel('Memory (MB)')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(2, 2, 3)
    for j, K in enumerate(k_values):
        speedup = results['python']['runtime'][:, j] / results['cuda']['runtime'][:, j]
        plt.plot(point_sizes, speedup, 'o-', label=f'K={K}')

    plt.title('CUDA Speedup over Python Implementation')
    plt.xlabel('Number of Points')
    plt.ylabel('Speedup Factor (higher is better)')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')

    plt.subplot(2, 2, 4)
    for j, K in enumerate(k_values):
        mem_reduction = results['python']['memory'][:, j] / results['cuda']['memory'][:, j]
        plt.plot(point_sizes, mem_reduction, 'o-', label=f'K={K}')

    plt.title('CUDA Memory Reduction over Python Implementation')
    plt.xlabel('Number of Points')
    plt.ylabel('Memory Reduction Factor (higher is better)')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')

    plt.tight_layout()
    plt.savefig('knn_inv_benchmark_results.png', dpi=300)
    plt.show()

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    largest_idx = len(point_sizes) - 1
    plt.plot(k_values, results['cuda']['runtime'][largest_idx, :], 'o-', label='CUDA')
    plt.plot(k_values, results['python']['runtime'][largest_idx, :], 's--', label='Python')
    plt.title(f'Runtime vs K (Points = {point_sizes[largest_idx]})')
    plt.xlabel('K Value')
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(k_values, results['cuda']['memory'][largest_idx, :], 'o-', label='CUDA')
    plt.plot(k_values, results['python']['memory'][largest_idx, :], 's--', label='Python')
    plt.title(f'Memory Usage vs K (Points = {point_sizes[largest_idx]})')
    plt.xlabel('K Value')
    plt.ylabel('Memory (MB)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('knn_inv_k_scaling.png', dpi=300)
    plt.show()


def plot_pconv_linear_opt_benchmark(results):
    point_sizes = results['point_sizes']

    plt.figure(figsize=(20, 12))

    # forward times
    plt.subplot(2, 2, 1)
    plt.plot(point_sizes, results['unfused']['forward'], 'o-', label='Unfused')
    plt.plot(point_sizes, results['fused']['forward'], 's-', label='PConv + Linear Fused')
    plt.plot(point_sizes, results['opt_fused']['forward'], '^-', label='PConv + Linear Fused using kNN Inv')
    plt.title('Forward Time')
    plt.xlabel('Number of Points')
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.legend()

    # backward times
    plt.subplot(2, 2, 2)
    plt.plot(point_sizes, results['unfused']['backward'], 'o-', label='Unfused')
    plt.plot(point_sizes, results['fused']['backward'], 's-', label='PConv + Linear Fused')
    plt.plot(point_sizes, results['opt_fused']['backward'], '^-', label='PConv + Linear Fused using kNN Inv')
    plt.title('Backward Time')
    plt.xlabel('Number of Points')
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.legend()

    # total times
    plt.subplot(2, 2, 3)
    plt.plot(point_sizes, results['unfused']['total'], 'o-', label='Unfused')
    plt.plot(point_sizes, results['fused']['total'], 's-', label='PConv + Linear Fused')
    plt.plot(point_sizes, results['opt_fused']['total'], '^-', label='PConv + Linear Fused using kNN Inv')
    plt.title('Total Time (Forward + Backward)')
    plt.xlabel('Number of Points')
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.legend()

    # memory usage
    plt.subplot(2, 2, 4)
    plt.plot(point_sizes, results['unfused']['memory'], 'o-', label='Unfused')
    plt.plot(point_sizes, results['fused']['memory'], 's-', label='PConv + Linear Fused')
    plt.plot(point_sizes, results['opt_fused']['memory'], '^-', label='PConv + Linear Fused using kNN Inv')
    plt.title('Memory Usage')
    plt.xlabel('Number of Points')
    plt.ylabel('Memory (MB)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('pconv_benchmark_results.png', dpi=300)
    plt.show()

    plt.figure(figsize=(20, 6))

    # Speedup relative to unfused
    plt.subplot(1, 2, 1)
    plt.plot(point_sizes, 
             [u/f for u, f in zip(results['unfused']['total'], results['fused']['total'])], 
             's-', label='PConv + Linear Fused vs. Unfused')
    plt.plot(point_sizes, 
             [u/o for u, o in zip(results['unfused']['total'], results['opt_fused']['total'])], 
             '^-', label='PConv + Linear Fused using kNN Inv vs. Unfused')
    plt.plot(point_sizes, 
             [f/o for f, o in zip(results['fused']['total'], results['opt_fused']['total'])], 
             'D-', label='PConv + Linear Fused using kNN Inv vs. PConv + Linear Fused')
    plt.title('Speedup Factor')
    plt.xlabel('Number of Points')
    plt.ylabel('Speedup (higher is better)')
    plt.grid(True)
    plt.legend()

    # Memory savings relative to unfused
    plt.subplot(1, 2, 2)
    plt.plot(point_sizes, 
             [u/f for u, f in zip(results['unfused']['memory'], results['fused']['memory'])], 
             's-', label='PConv + Linear Fused vs. Unfused')
    plt.plot(point_sizes, 
             [u/o for u, o in zip(results['unfused']['memory'], results['opt_fused']['memory'])], 
             '^-', label='PConv + Linear Fused using kNN Inv vs. Unfused')
    plt.plot(point_sizes, 
             [f/o for f, o in zip(results['fused']['memory'], results['opt_fused']['memory'])], 
             'D-', label='PConv + Linear Fused using kNN Inv vs. PConv + Linear Fused')
    plt.title('Memory Reduction Factor')
    plt.xlabel('Number of Points')
    plt.ylabel('Memory Reduction (higher is better)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('pconv_benchmark_speedup.png', dpi=300)
    plt.show()


def test_cutlass_vs_cuda_kernel():
    device = torch.device("cuda")
    torch.manual_seed(42)

    B = 2
    M = 512        # input points
    Nout = 256     # output points
    K = 16         # neighbors
    C_in = 8
    C_add = 4
    C_mid = 8
    C_out = 16
    C_concat = C_in + C_add

    input = torch.rand(B, M, C_in, device=device)
    neighbor_inds = torch.randint(0, M, (B, Nout, K), dtype=torch.int64, device=device)
    additional_features = torch.rand(B, Nout, K, C_add, device=device)
    weights = torch.rand(B, Nout, K, C_mid, device=device)
    linear_weights = torch.rand(C_out, C_concat * C_mid, device=device)
    linear_bias = torch.rand(C_out, device=device)

    input.requires_grad_()
    weights.requires_grad_()
    additional_features.requires_grad_()
    linear_weights.requires_grad_()
    linear_bias.requires_grad_()

    # CUDA
    out_cuda, pconv_out_cuda = pcf_cuda.pconv_linear_forward(
        input, neighbor_inds, weights, additional_features, linear_weights, linear_bias
    )

    # CUTLASS
    out_cutlass, pconv_out_cutlass = pcf_cuda.pconv_linear_cutlass_forward(
        input, neighbor_inds, weights, additional_features, linear_weights, linear_bias
    )

    print("\n--- Output Shapes ---")
    print("final_output: ", out_cuda.shape)
    print("pconv_output: ", pconv_out_cuda.shape)

    # Compare max abs diff
    diff_final = (out_cuda - out_cutlass).abs().max().item()
    diff_pconv = (pconv_out_cuda - pconv_out_cutlass).abs().max().item()

    print("\n--- Forward Max Differences ---")
    print(f"Max diff (final_output): {diff_final:.6f}")
    print(f"Max diff (pconv_output): {diff_pconv:.6f}")

    try:
        torch.testing.assert_close(
            out_cuda, out_cutlass, rtol=1e-4, atol=1e-4,
            msg="Final output mismatch between CUDA and CUTLASS"
        )
        torch.testing.assert_close(
            pconv_out_cuda, pconv_out_cutlass, rtol=1e-4, atol=1e-4,
            msg="PConv intermediate output mismatch between CUDA and CUTLASS"
        )
        print("CUDA and CUTLASS outputs match for final_output and pconv_output!")

    except AssertionError as e:
        print("Mismatch detected!")
        print(str(e))


def test_pconv_linear_cutlass():
    path = "/nfs/stak/users/sivakuml/hpc-memory/cutlass/data/500000pts"
    device = torch.device("cuda")
    
    feats_x = torch.load(f"{path}/feat_x.pt").to(device).contiguous()
    nei_inds = torch.load(f"{path}/nei_inds.pt").to(device).contiguous()
    weights = torch.load(f"{path}/weights.pt").to(device).contiguous()
    feat_pe = torch.load(f"{path}/feat_pe.pt").to(device).contiguous()
    linear_weights = torch.load(f"{path}/linear_weights.pt").to(device).contiguous()
    linear_bias = torch.load(f"{path}/linear_bias.pt").to(device).contiguous()

    inverse_neighbors, inverse_k, inverse_idx = pcf_cuda.compute_knn_inverse(nei_inds, feats_x.shape[1])

    out_channel = 64
    last_ch = 16
    weightnet_dim = 16

    # Unfused
    pconv = PConv()
    linear_layer = torch.nn.Linear((out_channel // 4 + last_ch) * weightnet_dim, out_channel // 2).to(device)
    linear_layer.weight.data.copy_(linear_weights)
    linear_layer.bias.data.copy_(linear_bias)

    # PConvLinear
    pconv_linear = PConvLinear((out_channel // 4 + last_ch) * weightnet_dim, out_channel // 2)
    pconv_linear.linear.weight.data.copy_(linear_weights)
    pconv_linear.linear.bias.data.copy_(linear_bias)
    pconv_linear = pconv_linear.cuda()

    # Optimized PConvLinear
    pconv_linear_opt = PConvLinearCutlass((out_channel // 4 + last_ch) * weightnet_dim, out_channel // 2)
    pconv_linear_opt.linear.weight.data.copy_(linear_weights)
    pconv_linear_opt.linear.bias.data.copy_(linear_bias)
    pconv_linear_opt = pconv_linear_opt.cuda()

    # dummy gradient
    grad_output = torch.randn_like(pconv_linear.linear(torch.empty(feats_x.shape[0], feats_x.shape[1], 
                                              (out_channel // 4 + last_ch) * weightnet_dim, device=device)))

    # -------------- Correctness Check ---------------
    # Forward & Backward (Unfused)
    pconv_out = pconv(feats_x, nei_inds, weights, feat_pe)
    out_unfused = linear_layer(pconv_out)
    out_unfused.backward(grad_output)

    torch.cuda.synchronize()

    unfused_grads = {
        "input": feats_x.grad.clone(),
        "weightnet": weights.grad.clone(),
        "additional": feat_pe.grad.clone(),
        "linear_weight": linear_layer.weight.grad.clone(),
        "linear_bias": linear_layer.bias.grad.clone()
    }

    feats_x.grad = None
    weights.grad = None
    feat_pe.grad = None
    linear_layer.weight.grad = None
    linear_layer.bias.grad = None

    # Forward & Backward (Regular Fused)
    out_fused = pconv_linear(feats_x, nei_inds, weights, feat_pe)
    out_fused.backward(grad_output)

    torch.cuda.synchronize()

    regular_fused_grads = {
        "input": feats_x.grad.clone(),
        "weightnet": weights.grad.clone(),
        "additional": feat_pe.grad.clone(),
        "linear_weight": pconv_linear.linear.weight.grad.clone(),
        "linear_bias": pconv_linear.linear.bias.grad.clone()
    }

    feats_x.grad = None
    weights.grad = None
    feat_pe.grad = None
    pconv_linear.linear.weight.grad = None
    pconv_linear.linear.bias.grad = None

    # Forward & Backward (Optimized Fused)
    out_opt_fused = pconv_linear_opt(feats_x, nei_inds, inverse_neighbors, inverse_k, inverse_idx, weights, feat_pe)
    out_opt_fused.backward(grad_output)

    torch.cuda.synchronize()

    opt_fused_grads = {
        "input": feats_x.grad.clone(),
        "weightnet": weights.grad.clone(),
        "additional": feat_pe.grad.clone(),
        "linear_weight": pconv_linear_opt.linear.weight.grad.clone(),
        "linear_bias": pconv_linear_opt.linear.bias.grad.clone()
    }


    print("\n---------- Correctness Check ----------")
    print("Forward output differences:")
    print("  Unfused vs Regular Fused:", torch.max(torch.abs(out_unfused - out_fused)).item())
    print("  Unfused vs Optimized Fused:", torch.max(torch.abs(out_unfused - out_opt_fused)).item())
    print("  Regular Fused vs Optimized Fused:", torch.max(torch.abs(out_fused - out_opt_fused)).item())

    print("\nGradient differences (max absolute error):")
    for key in unfused_grads:
        diff_unfused_reg = torch.max(torch.abs(unfused_grads[key] - regular_fused_grads[key])).item()
        diff_unfused_opt = torch.max(torch.abs(unfused_grads[key] - opt_fused_grads[key])).item()
        diff_reg_opt = torch.max(torch.abs(regular_fused_grads[key] - opt_fused_grads[key])).item()

        print(f"\n{key:13} grad differences:")
        print(f"  Unfused vs Regular Fused: {diff_unfused_reg:.6f}")
        print(f"  Unfused vs Optimized Fused: {diff_unfused_opt:.6f}")
        print(f"  Regular Fused vs Optimized Fused: {diff_reg_opt:.6f}")

        print(f"{key:13} stats:")
        print(f"  Unfused    - min: {unfused_grads[key].min().item():.6f}, max: {unfused_grads[key].max().item():.6f}, mean: {unfused_grads[key].mean().item():.6f}")
        print(f"  Reg Fused  - min: {regular_fused_grads[key].min().item():.6f}, max: {regular_fused_grads[key].max().item():.6f}, mean: {regular_fused_grads[key].mean().item():.6f}")
        print(f"  Opt Fused  - min: {opt_fused_grads[key].min().item():.6f}, max: {opt_fused_grads[key].max().item():.6f}, mean: {opt_fused_grads[key].mean().item():.6f}")


    print("\n---------- Performance Benchmarking ----------")
    num_runs = 100
    torch.cuda.synchronize()

    # Unfused
    total_time_forward = 0
    total_time_backward = 0
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        pconv_out = pconv(feats_x, nei_inds, weights, feat_pe)
        out_unfused = linear_layer(pconv_out)
        torch.cuda.synchronize()
        total_time_forward += (time.time() - start)

        feats_x.grad = None
        weights.grad = None
        feat_pe.grad = None
        linear_layer.weight.grad = None
        linear_layer.bias.grad = None

        torch.cuda.synchronize()
        start = time.time()
        out_unfused.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
        total_time_backward += (time.time() - start)

    print(f"(Unfused) Average Forward Time: {(total_time_forward / num_runs) * 1000:.2f} ms")
    print(f"(Unfused) Average Backward Time: {(total_time_backward / num_runs) * 1000:.2f} ms")
    print(f"(Unfused) Average Total Time: {((total_time_forward + total_time_backward) / num_runs) * 1000:.2f} ms")

    # Regular fused
    total_time_forward = 0
    total_time_backward = 0
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        out_fused = pconv_linear(feats_x, nei_inds, weights, feat_pe)
        torch.cuda.synchronize()
        total_time_forward += (time.time() - start)

        feats_x.grad = None
        weights.grad = None
        feat_pe.grad = None
        pconv_linear.linear.weight.grad = None
        pconv_linear.linear.bias.grad = None

        torch.cuda.synchronize()
        start = time.time()
        out_fused.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
        total_time_backward += (time.time() - start)

    print(f"(Regular Fused) Average Forward Time: {(total_time_forward / num_runs) * 1000:.2f} ms")
    print(f"(Regular Fused) Average Backward Time: {(total_time_backward / num_runs) * 1000:.2f} ms")
    print(f"(Regular Fused) Average Total Time: {((total_time_forward + total_time_backward) / num_runs) * 1000:.2f} ms")

    # Optimized fused
    total_time_forward = 0
    total_time_backward = 0
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        out_opt_fused = pconv_linear_opt(feats_x, nei_inds, inverse_neighbors, inverse_k, inverse_idx, weights, feat_pe)
        torch.cuda.synchronize()
        total_time_forward += (time.time() - start)

        feats_x.grad = None
        weights.grad = None
        feat_pe.grad = None
        pconv_linear_opt.linear.weight.grad = None
        pconv_linear_opt.linear.bias.grad = None

        torch.cuda.synchronize()
        start = time.time()
        out_opt_fused.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
        total_time_backward += (time.time() - start)

    print(f"(Optimized Fused) Average Forward Time: {(total_time_forward / num_runs) * 1000:.2f} ms")
    print(f"(Optimized Fused) Average Backward Time: {(total_time_backward / num_runs) * 1000:.2f} ms")
    print(f"(Optimized Fused) Average Total Time: {((total_time_forward + total_time_backward) / num_runs) * 1000:.2f} ms")


    print("\n---------- Memory Usage Test ----------")
    # Unfused
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()

    # Forward
    pconv_out = pconv(feats_x, nei_inds, weights, feat_pe)
    torch.cuda.synchronize()
    mem_after_pconv = torch.cuda.memory_allocated()

    out_unfused = linear_layer(pconv_out)
    torch.cuda.synchronize()
    mem_after_linear = torch.cuda.memory_allocated()

    # Backward
    out_unfused.backward(grad_output)
    torch.cuda.synchronize()
    unfused_peak = torch.cuda.max_memory_allocated()

    print(f"\nUnfused Memory Usage (MB):")
    print(f"  PConv allocation: {(mem_after_pconv - mem_before) / 1024**2:.2f}")
    print(f"  Total forward allocation: {(mem_after_linear - mem_before) / 1024**2:.2f}")
    print(f"  Peak memory: {unfused_peak / 1024**2:.2f}")
    print(f"  Additional memory for backward: {(unfused_peak - mem_after_linear) / 1024**2:.2f}")

    feats_x.grad = None
    weights.grad = None
    feat_pe.grad = None
    linear_layer.weight.grad = None
    linear_layer.bias.grad = None
    torch.cuda.empty_cache()

    def test_memory_usage(model, name, inverse_neighbors=None, inverse_k=None, inverse_idx=None):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        # Forward
        if inverse_neighbors is not None:
            out = model(feats_x, nei_inds, inverse_neighbors, inverse_k, inverse_idx, weights, feat_pe)
        else:
            out = model(feats_x, nei_inds, weights, feat_pe)
        torch.cuda.synchronize()
        mem_after_forward = torch.cuda.memory_allocated()

        # Backward
        out.backward(grad_output)
        torch.cuda.synchronize()
        mem_peak = torch.cuda.max_memory_allocated()

        print(f"\n{name} Memory Usage (MB):")
        print(f"  Forward allocation: {(mem_after_forward - mem_before) / 1024**2:.2f}")
        print(f"  Peak memory: {mem_peak / 1024**2:.2f}")
        print(f"  Additional memory for backward: {(mem_peak - mem_after_forward) / 1024**2:.2f}")

        feats_x.grad = None
        weights.grad = None
        feat_pe.grad = None
        model.linear.weight.grad = None
        model.linear.bias.grad = None
        torch.cuda.empty_cache()

        return mem_peak

    reg_peak = test_memory_usage(pconv_linear, "Regular Fused")
    opt_peak = test_memory_usage(pconv_linear_opt, "Optimized Fused", inverse_neighbors, inverse_k, inverse_idx)

    print("\n---------- Memory Reduction ----------")
    print(f"Unfused vs Regular Fused: {(unfused_peak - reg_peak) / 1024**2:.2f} MB ({100 * (1 - reg_peak/unfused_peak):.2f}%)")
    print(f"Unfused vs Optimized Fused: {(unfused_peak - opt_peak) / 1024**2:.2f} MB ({100 * (1 - opt_peak/unfused_peak):.2f}%)")
    print(f"Regular Fused vs Optimized Fused: {(reg_peak - opt_peak) / 1024**2:.2f} MB ({100 * (1 - opt_peak/reg_peak):.2f}%)")

    print("\nTensor Shapes:")
    print(f"  Input features: {feats_x.shape}")
    print(f"  Neighbor indices: {nei_inds.shape}")
    print(f"  Weights: {weights.shape}")
    print(f"  Additional features: {feat_pe.shape}")
    print(f"  Linear weights: {linear_weights.shape}")


def test_pconv_linear_cutlass_by_resolution(
    resolutions = [0.10, 0.05, 0.02],   # 10 cm, 5 cm, 2 cm spacing
    bbox_size = (1.0, 1.0, 1.0),     # size of the box in meters (x_max, y_max, z_max)
    K = 64,
    num_runs = 5
):
    """
    Test PConv + Linear optimization on grid point clouds at specified resolutions.

    Args:
        resolutions: list of point spacing (in meters)
        bbox_size: tuple (x_max, y_max, z_max) defining the volume
        K: number of neighbors for KNN
        num_runs: iterations per resolution

    Returns:
        Dict of results indexed by resolution
    """
    device = torch.device("cuda")
    torch.manual_seed(42)

    B = 1           # batch size
    C_in = 16       # input feature channels 
    C_add = 16      # additional feature channels
    C_mid = 16      # mid feature channels
    C_out = 64      # linear output channels

    results = {
        'resolutions': resolutions,
        'unfused':   {'forward': [], 'backward': [], 'total': [], 'memory': []},
        'fused':     {'forward': [], 'backward': [], 'total': [], 'memory': []},
        'opt_fused': {'forward': [], 'backward': [], 'total': [], 'memory': []},
    }

    pconv           = PConv().cuda()
    linear_layer    = torch.nn.Linear((C_in + C_add) * C_mid, C_out).cuda()
    pconv_linear    = PConvLinear((C_in + C_add) * C_mid, C_out).cuda()
    pconv_linear_opt= PConvLinearCutlass((C_in + C_add) * C_mid, C_out).cuda()

    grad_output = torch.randn(B, 1, C_out, device=device)

    for res in resolutions:
        x_max, y_max, z_max = bbox_size
        xs = torch.arange(0, x_max + 1e-6, res, device=device)
        ys = torch.arange(0, y_max + 1e-6, res, device=device)
        zs = torch.arange(0, z_max + 1e-6, res, device=device)
        grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'), dim=-1)
        points = grid.reshape(-1, 3)
        n_points = points.shape[0]

        print(f"\n===== Resolution {res*100:.0f} cm → {n_points} points =====")
        input_points   = points.unsqueeze(0).requires_grad_(False)
        input_features = torch.randn(B, n_points, C_in, device=device, requires_grad=True)

        neighbor_inds = knn(input_points, input_points, K).contiguous()
        inv_nei, inv_k, inv_idx = pcf_cuda.compute_knn_inverse(neighbor_inds, n_points)
        torch.cuda.synchronize()

        weights             = torch.randn(B, n_points, K, C_mid, device=device, requires_grad=True)
        additional_features = torch.randn(B, n_points, K, C_add, device=device, requires_grad=True)

        for mod in (linear_layer, pconv_linear.linear, pconv_linear_opt.linear):
            mod.weight.data.normal_()
            mod.bias.data.zero_()

        # warm-up
        for _ in range(2):
            # ----- Unfused -----
            out1 = linear_layer(pconv(input_features, neighbor_inds, weights, additional_features))
            out1.backward(torch.randn_like(out1))
            input_features.grad = weights.grad = additional_features.grad = None

            # ----- Fused -----
            out2 = pconv_linear(input_features, neighbor_inds, weights, additional_features)
            out2.backward(torch.randn_like(out2))
            input_features.grad = weights.grad = additional_features.grad = None

            # ----- Opt Fused -----
            out3 = pconv_linear_opt(
                input_features, neighbor_inds, inv_nei, inv_k, inv_idx,
                weights, additional_features
            )
            out3.backward(torch.randn_like(out3))
            input_features.grad = weights.grad = additional_features.grad = None

        unfused_times = {'forward': [], 'backward': [], 'memory': []}
        fused_times   = {'forward': [], 'backward': [], 'memory': []}
        opt_times     = {'forward': [], 'backward': [], 'memory': []}

        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}")

            # ----- Unfused -----
            torch.cuda.empty_cache(); gc.collect()
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            mem0 = torch.cuda.memory_allocated()

            torch.cuda.synchronize(); t0 = time.time()
            out1 = pconv(input_features, neighbor_inds, weights, additional_features)
            out1 = linear_layer(out1)
            torch.cuda.synchronize()
            unf_fwd = (time.time() - t0) * 1e3

            input_features.grad = weights.grad = additional_features.grad = None
            torch.cuda.synchronize(); t0 = time.time()
            out1.backward(torch.randn_like(out1))
            torch.cuda.synchronize()
            unf_bwd = (time.time() - t0) * 1e3

            unf_mem = (torch.cuda.max_memory_allocated() - mem0) / (1024**2)
            unfused_times['forward'].append(unf_fwd)
            unfused_times['backward'].append(unf_bwd)
            unfused_times['memory'].append(unf_mem)

            # ----- Fused -----
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            mem0 = torch.cuda.memory_allocated()

            torch.cuda.synchronize(); t0 = time.time()
            out2 = pconv_linear(input_features, neighbor_inds, weights, additional_features)
            torch.cuda.synchronize()
            fwd = (time.time() - t0) * 1e3

            input_features.grad = weights.grad = additional_features.grad = None
            torch.cuda.synchronize(); t0 = time.time()
            out2.backward(torch.randn_like(out2))
            torch.cuda.synchronize()
            bwd = (time.time() - t0) * 1e3

            mem = (torch.cuda.max_memory_allocated() - mem0) / (1024**2)
            fused_times['forward'].append(fwd)
            fused_times['backward'].append(bwd)
            fused_times['memory'].append(mem)

            # ----- Opt Fused -----
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            mem0 = torch.cuda.memory_allocated()

            torch.cuda.synchronize(); t0 = time.time()
            out3 = pconv_linear_opt(
                input_features, neighbor_inds, inv_nei, inv_k, inv_idx,
                weights, additional_features
            )
            torch.cuda.synchronize()
            fwd_o = (time.time() - t0) * 1e3

            input_features.grad = weights.grad = additional_features.grad = None
            torch.cuda.synchronize(); t0 = time.time()
            out3.backward(torch.randn_like(out3))
            torch.cuda.synchronize()
            bwd_o = (time.time() - t0) * 1e3

            mem_o = (torch.cuda.max_memory_allocated() - mem0) / (1024**2)
            opt_times['forward'].append(fwd_o)
            opt_times['backward'].append(bwd_o)
            opt_times['memory'].append(mem_o)

        for name, times in [
            ('unfused', unfused_times),
            ('fused',   fused_times),
            ('opt_fused', opt_times),
        ]:
            results[name]['forward'].append(np.mean(times['forward']))
            results[name]['backward'].append(np.mean(times['backward']))
            results[name]['total'].append(np.mean(times['forward']) + np.mean(times['backward']))
            results[name]['memory'].append(np.mean(times['memory']))

            print(f"  {name.upper()} @ {res*100:.0f} cm: "
                  f"fwd {results[name]['forward'][-1]:.1f} ms, "
                  f"bwd {results[name]['backward'][-1]:.1f} ms, "
                  f"mem {results[name]['memory'][-1]:.2f} MB")

    return results


def benchmark_pconv_linear_with_tflops(point_sizes=[50000, 100000, 200000], K=64, num_runs=5):
    device = torch.device("cuda")
    torch.manual_seed(42)

    B = 1
    C_in = 16
    C_add = 16
    C_mid = 16
    C_out = 64

    results = {
        'point_sizes': point_sizes,
        'unfused': {'forward': [], 'backward': [], 'total': [], 'memory': [], 'tflops_forward': [], 'tflops_backward': []},
        'fused': {'forward': [], 'backward': [], 'total': [], 'memory': [], 'tflops_forward': [], 'tflops_backward': []},
        'opt_fused': {'forward': [], 'backward': [], 'total': [], 'memory': [], 'tflops_forward': [], 'tflops_backward': []}
    }

    def compute_flops(n_points, K, C_in, C_add, C_mid, C_out):
        """
        Estimate FLOPs for forward pass
        """
        # FLOPs for PConv
        flops_pconv = n_points * K * ((C_in + C_add) * C_mid * 2)  # mul + add per neighbor

        # FLOPs for Linear
        flops_linear = n_points * ((C_in + C_add) * C_mid) * C_out * 2  # matmul
        total_flops_forward = flops_pconv + flops_linear

        # Backward FLOPs roughly ~2x forward FLOPs (general estimate for matmuls)
        total_flops_backward = 2 * total_flops_forward

        return total_flops_forward, total_flops_backward

    for n_points in point_sizes:
        print(f"\n\n===== Benchmarking with {n_points} points =====")

        input_points = torch.randn(B, n_points, 3, device=device, requires_grad=True)
        input_features = torch.randn(B, n_points, C_in, device=device, requires_grad=True)

        neighbor_inds = knn(input_points, input_points, K).contiguous()
        inverse_neighbors, inverse_k, inverse_idx = pcf_cuda.compute_knn_inverse(neighbor_inds, input_features.shape[1])

        weights = torch.randn(B, n_points, K, C_mid, device=device, requires_grad=True)
        additional_features = torch.randn(B, n_points, K, C_add, device=device, requires_grad=True)
        linear_weights = torch.randn(C_out, (C_in + C_add) * C_mid, device=device, requires_grad=True)
        linear_bias = torch.randn(C_out, device=device, requires_grad=True)

        pconv = PConv()
        linear_layer = torch.nn.Linear((C_in + C_add) * C_mid, C_out).to(device)
        linear_layer.weight.data.copy_(linear_weights)
        linear_layer.bias.data.copy_(linear_bias)

        pconv_linear = PConvLinear((C_in + C_add) * C_mid, C_out).cuda()
        pconv_linear.linear.weight.data.copy_(linear_weights)
        pconv_linear.linear.bias.data.copy_(linear_bias)

        pconv_linear_opt = PConvLinearCutlass((C_in + C_add) * C_mid, C_out).cuda()
        pconv_linear_opt.linear.weight.data.copy_(linear_weights)
        pconv_linear_opt.linear.bias.data.copy_(linear_bias)

        grad_output = torch.randn(B, n_points, C_out, device=device)

        # Warm-up
        for _ in range(2):
            out = linear_layer(pconv(input_features, neighbor_inds, weights, additional_features))
            out.backward(grad_output)

            out = pconv_linear(input_features, neighbor_inds, weights, additional_features)
            out.backward(grad_output)

            out = pconv_linear_opt(input_features, neighbor_inds, inverse_neighbors, inverse_k, inverse_idx,
                                   weights, additional_features)
            out.backward(grad_output)

        unfused_times = {'forward': [], 'backward': [], 'memory': []}
        fused_times = {'forward': [], 'backward': [], 'memory': []}
        opt_fused_times = {'forward': [], 'backward': [], 'memory': []}

        for r in range(num_runs):
            # ----- Unfused -----
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()

            start = time.time()
            out_unfused = linear_layer(pconv(input_features, neighbor_inds, weights, additional_features))
            torch.cuda.synchronize()
            forward_time = time.time() - start

            input_features.grad = None
            weights.grad = None
            additional_features.grad = None
            linear_layer.weight.grad = None
            linear_layer.bias.grad = None

            start = time.time()
            out_unfused.backward(grad_output)
            torch.cuda.synchronize()
            backward_time = time.time() - start

            memory_used = torch.cuda.max_memory_allocated() - mem_before

            unfused_times['forward'].append(forward_time * 1000)
            unfused_times['backward'].append(backward_time * 1000)
            unfused_times['memory'].append(memory_used / (1024 ** 2))

            # ----- Fused -----
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()

            start = time.time()
            out_fused = pconv_linear(input_features, neighbor_inds, weights, additional_features)
            torch.cuda.synchronize()
            forward_time = time.time() - start

            input_features.grad = None
            weights.grad = None
            additional_features.grad = None
            pconv_linear.linear.weight.grad = None
            pconv_linear.linear.bias.grad = None

            start = time.time()
            out_fused.backward(grad_output)
            torch.cuda.synchronize()
            backward_time = time.time() - start

            memory_used = torch.cuda.max_memory_allocated() - mem_before

            fused_times['forward'].append(forward_time * 1000)
            fused_times['backward'].append(backward_time * 1000)
            fused_times['memory'].append(memory_used / (1024 ** 2))

            # ----- Optimized Fused -----
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()

            start = time.time()
            out_opt_fused = pconv_linear_opt(input_features, neighbor_inds, inverse_neighbors, inverse_k, inverse_idx,
                                             weights, additional_features)
            torch.cuda.synchronize()
            forward_time = time.time() - start

            input_features.grad = None
            weights.grad = None
            additional_features.grad = None
            pconv_linear_opt.linear.weight.grad = None
            pconv_linear_opt.linear.bias.grad = None

            start = time.time()
            out_opt_fused.backward(grad_output)
            torch.cuda.synchronize()
            backward_time = time.time() - start

            memory_used = torch.cuda.max_memory_allocated() - mem_before

            opt_fused_times['forward'].append(forward_time * 1000)
            opt_fused_times['backward'].append(backward_time * 1000)
            opt_fused_times['memory'].append(memory_used / (1024 ** 2))

        # FLOP calc
        flops_fwd, flops_bwd = compute_flops(n_points, K, C_in, C_add, C_mid, C_out)

        for impl, times in [('unfused', unfused_times),
                            ('fused', fused_times),
                            ('opt_fused', opt_fused_times)]:
            fwd_mean = np.mean(times['forward']) / 1000     # sec
            bwd_mean = np.mean(times['backward']) / 1000    # sec

            tflops_forward = (flops_fwd / 1e12) / fwd_mean
            tflops_backward = (flops_bwd / 1e12) / bwd_mean

            results[impl]['forward'].append(np.mean(times['forward']))
            results[impl]['backward'].append(np.mean(times['backward']))
            results[impl]['total'].append(np.mean(times['forward']) + np.mean(times['backward']))
            results[impl]['memory'].append(np.mean(times['memory']))
            results[impl]['tflops_forward'].append(tflops_forward)
            results[impl]['tflops_backward'].append(tflops_backward)

            print(f"\n{impl.upper()} results for {n_points} points:")
            print(f"  Forward time: {np.mean(times['forward']):.2f} ms ({tflops_forward:.2f} TFLOPS)")
            print(f"  Backward time: {np.mean(times['backward']):.2f} ms ({tflops_backward:.2f} TFLOPS)")
            print(f"  Total time: {np.mean(times['forward']) + np.mean(times['backward']):.2f} ms")
            print(f"  Memory usage: {np.mean(times['memory']):.2f} MB")

    return results


if __name__ == "__main__":
    test_pconv_linear()
    test_pconv_linear_with_memory()

    test_knn_inv()
    results = benchmark_knn_inv(
                    point_sizes=[25000, 50000, 100000, 200000],
                    k_values=[16, 32, 64, 128],
                    num_runs=3
    )
    plot_knn_inv_benchmark(results)

    test_pconv_linear_opt()
    results = test_pconv_linear_opt_random(
                    point_sizes=[25000, 50000, 100000, 200000],
                    K=64,
                    num_runs=3
    )
    plot_pconv_linear_opt_benchmark(results)

    test_cutlass_vs_cuda_kernel()
    test_pconv_linear_cutlass()

    results = test_pconv_linear_cutlass_random(
                    point_sizes=[25000, 50000, 100000, 200000],
                    K=64,
                    num_runs=10
    )
    plot_pconv_linear_opt_benchmark(results)

    test_pconv_linear_cutlass_by_resolution(
                    K=64,
                    num_runs=10
    )

    results = benchmark_pconv_linear_with_tflops(
                    point_sizes=[5000, 10000, 25000, 50000, 100000, 200000, 400000],
                    K=64,
                    num_runs=10
    )