import torch
from torch.nn import functional as F
import pcf_cuda
import warnings
from pointcept.models.utils import offset2batch


# CheckpointFunction class taken from https://github.com/csrhddlam/pytorch-checkpoint/blob/master/checkpoint.py
# copyright (c) 2018 Huiyu Wang
# MIT License: https://github.com/csrhddlam/pytorch-checkpoint/blob/master/LICENSE

def detach_variable(inputs):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)


def check_backward_validity(inputs):
    if not any(inp.requires_grad for inp in inputs):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")


# Forward:
# Performs the forward pass of a fused Point Convolution (PConv) + Linear layer using CUTLASS GEMMs.
# Gathers neighbor features, applies batched GEMM for PConv,
# then projects the result through a linear layer with optional batching for large inputs.
#
# Backward:
# Computes gradients for a fused Point Convolution (PConv) + Linear layer using two CUDA kernels.
# One handles output-point gradients including PConv and linear layers,
# while the other covers input-only points to avoid divergence.
# Uses precomputed reverse indices to ensure full and optimized gradient coverage.
class PConvLinearOptFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_feat, neighbor_inds, inverse_neighbors, inverse_k, inverse_idx,
                        weightnet, additional_features, linear_weights, linear_bias):
        neighbor_inds.requires_grad = False

        # Match the dtypes
        linear_weights_new_type = linear_weights.to(input_feat)
        linear_bias_new_type = linear_bias.to(input_feat)

        output, pconv_output = pcf_cuda.pconv_linear_cutlass_forward(
            input_feat, neighbor_inds, weightnet, additional_features, 
            linear_weights_new_type, linear_bias_new_type)
        ctx.save_for_backward(input_feat, inverse_neighbors, inverse_k, inverse_idx, 
                            neighbor_inds, weightnet, additional_features, 
                            linear_weights_new_type, linear_bias, pconv_output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        input_feat, inverse_neighbors, inverse_k, inverse_idx, neighbor_inds, \
        weightnet, additional_features, linear_weights, bias_type, pconv_output = saved

        grad_output = grad_output.contiguous()

        grads = pcf_cuda.pconv_linear_opt_backward(
            grad_output, input_feat, inverse_neighbors, inverse_k, 
            inverse_idx, neighbor_inds, weightnet, additional_features,
            linear_weights, pconv_output)

        return grads[0], None, None, None, None, grads[1], grads[2], grads[3].to(bias_type), grads[4].to(bias_type)

# Wrapper for PConvLinearOptFunction
class PConvLinearOpt(torch.nn.Module):
    """
    Optimized PConv + Linear fused layer
    """
    def __init__(self, in_features, out_features):
        super(PConvLinearOpt, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, input_features, neighbor_inds, inverse_neighbors, inverse_k, inverse_idx, weightnet, additional_features=None):
        if additional_features is None:
            additional_features = torch.zeros(input_features.shape[0], input_features.shape[1], 
                                          neighbor_inds.shape[2], 0, device=input_features.device)
        return PConvLinearOptFunction.apply(input_features, neighbor_inds, inverse_neighbors, inverse_k, inverse_idx,
                                                weightnet, additional_features, self.linear.weight, self.linear.bias)

def _bn_function_factory(mlp_convs):
    # Used for the gradient checkpointing in WeightNet
    def bn_function(*inputs):
        output = inputs[0]
        for conv in mlp_convs:
            output = conv(output)
        return output
    return bn_function



class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        for i in range(len(ctx.input_tensors)):
            temp = ctx.input_tensors[i]
            ctx.input_tensors[i] = temp.detach()
            ctx.input_tensors[i].requires_grad = temp.requires_grad
        with torch.enable_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        input_grads = torch.autograd.grad(output_tensors, ctx.input_tensors + ctx.input_params, output_grads, allow_unused=True)
        return (None, None) + input_grads

# cp_batchnorm.py taken from https://github.com/csrhddlam/pytorch-checkpoint/blob/master/
# copyright (c) 2018 Huiyu Wang
# MIT License: https://github.com/csrhddlam/pytorch-checkpoint/blob/master/LICENSE
class CpBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(CpBatchNorm2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        self._check_input_dim(input)
        if self.training:
            exponential_average_factor = 0.0
            if self.training and self.track_running_stats:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / self.num_batches_tracked.item()
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats, 0.0, self.eps)

'''
Old version:
def index_points(points, idx):
    """
    Input:
        points: input points data, shape [B, N, C]
        idx: sample index data, shape [B, S] / [B, S, K]
    Return:
        new_points:, indexed points data, shape [B, S, C] / [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(
        device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
'''

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Index point features for both batched and single-batch inputs.

    Args:
        points: Tensor of shape (B, N, C) or (N, C)
        idx: Tensor of shape (B, S), (B, S, K), (S,), or (S, K)

    Returns:
        new_points: Tensor of shape (B, S, C), (S, C) or (B, S, K, C)
    """
    if points.dim() == 3:
        B, N, C = points.shape
    if (points.dim() == 2) or (B == 1):
        if points.dim() == 2:
            return points[idx,:]
        # No batch index needed; direct indexing
        new_points = points[:, idx.squeeze(0), :]
        # Preserve any extra dims in idx
        return new_points.view(1, *idx.shape[1:], C)
    # Batched indexing path
    device = points.device
    batch_shape = list(idx.shape)
    batch_shape[1:] = [1] * (len(batch_shape) - 1)
    batch_indices = torch.arange(B, device=device).view(batch_shape).expand_as(idx)
    new_points = points[batch_indices, idx, :]

    return new_points

def VI_coordinate_transform(localized_xyz, gathered_norm, sparse_xyz_norm, K):
    """
    Compute the viewpoint-invariance aware relative position encoding in VI_PointConv
    From: X. Li et al. Improving the Robustness of Point Convolution on k-Nearest Neighbor Neighborhoods with a Viewpoint-Invariant Coordinate Transform. WACV 2023
    Code copyright 2020 Xingyi Li (MIT License)
    Input:
        dense_xyz: 3D coordinates (note VI only works on 3D)
        nei_inds: indices of neighborhood points for each point
        dense_xyz_norm: surface normals for each point
        sparse_xyz_norm: surface normals for each point in the lower resolution (normally
                the same as dense_xyz_norm, except when downsampling)
    Return:
        VI-transformed point coordinates: a concatenation of rotation+scale invariant dimensions, scale-invariant dimensions and non-invariant dimensions
    """
    r_hat = F.normalize(localized_xyz, dim=3)
    v_miu = sparse_xyz_norm.unsqueeze(
        dim=2) - torch.matmul(
        sparse_xyz_norm.unsqueeze(
            dim=2), r_hat.permute(
                0, 1, 3, 2)).permute(
                    0, 1, 3, 2) * r_hat
    v_miu = F.normalize(v_miu, dim=3)
    w_miu = torch.cross(r_hat, v_miu, dim=3)
    w_miu = F.normalize(w_miu, dim=3)
    theta1 = torch.matmul(gathered_norm, sparse_xyz_norm.unsqueeze(dim=3))
    theta2 = torch.matmul(r_hat, sparse_xyz_norm.unsqueeze(dim=3))
    theta3 = torch.sum(r_hat * gathered_norm, dim=3, keepdim=True)
    theta4 = torch.matmul(localized_xyz, sparse_xyz_norm.unsqueeze(dim=3))
    theta5 = torch.sum(gathered_norm * r_hat, dim=3, keepdim=True)
    theta6 = torch.sum(gathered_norm * v_miu, dim=3, keepdim=True)
    theta7 = torch.sum(gathered_norm * w_miu, dim=3, keepdim=True)
    theta8 = torch.sum(
        localized_xyz *
        torch.cross(
            gathered_norm,
            sparse_xyz_norm.unsqueeze(
                dim=2).repeat(
                1,
                1,
                K,
                1),
            dim=3),
        dim=3,
        keepdim=True)
    theta9 = torch.norm(localized_xyz, dim=3, keepdim=True)
    return torch.cat([theta1,
                      theta2,
                      theta3,
                      theta4,
                      theta5,
                      theta6,
                      theta7,
                      theta8,
                      theta9,
                      localized_xyz],
                     dim=3).contiguous()

class PermutedBN(torch.nn.Module):
    '''
    Permuted Batch Normalization layer, now works for 2D-tensor (N x C) and 3D-tensor inputs (B x N x C)
    '''
    def __init__(self, out_dim, momentum=0.1):
        super(PermutedBN, self).__init__()
        self.bn = torch.nn.BatchNorm1d(out_dim, momentum=momentum)

    def forward(self, x):
        if x.dim() == 2:
            return self.bn(x)
        if x.dim() == 3:
            return self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            raise NotImplementedError

# We did not like that the pyTorch batch normalization requires C to be the 2nd dimension of the Tensor
# It's hard to deal with it during training time, but we can fuse it during inference time
# This one takes in a 4D tensor of shape BNKC, run a linear layer and a BN layer, and then fuses it during inference time
# Output is BNKC as well
# B is batch size, N is number of points, K is number of neighbors
# one would need to call the fuse function during inference time (see
# utils.replace_bn_layers)
class Linear_BN(torch.nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            bn_ver='2d',
            bn_weight_init=1,
            bn_momentum=0.1):
        super().__init__()
        self.c = torch.nn.Linear(in_dim, out_dim)
        self.bn_ver = bn_ver
        if bn_ver == '2d':
            bn = CpBatchNorm2d(out_dim, momentum=bn_momentum)
        else:
            bn = torch.nn.BatchNorm1d(out_dim, momentum=bn_momentum)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
#        torch.nn.init.constant_(bn.bias, 0)
        self.bn = bn

    @torch.no_grad()
    @torch.jit.ignore()
    def fuse(self):
        w = self.bn.weight / (self.bn.running_var + self.bn.eps) ** 0.5
        w = self.c.weight * w[:, None]
        b = self.bn.bias + (self.c.bias - self.bn.running_mean) * self.bn.weight / \
            (self.bn.running_var + self.bn.eps)**0.5
        new_layer = torch.nn.Linear(w.size(1), w.size(0))
        new_layer.weight.data.copy_(w)
        new_layer.bias.data.copy_(b)
        return new_layer

    def forward(self, x):
        x = self.c(x)
        if x.dim() == 4:
            return self.bn(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        else:
            if x.dim() == 2:
                return self.bn(x)
            return self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)


def replace_bn_layers(module: torch.nn.Module, verbose: bool = True):
    """
    Recursively replace any submodule that has a callable `.fuse()` method.
    Returns (module, num_fused).
    """
    fused_count = 0

    # First recurse so we fuse deepest children before parents
    for name, child in list(module.named_children()):
        new_child, child_fused = replace_bn_layers(child, verbose=verbose)
        if child_fused:
            module._modules[name] = new_child
            fused_count += child_fused

    # If this module itself supports `.fuse()`, use it
    if hasattr(module, "fuse") and callable(getattr(module, "fuse")):
        try:
            fused = module.fuse()
            if not isinstance(fused, nn.Module):
                raise TypeError(f"fuse() must return an nn.Module, got {type(fused)}")
            if verbose:
                print(f"Fused: {module.__class__.__name__} -> {fused.__class__.__name__}")
            return fused, fused_count + 1
        except Exception as e:
            raise RuntimeError(f"Failed to fuse {module}: {e}") from e

    return module, fused_count

class PointLayerNorm(torch.nn.Module):
    def __init__(self, channels, eps=1e-5, elementwise_affine=True, bias = True):
        """
        Applies Layer Normalization for packed representation of point clouds
        as described in the paper Layer Normalization (https://arxiv.org/abs/1607.06450)
        Input:
            x: input points data, shape [B, N, C]
        Return:
            normalized points data, shape [B, N, C]
        """
        self.eps = eps
        self.channels = channels
        self.elementwise_affine = elementwise_affine
        self.bias = bias
        super(PointLayerNorm, self).__init__()
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(channels))
        if bias:
            self.bias_val = torch.nn.Parameter(torch.zeros(channels))
    def forward(self, x, batch_offsets, batch_indices):
        from torch_scatter import segment_csr
        if batch_offsets.dim() == 1 and x.dim() > 2:
            batch_offsets = torch.unsqueeze(batch_offsets,0)
        
        if x.dim() > 2:
            # first reduce all the extra dimensions before running segment_csr
            dims = tuple(range(2,x.dim()))
            mean_x = torch.mean(x, dims)
        mean_x = segment_csr(mean_x, batch_offsets, reduce='mean')
        mean_x = mean_x[:, batch_indices]
        while mean_x.dim() < x.dim():
            mean_x = mean_x.unsqueeze(-1)
        normalized_x = x - mean_x
        if normalized_x.dim() > 2:
            dims = tuple(range(2,normalized_x.dim()))
            normalized_2 = torch.mean(normalized_x**2, dims)
        else:
            normalized_2 = normalized_x **2
        std_x = torch.sqrt(segment_csr(normalized_2, batch_offsets, reduce='mean') + self.eps)
        while std_x.dim() < x.dim():
            std_x = std_x[:,batch_indices].unsqueeze(-1)
        if self.elementwise_affine is True:
            if self.bias is True:
                return normalized_x / std_x * self.weight + self.bias_val
            else:
                return normalized_x / std_x * self.weight
        else:
            if self.bias is True:
                return normalized_x / std_x + self.bias_val
            else:
                return normalized_x / std_x

class PointGroupNorm(torch.nn.Module):
    """
    Group Normalization for packed point-cloud tensors WITH per-batch aggregation
    across points and middle dims.

    Input/Shapes:
        x: [B, N, *M, C]    (C is last; *M can be empty or multiple dims)
        batch_offsets: CSR pointer for the N dimension; shape [B+1] (or [1, B+1] broadcastable)
        batch_indices: Long tensor of shape [N] mapping each point to its batch (0..B-1)

    Stats:
        For each batch b and group g, mean/var are computed over (N, *M, Cg).

    Args:
        channels (int): total channels C
        num_groups (int): number of groups G (C must be divisible by G)
        eps (float): numerical stability
        elementwise_affine (bool): learnable per-channel gamma/beta
        bias (bool): include beta if True

    Returns:
        Tensor with the same shape as x, normalized per (batch, group).
    """
    def __init__(self, channels, num_groups = 4, eps=1e-5,
                 elementwise_affine=True, bias=True):
        super().__init__()
        assert channels % num_groups == 0, "channels must be divisible by num_groups"
        self.channels = channels
        self.num_groups = num_groups
        self.group_size = channels // num_groups
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = bias

        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(channels))
            self.bias_val = torch.nn.Parameter(torch.zeros(channels) if bias else torch.empty(0))
            if not bias:
                self.register_parameter('bias_val', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias_val', None)

    def forward(self, x, batch_offsets, batch_indices):
        from torch_scatter import segment_csr

        assert x.size(-1) == self.channels, \
            f"Expected last dim C={self.channels}, got {x.size(-1)}"

        # Make batch_offsets broadcastable like in your PointLayerNorm
        if batch_offsets.dim() == 1 and x.dim() > 2:
            batch_offsets = batch_offsets.unsqueeze(0)  # [1, B+1]

        B, N = x.shape[0], x.shape[1]
        G, Cg = self.num_groups, self.group_size

        # Reshape channels into groups: [B, N, *M, G, Cg]
        xg = x.view(*x.shape[:-1], G, Cg)

        # Prepare the dims that represent *M (middle dims)
        num_middle = x.dim() - 3  # dims between N and C
        # In xg, middle dims are at positions: 2 .. (1 + num_middle)
        if num_middle > 0:
            middle_reduce_dims = tuple(range(2, 2 + num_middle))  # reduce over all *M
        else:
            middle_reduce_dims = ()

        # ---------- Mean over (N, *M, Cg) per (batch, group) ----------
        # First reduce within each point/middle over (Cg and *M) -> [B, N, G]
        if num_middle > 0:
            mean_point_group = xg.mean(dim=(-1,) + middle_reduce_dims)  # reduce Cg and *M
        else:
            mean_point_group = xg.mean(dim=-1)  # only reduce Cg

        # Then average across points within each batch using CSR -> [B, G]
        mean_batch_group = segment_csr(mean_point_group, batch_offsets, reduce='mean')

        # Map per-(batch,group) means back to all points, then broadcast over *M and Cg
        # Start: [B, G] -> insert N via batch_indices -> [B, N, G]
        mean_bg_pts = mean_batch_group[:, batch_indices, :]  # [B, N, G]
        # Expand to [B, N, *M, G, Cg]
        if num_middle > 0:
            # add *M dims as singleton, then expand to xg shape
            expand_shape = list(xg.shape)
            mean_bg_pts = mean_bg_pts.view(B, N, *([1] * num_middle), G)
        mean_bg_pts = mean_bg_pts.unsqueeze(-1)  # add Cg dim -> [B, N, *M, G, 1]
        mean_bg_pts = mean_bg_pts.expand_as(xg)

        # ---------- Variance over (N, *M, Cg) per (batch, group) ----------
        dev = xg - mean_bg_pts  # [B, N, *M, G, Cg]
        if num_middle > 0:
            var_point_group = (dev * dev).mean(dim=(-1,) + middle_reduce_dims)  # [B, N, G]
        else:
            var_point_group = (dev * dev).mean(dim=-1)  # [B, N, G]

        var_batch_group = segment_csr(var_point_group, batch_offsets, reduce='mean')  # [B, G]
        std_group = torch.sqrt(var_batch_group + self.eps)
        std_bg_pts = std_group[:, batch_indices, :]  # [B, N, G]
        if num_middle > 0:
            std_bg_pts = std_bg_pts.view(B, N, *([1] * num_middle), G)
        std_bg_pts = std_bg_pts.unsqueeze(-1).expand_as(xg)  # [B, N, *M, G, Cg]

        # ---------- Normalize and reshape back ----------
        xg_norm = (xg - mean_bg_pts) / std_bg_pts  # [B, N, *M, G, Cg]
        x_norm = xg_norm.view(*x.shape)  # back to [B, N, *M, C]

        # ---------- Affine ----------
        if self.elementwise_affine:
            w = self.weight
            if self.use_bias and self.bias_val is not None:
                b = self.bias_val
                return x_norm * w + b
            else:
                return x_norm * w
        else:
            return x_norm
            
class BatchLayerNorm(torch.nn.Module):
    def __init__(self, channels, eps=1e-5, elementwise_affine=True, bias = True):
        """
        Applies Layer Normalization for packed representation of point clouds
        as described in the paper Layer Normalization (https://arxiv.org/abs/1607.06450)
        Input:
            x: input points data, shape [B, N, C]
        Return:
            normalized points data, shape [B, N, C]
        """
        self.eps = eps
        self.channels = channels
        self.elementwise_affine = elementwise_affine
        self.bias = bias
        super(BatchLayerNorm, self).__init__()
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(channels))
        if bias:
            self.bias_val = torch.nn.Parameter(torch.zeros(channels))
    def forward(self, x):
        
        mean_x = torch.mean(x, tuple(range(1,x.dim())))
        
        normalized_x = x - mean_x
        std_x = torch.sqrt(torch.mean(normalized_x ** 2, tuple(range(1,x.dim()))))+ self.eps
        if self.elementwise_affine is True:
            if self.bias is True:
                return normalized_x / std_x * self.weight + self.bias_val
            else:
                return normalized_x / std_x * self.weight
        else:
            if self.bias is True:
                return normalized_x / std_x + self.bias_val
            else:
                return normalized_x / std_x
            
class PointInstanceNorm(torch.nn.Module):
    def __init__(self, channels, eps=1e-5, elementwise_affine=True, bias = True):
        """
        Applies Layer Normalization for packed representation of point clouds
        as described in the paper Layer Normalization (https://arxiv.org/abs/1607.06450)
        Input:
            x: input points data, shape [B, N, C]
        Return:
            normalized points data, shape [B, N, C]
        """
        self.eps = eps
        self.channels = channels
        self.elementwise_affine = elementwise_affine
        self.bias = bias
        super(PointInstanceNorm, self).__init__()
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(channels))
        if bias:
            self.bias_val = torch.nn.Parameter(torch.zeros(channels))
    def forward(self, x, batch_offsets, batch_indices):
        from torch_scatter import segment_csr
        if batch_offsets.dim() == 1 and x.dim() > 2:
            batch_offsets = torch.unsqueeze(batch_offsets,0)
        
        if x.dim() == 4:
            # first reduce the K dimension (2nd) before running segment_csr
            mean_x = torch.mean(x, -2)
        else:
            mean_x = x
        mean_x = segment_csr(mean_x, batch_offsets, reduce='mean')
        mean_x = mean_x[:, batch_indices]
        if mean_x.dim() < x.dim():
            mean_x = mean_x.unsqueeze(-2)
        normalized_x = x - mean_x
        if normalized_x.dim() == 4:
            normalized_2 = torch.mean(normalized_x**2, -2)
        else:
            normalized_2 = normalized_x **2
        std_x = torch.sqrt(segment_csr(normalized_2, batch_offsets, reduce='mean') + self.eps)
        if std_x.dim() < x.dim():
            std_x = std_x[:,batch_indices].unsqueeze(-2)
        else:
            std_x = std_x[:,batch_indices]
        if self.elementwise_affine is True:
            if self.bias is True:
                return normalized_x / std_x * self.weight + self.bias_val
            else:
                return normalized_x / std_x * self.weight
        else:
            if self.bias is True:
                return normalized_x / std_x + self.bias_val
            else:
                return normalized_x / std_x