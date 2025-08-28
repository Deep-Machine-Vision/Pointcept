# pointcept/models/utils/knn.py
# This file contains functions to compute k-nearest neighbors (KNN) using different methods.
# by Pritesh Verma, Wenxuan Wu, and Fuxin Li. Oregon State University.

import torch
import numpy as np
from sklearn.neighbors import KDTree
try:
    from pykeops.torch import LazyTensor
    keops_available = True
except ImportError:
    keops_available = False
try:
    import cupy as cp
    from cuvs.neighbors import brute_force
    cuvs_available = True
except ImportError:
    cuvs_available = False

def knn_keops(ref_points, query_points, K):
    """
    Compute k-nearest neighbors using KeOps, and return indices.
    """
    if isinstance(ref_points, np.ndarray):
        ref_points = torch.tensor(ref_points, dtype=torch.float32)
    if isinstance(query_points, np.ndarray):
        query_points = torch.tensor(query_points, dtype=torch.float32)
    
    
    ref_lazy = LazyTensor(ref_points[:, None, :])  # Mx1xD
    query_lazy = LazyTensor(query_points[None, :, :])  # 1xNxD
    distances = ((ref_lazy - query_lazy) ** 2).sum(-1)  # Pairwise squared distances (MxN)
   
    indices = distances.argKmin(K, dim=0)
    
    del ref_lazy, query_lazy, distances
    assert isinstance(indices, torch.Tensor), "indices is not a torch.Tensor"

    return indices

def knn_cuvs_brute_force(ref_points, query_points, K):
    """
    Compute k-nearest neighbors using brute force, and return indices.
    """
    ref_points = cp.asarray(ref_points)
    query_points = cp.asarray(query_points)
    index = brute_force.build(ref_points, metric='sqeuclidean')

    dist, ind = brute_force.search(index, query_points, K)
    ind = torch.as_tensor(ind)
    return ind

def compute_knn(ref_points, query_points, K, dilated_rate=1, method='cuvs_brute_force'):
    """
    Compute KNN
    Input:
        ref_points: reference points (MxD)
        query_points: query points (NxD)
        K: the amount of neighbors for each point
        dilated_rate: If set to larger than 1, then select more neighbors and then choose from them
        (Engelmann et al. Dilated Point Convolutions: On the Receptive Field Size of Point Convolutions on 3D Point Clouds. ICRA 2020)
    Output:
        neighbors_idx: for each query point, its K nearest neighbors among the reference points (N x K)
    """
    num_ref_points = ref_points.shape[0]


    if num_ref_points < K or num_ref_points < dilated_rate * K:
        num_query_points = query_points.shape[0]
        inds = np.random.choice(
            num_ref_points, (num_query_points, K)).astype(
            np.int32)
        # Convert to torch tensor if ref_points is a torch tensor
        if isinstance(ref_points, torch.Tensor):
            inds = torch.tensor(inds).to(ref_points.device)
        return inds
    if method == 'sklearn':
        kdt = KDTree(ref_points)
        neighbors_idx = kdt.query(
            query_points,
            k=K * dilated_rate,
            return_distance=False)
    
    elif method == 'keops':
        assert keops_available, "KeOps is not available. Please install it to use knn_keops."
        neighbors_idx = knn_keops(ref_points, query_points, K*dilated_rate)
    
    elif method == 'cuvs_brute_force':
        assert cuvs_available, "cuvs is not available. Please install it to use knn_cuvs_brute_force."
        neighbors_idx = knn_cuvs_brute_force(ref_points, query_points, K*dilated_rate)
    else:
        raise Exception('compute_knn: unsupported knn algorithm')
    if dilated_rate > 1:
        neighbors_idx = np.array(
            neighbors_idx[:, ::dilated_rate], dtype=np.int32)

    if method == 'keops':
        assert isinstance(neighbors_idx, torch.Tensor), "neighbors_idx is not a torch.Tensor"
    return neighbors_idx

def compute_knn_batched(
    ref_points,
    query_points,
    batch_offset_ref,
    batch_offset_query,
    K,
    dilated_rate=1,
    method='cuvs_brute_force',
    return_local_index=False,
):
    """
    Batched KNN over contiguous blocks in `points`:
      - Block 0: points[0:N1] queries neighbors within points[0:N1]
      - Block 1: points[N1:N1+N2] within the same block
      - ...
    Args:
        ref_points: (M x D) reference points
        query_points: (N x D) query points
        batch_offset_ref: iterable of ints [M1, M2, ..., MB] which indicates the start of each reference point cloud
        batch_offset_query: iterable of ints [N1, N2, ..., NB] with indicates the start of each query point cloud
        K: neighbors per query
        dilated_rate: passed through to compute_knn
        method: 'sklearn' | 'keops' | 'cuvs_brute_force'
        return_local_index: if True, indices are 0..(Ni-1) within each block;
                            if False (default), indices are global 0..(N-1)
    Returns:
        neighbors_idx_all: (N x K) indices (torch.Tensor if `points` is torch.Tensor, else np.ndarray)
    """
    import numpy as np
    import torch

    # Figure out return type / device
    is_torch = isinstance(query_points, torch.Tensor)
    device = query_points.device if is_torch else None

    M = ref_points.shape[0]
    N = query_points.shape[0]
    # Container for each block's indices
    block_inds = []

    for i in range(len(batch_offset_query)):
        if i>0:
            pts_i_ref = ref_points[batch_offset_ref[i-1]:batch_offset_ref[i]]
            pts_i_query = query_points[batch_offset_query[i-1]:batch_offset_query[i]]  # (Ni x D)
        else:
            pts_i_ref = ref_points[:batch_offset_ref[i]]
            pts_i_query = query_points[:batch_offset_query[i]] 

        inds_i = compute_knn(
            pts_i_ref, pts_i_query, K,
            dilated_rate=dilated_rate,
            method=method
        )

        # Convert to torch on right device if needed
        if is_torch and not isinstance(inds_i, torch.Tensor):
            inds_i = torch.as_tensor(inds_i, device=device)
        elif (not is_torch) and isinstance(inds_i, torch.Tensor):
            inds_i = inds_i.cpu().numpy()

        # Map local→global if requested
        if not return_local_index and i > 0:
            if is_torch:
                inds_i = inds_i + batch_offset_ref[i-1]
            else:
                inds_i = inds_i + batch_offset_ref[i-1]

        block_inds.append(inds_i)

    # Concatenate along the query dimension (N x K)
    if is_torch:
        neighbors_idx_all = torch.cat(block_inds, dim=0)
    else:
        neighbors_idx_all = np.concatenate(block_inds, axis=0)

    return neighbors_idx_all