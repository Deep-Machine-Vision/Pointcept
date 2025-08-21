# pointcept/models/utils/knn.py
# This file contains functions to compute k-nearest neighbors (KNN) using different methods.
# by Pritesh Verma, Wenxuan Wu, and Fuxin Li. Oregon State University.

import torch
import numpy as np
from sklearn.neighbors import KDTree
try:
    from pykeops.torch import LazyTensor
except ImportError:
    keops_available = False
try:
    import cuvs as cp
    from cuvs.neighbors import brute_force
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
    ind = cp.asarray(ind)
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
