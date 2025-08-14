def grid_sampling(point, stride = 2, traceable = False, reduce = 'mean', pad=1):
    assert reduce in ["sum", "mean", "min", "max"]
    # Grid Sampling taken from PTv3 code
    if "grid_coord" in point.keys():
        grid_coord = point.grid_coord
    elif {"coord", "grid_size"}.issubset(point.keys()):
        grid_coord = torch.div(
            point.coord - point.coord.min(0)[0],
            point.grid_size,
            rounding_mode="trunc",
        ).int()
    else:
        raise AssertionError(
            "[gird_coord] or [coord, grid_size] should be include in the Point"
        )
    grid_coord = torch.div(grid_coord, stride, rounding_mode="trunc")
    grid_coord = grid_coord | point.batch.view(-1, 1) << 48
    grid_coord, cluster, counts = torch.unique(
        grid_coord,
        sorted=True,
        return_inverse=True,
        return_counts=True,
        dim=0,
    )
    grid_coord = grid_coord & ((1 << 48) - 1)
    # indices of point sorted by cluster, for torch_scatter.segment_csr
    _, indices = torch.sort(cluster)
    # index pointer for sorted point, for torch_scatter.segment_csr
    idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
    # head_indices of each cluster, for reduce attr e.g. code, batch
    head_indices = indices[idx_ptr[:-1]]
    point_dict = Dict(
        feat=torch_scatter.segment_csr(
            point.feat[indices], idx_ptr, reduce= reduce
        ),
        coord=torch_scatter.segment_csr(
            point.coord[indices], idx_ptr, reduce=reduce
        ),
        grid_coord=grid_coord,
        batch=point.batch[head_indices],
    )
    if "origin_coord" in point.keys():
        point_dict["origin_coord"] = torch_scatter.segment_csr(
            point.origin_coord[indices], idx_ptr, reduce=reduce
        )
    if "condition" in point.keys():
        point_dict["condition"] = point.condition
    if "context" in point.keys():
        point_dict["context"] = point.context
    if "name" in point.keys():
        point_dict["name"] = point.name
    if "split" in point.keys():
        point_dict["split"] = point.split
    if "color" in point.keys():
        point_dict["color"] = torch_scatter.segment_csr(
            point.color[indices], idx_ptr, reduce=reduce
        )
    if "normal" in point.keys() and point.normal is not None:
        point_dict["normal"] = torch_scatter.segment_csr(
            point.normal[indices], idx_ptr, reduce=reduce
        )
        # Make sure it's still a unit vector after reduction
        point_dict["normal"] = point_dict["normal"] / torch.norm(
            point_dict["normal"], dim=-1, keepdim=True
        )
    if "grid_size" in point.keys():
        point_dict["grid_size"] = point.grid_size * stride

    if traceable:
        point_dict["pooling_inverse"] = cluster
        point_dict["pooling_parent"] = point
    order = point.order
    point = Point(point_dict)
    return point