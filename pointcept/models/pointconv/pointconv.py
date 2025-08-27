from timm.layers import DropPath


import torch
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import torch.nn as nn
from .pointconv_utils import PConvLinearOpt, index_points, VI_coordinate_transform
# Treatments to deal with batch normalization and gradient checkpointing for batch normalization
# Can be removed if not using batch normalization
from .pointconv_utils import CheckpointFunction,CpBatchNorm2d,Linear_BN,PermutedBN,_bn_function_factory


class PointLinearLayer(nn.Module):
    """
    A simple linear layer for point clouds with fused batch normalization if bn is chosen as the normalization layer and optional activation layer
    """
    def __init__(self, in_channels, out_channels, norm_layer=None, act_layer=None, bn_ver='2d'):
        super(PointLinearLayer, self).__init__()
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        if norm_layer == 'bn':
            self.linear = Linear_BN(in_channels, out_channels, bn_ver)
        else:
            # norm_layer can also be None, then there is no normalization
            if norm_layer and not callable(norm_layer):
                raise AssertionError("norm_layer must be 'bn' or a function pointer")
            self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.linear(x)
        if callable(self.norm_layer):
            x = self.norm_layer(x)
        if callable(self.act_layer):
            x = self.act_layer(x)
        return x

# Using nn.Module right now for using pointconv layers as standalone modules
# PointModule doesn't seem to be different from nn.Module
class WeightNet(nn.Module):
    '''
    WeightNet for PointConv. This runs a few MLP layers (defined by hidden_unit) on the 
    point coordinates and outputs generated weights for each neighbor of each point. 
    The weights will then be matrix-multiplied with the input to perform convolution

    Parameters:
        in_channel: Number of input channels
        out_channel: Number of output channels
        hidden_unit: Number of hidden units, a list which can contain multiple hidden layers
        norm_layer: 'bn' for Batch Normalization (necessary because of the special treatments needed for BN)
                       or a function pointer from pyTorch (e.g. torch.nn.LayerNorm)
        efficient: If set to True, then gradient checkpointing is used in training to reduce memory cost. 
                   Note: you may not want to do this if you have global gradient checkpointing.
    Input: Coordinates for all the kNN neighborhoods
           input shape is B x N x K x in_channel, B is batch size, in_channel is the dimensionality of
            the coordinates (usually 3 for 3D or 2 for 2D, 12 for VI), K is the neighborhood size,
            N is the number of points
    Output: The generated weights B x N x K x C_mid
    '''
    def __init__(
            self,
            in_channel,
            out_channel,
            hidden_unit=[8, 8],
            norm_layer = 'bn',
            act_layer = torch.nn.ReLU(inplace=True),
            efficient=False):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.efficient = efficient
        if norm_layer == 'bn':
            self.bn = True
        else:
            self.bn = False
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(PointLinearLayer(in_channel, out_channel))
        else:
            self.mlp_convs.append(PointLinearLayer(in_channel, hidden_unit[0], norm_layer=norm_layer, act_layer=act_layer))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(
                    PointLinearLayer(hidden_unit[i - 1], hidden_unit[i], norm_layer=norm_layer))
                # TODO: Copilot suggested to remove the last norm layer, maybe try it at some point
            self.mlp_convs.append(PointLinearLayer(hidden_unit[-1], out_channel, norm_layer=norm_layer,act_layer = act_layer))

    def real_forward(self, localized_xyz):
        # xyz : BxNxKxC
        weights = localized_xyz
        for conv in self.mlp_convs:
            weights = conv(weights)

        return weights

    def forward(self, localized_xyz):
        # Call with localized_xyz of shape BxNxKxC
        # localized_xyz is the coordinates of the kNN neighborhoods, B is batch size, N is number of points,
        # K is the neighborhood size, C is the dimensionality of the coordinates (usually 3 for 3D or 2 for 2D, 12 for VI)

        # PyTorch gradient checkpointing has a bug for batch normalization, hence we need to handle it differently
        if self.efficient and self.training and self.bn:
            # Try this so that weights have gradient
            #            weights = self.mlp_convs[0](localized_xyz)
            conv_bn_relu = _bn_function_factory(self.mlp_convs)
            dummy = torch.zeros(
                1,
                dtype=torch.float32,
                requires_grad=True,
                device=localized_xyz.device)
            args = [localized_xyz + dummy]
            if self.training:
                for conv in self.mlp_convs:
                    args += tuple(conv.linear.bn.parameters())
                    args += tuple(conv.linear.c.parameters())
                weights = CheckpointFunction.apply(conv_bn_relu, 1, *args)
        elif self.efficient and self.training:
            # Use gradient checkpointing for memory efficiency. WeightNet can be
            # very inefficient during training if checkpointing is not used
            weights = checkpoint.checkpoint(self.real_forward, localized_xyz)
        else:
            weights = self.real_forward(localized_xyz)
        return weights

class PointConvResBlock(nn.Module):
    '''
    PointConv block with a positional embedding concatenated to the features
    and a ResNet-style bottleneck structure and shortcut connections
    Parameters:
        in_channel: Number of input channels
        out_channel: Number of output channels
        weightnet: Number of input/output channels for weightnet
    Input:
        dense_xyz: tensor (batch_size, num_points, 3). The coordinates of the points before subsampling (if it is a "strided" convolution wihch simultaneously subsamples the point cloud)
        dense_feats: tensor (batch_size, num_points, num_dims). The features of the points before subsampling.
        nei_inds: tensor (batch_size, num_points2, K). The neighborhood indices of the K nearest neighbors of each point (after subsampling). The indices should index into dense_xyz and dense_feats,
                  as during subsampling features at new coordinates are aggregated from the points before subsampling
        dense_xyz_norm: tensor (batch_size, num_points, 3). The surface normals of the points before subsampling
        sparse_xyz: tensor (batch_size, num_points2, 3). The coordinates of the points after subsampling (if there is no subsampling, just input None for this and the next)
        sparse_xyz_norm: tensor (batch_size, num_points2, 3). The surface normals of the points after subsampling
        vi_features: tensor (batch_size, num_points2, 12). VI features only needs to be computed once per stage. If it has been computed in a previous layer,
                     it can be saved and directly inputted here.
        Note: batch_size is usually 1 since we are using the packed representation packing multiple point clouds into one. However this dimension needs to be there for pyTorch to work properly.
    Output:
        new_feat: output features
        weightNetInput: the input to weightNet, which are relative coordinates
                        or viewpoint-invariance aware transforms of it
    '''
    def __init__(self, in_channel, out_channel, 
                 USE_VI = True,
                 USE_CUDA_KERNEL = True,
                 weightnet=[9, 16], 
                 norm_layer = 'bn',
                 act_layer = torch.nn.LeakyReLU(0.1,inplace=True),
                 drop_out_rate = 0.0, 
                 drop_path_rate=0.0):
        super(PointConvResBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.USE_CUDA_KERNEL = USE_CUDA_KERNEL
        self.USE_VI = USE_VI

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.act_layer = act_layer
        self.norm_layer = norm_layer

        # positonal encoder
        self.pe_convs = WeightNet(
            3, min(out_channel // 4, 32), hidden_unit=[out_channel // 4], norm_layer = norm_layer, act_layer = act_layer, efficient=True)
        last_ch = min(out_channel // 4, 32)

        # First downscaling mlp
        if in_channel != out_channel // 4:
            self.unary1 = PointLinearLayer(
                in_channel,
                out_channel // 4,
                norm_layer = norm_layer, bn_ver = '1d')
        else:
            self.unary1 = nn.Identity()

        self.weightnet = WeightNet(weightnet[0], weightnet[1], norm_layer = norm_layer, act_layer = act_layer, efficient=True)
        if self.USE_CUDA_KERNEL:
            self.pconv_linear_opt = PConvLinearOpt((out_channel // 4 + last_ch) * weightnet[-1], out_channel // 2)
            # need to customly do BN without the fusion as of this CUDA kernel
            if self.norm_layer == 'bn':
                self.norm_layer = PermutedBN(out_channel // 2, momentum=0.1)
        else: 
            # we would have to do normalization separately because the CUDA kernel doesn't have normalization built in it
            self.linear = PointLinearLayer((out_channel // 4 + last_ch) * weightnet[-1], out_channel // 2, norm_layer=None, act_layer=None, bn_ver = '1d')

        self.dropout = nn.Dropout(
            p=drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # Second upscaling mlp
        self.unary2 = PointLinearLayer(out_channel // 2, out_channel, norm_layer = norm_layer, act_layer = None, bn_ver = '1d')

        # Shortcut optional mlp
        if in_channel != out_channel:
            self.unary_shortcut = PointLinearLayer(in_channel,out_channel,norm_layer = norm_layer, act_layer = None, bn_ver = '1d')
        else:
            self.unary_shortcut = nn.Identity()

        return

    def forward(
            self,
            dense_xyz,
            dense_feats,
            nei_inds,
            dense_xyz_norm=None,
            sparse_xyz=None,
            sparse_xyz_norm=None,
            vi_features=None,
            inv_neighbors=None,
            inv_k=None,
            inv_idx=None):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3), if None, then assume sparse_xyz = dense_xyz
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        dense_xyz_norm: tensor (batch_size, num_points, 3), used to compute VI features
        sparse_xyz_norm: tensor (batch_size, num_points2, 3), only when sparse_xyz is used
        vi_features: tensor (batch_size, num_points2, 12), if None, then compute VI features from dense_xyz_norm
        inv_neighbors: tensor (batch_size, num_points2, K), inverse neighbors for the CUDA kernel
        inv_k: tensor (batch_size, num_points2), inverse k for the CUDA kernel
        inv_idx: tensor (batch_size, num_points2, K), inverse indices for the CUDA kernel
        """
        no_batch = False
        # Deal with no batch dimension case
        if dense_xyz.dim() == 2:
            dense_xyz = dense_xyz.unsqueeze(0)
        if sparse_xyz is not None and len(sparse_xyz) > 0 and sparse_xyz.dim() == 2:
            sparse_xyz = sparse_xyz.unsqueeze(0)
        if dense_feats.dim() == 2:
            dense_feats = dense_feats.unsqueeze(0)
            no_batch = True
        if nei_inds.dim() == 2:
            nei_inds = nei_inds.unsqueeze(0)
        if dense_xyz_norm is not None and len(dense_xyz_norm) > 0 and dense_xyz_norm.dim() == 2:
            dense_xyz_norm = dense_xyz_norm.unsqueeze(0)
        if sparse_xyz_norm is not None and len(sparse_xyz_norm) > 0 and sparse_xyz_norm.dim() == 2:
            sparse_xyz_norm = sparse_xyz_norm.unsqueeze(0)
        if vi_features is not None and vi_features.dim() == 3:
            vi_features = vi_features.unsqueeze(0)
        B, N, _ = dense_xyz.shape
        if sparse_xyz is not None:
            _, M, _ = sparse_xyz.shape
        else:
            M = N
        _, _, K = nei_inds.shape

        # First downscaling mlp
        feats_x = self.unary1(dense_feats)

        gathered_xyz = index_points(dense_xyz, nei_inds)
        # localized_xyz = gathered_xyz - sparse_xyz.view(B, M, 1, D) #[B, M, K,
        # D]
        if sparse_xyz is not None:
            localized_xyz = gathered_xyz - sparse_xyz.unsqueeze(dim=2)
        else:
            localized_xyz = gathered_xyz - dense_xyz.unsqueeze(dim=2)

        feat_pe = self.pe_convs(localized_xyz)  # [B, M, K, D]

        if self.USE_VI is True:
            if dense_xyz_norm is None or len(dense_xyz_norm) == 0:
                raise AssertionError(
                    "dense_xyz_norm must be provided for VI features")
            gathered_norm = index_points(dense_xyz_norm, nei_inds)
            if vi_features is None:
                if sparse_xyz is not None:
                    weightNetInput = VI_coordinate_transform(
                        localized_xyz, gathered_norm, sparse_xyz_norm, K)
                else:
                    weightNetInput = VI_coordinate_transform(
                        localized_xyz, gathered_norm, dense_xyz_norm, K)
            else:
                weightNetInput = vi_features
        else:
            weightNetInput = localized_xyz

        # If not using CUDA kernel, then we need to sparse gather the features
        # here
        if not self.USE_CUDA_KERNEL:
            gathered_feat = index_points(feats_x, nei_inds)  # [B, M, K, in_ch]
            new_feat = torch.cat([gathered_feat, feat_pe], dim=-1)

        weights = self.weightnet(weightNetInput)

        if self.USE_CUDA_KERNEL:
            feats_x = feats_x.contiguous()
            # When the point cloud size drop under K, contiguous will make it int32
            # so we have to convert it back to avoid a bug
            nei_inds = nei_inds.contiguous().long()
            weights = weights.contiguous()
            feat_pe = feat_pe.contiguous()
            new_feat = self.pconv_linear_opt(feats_x,nei_inds,inv_neighbors, inv_k, inv_idx, 
                                            weights, feat_pe)        
        else:
            new_feat = torch.matmul(
                input=new_feat.permute(
                    0, 1, 3, 2), other=weights).view(
                B, M, -1)
        # In the fused CUDA kernel, we have already done the linear layer
        if not self.USE_CUDA_KERNEL:
            new_feat = self.linear(new_feat)
        if callable(self.norm_layer):
            new_feat = self.norm_layer(new_feat)

        # Dropout
        new_feat = self.dropout(new_feat)
        new_feat = self.act_layer(new_feat)

        # Second upscaling mlp
        new_feat = self.unary2(new_feat)
        if sparse_xyz is not None:
            sparse_feats = torch.max(
                index_points(
                    dense_feats,
                    nei_inds),
                dim=2)[0]
        else:
            sparse_feats = dense_feats

        shortcut = self.unary_shortcut(sparse_feats)

        new_feat = self.act_layer(self.drop_path(new_feat) + shortcut)

        if no_batch:
            new_feat = new_feat.squeeze(0)
            weightNetInput = weightNetInput.squeeze(0)
        return new_feat, weightNetInput

class PointConvSimple(nn.Module):
    '''
    This layer implements VI_PointConv and PointConv (set USE_VI = false) WITHOUT the bottleneck layer and without position encoding as features
    We use this only for the first layer, where input dimensionality is 3 and there is no point to use bottleneck
    Parameters:
        in_channel: Number of input channels
        out_channel: Number of output channels
        weightnet: Number of input/output channels for weightnet
        USE_VI: If not specified, then cfg.USE_VI is adopted, otherwise this overwrites cfg.USE_VI
    Input:
        dense_xyz: tensor (batch_size, num_points, 3). The coordinates of the points before subsampling (if it 
                   is a "strided" convolution wihch simultaneously subsamples the point cloud)
        dense_feats: tensor (batch_size, num_points, num_dims). The features of the points before subsampling.
        nei_inds: tensor (batch_size, num_points2, K). The neighborhood indices of the K nearest neighbors of 
                  each point (after subsampling). The indices should index into dense_xyz and dense_feats,
                  as during subsampling features at new coordinates are aggregated from the points before subsampling
        dense_xyz_norm: tensor (batch_size, num_points, 3). The surface normals of the points before subsampling
        sparse_xyz: tensor (batch_size, num_points2, 3). The coordinates of the points after subsampling (if there 
                    is no subsampling, just input None for this and the next)
        sparse_xyz_norm: tensor (batch_size, num_points2, 3). The surface normals of the points after subsampling
        vi_features: tensor (batch_size, num_points2, 12). VI features only needs to be computed once per stage. 
                     If it has been computed in a previous layer, it can be saved and directly inputted here.
        Note: batch_size is usually 1 since we are using the packed representation packing multiple point clouds into 
              one. However this dimension needs to be there for pyTorch to work properly.
    Output:
        new_feat: output features
        weightNetInput: the input to weightNet, which are relative coordinates or viewpoint-invariance aware transforms of it
    '''
    def __init__(
            self,
            in_channel,
            out_channel,
            weightnet=[9, 16],
            norm_layer = 'bn',
            act_layer = torch.nn.ReLU(inplace=True),
            USE_VI=False,
            USE_PE=False,
            USE_CUDA_KERNEL=True,
            dropout_rate=0.0):
        super(PointConvSimple, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.USE_VI = USE_VI
        self.USE_PE = USE_PE
        self.act_layer = act_layer
        last_ch = in_channel
        if USE_PE:
            if self.USE_VI:
                last_ch = in_channel + 12
            else:
                last_ch = in_channel + 3
        else:
            last_ch = in_channel

        self.weightnet = WeightNet(weightnet[0], weightnet[1], efficient=True)

        if norm_layer == 'bn':
            self.norm_layer = PermutedBN(out_channel)
        elif callable(norm_layer):
            self.norm_layer = norm_layer
        else:
            self.norm_layer = None

        if self.USE_CUDA_KERNEL:
            self.pconv_linear_opt = PConvLinearOpt(last_ch * weightnet[-1], out_channel)
        else:
            if norm_layer == 'bn':
                self.linear = Linear_BN(
                    last_ch * weightnet[-1], out_channel, bn_ver='1d')
            else:
                self.linear = nn.Linear(last_ch * weightnet[-1], out_channel)

        self.dropout = nn.Dropout(
            p=dropout_rate) if dropout_rate > 0. else nn.Identity()

    def forward(
            self,
            dense_xyz,
            dense_feats,
            nei_inds,
            dense_xyz_norm=None,
            sparse_xyz=None,
            sparse_xyz_norm=None,
            inv_neighbors=None,
            inv_k=None,
            inv_idx=None):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3)
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        dense_xyz_norm: normals of the dense xyz, tensor (batch_size, num_points, 3)
        sparse_xyz_norm: normals of the sparse xyz, tensor (batch_size, num_points2, 3)
        norms are required if USE_VI is true
        """
        no_batch = False
        # Deal with no batch dimension case
        if dense_xyz.dim() == 2:
            dense_xyz = dense_xyz.unsqueeze(0)
        if sparse_xyz is not None and len(sparse_xyz) > 0 and sparse_xyz.dim() == 2:
            sparse_xyz = sparse_xyz.unsqueeze(0)
        if dense_feats.dim() == 2:
            dense_feats = dense_feats.unsqueeze(0)
            no_batch = True
        if nei_inds.dim() == 2:
            nei_inds = nei_inds.unsqueeze(0)
        if dense_xyz_norm is not None and len(dense_xyz_norm) > 0 and dense_xyz_norm.dim() == 2:
            dense_xyz_norm = dense_xyz_norm.unsqueeze(0)
        if sparse_xyz_norm is not None and len(sparse_xyz_norm) > 0 and sparse_xyz_norm.dim() == 2:
            sparse_xyz_norm = sparse_xyz_norm.unsqueeze(0)
        B, N, _ = dense_xyz.shape
        if sparse_xyz is not None:
            _, M, _ = sparse_xyz.shape
        else:
            M = N
        _, _, K = nei_inds.shape

        # nei_inds = nei_inds.clone().detach()
        # nei_inds_mask = (nei_inds != -1).float()
        # nn_idx_divider = nei_inds_mask.sum(dim = -1)
        # nn_idx_divider[nn_idx_divider == 0] = 1
        # nei_inds[nei_inds == -1] = 0

        gathered_xyz = index_points(dense_xyz, nei_inds)
        # localized_xyz = gathered_xyz - sparse_xyz.view(B, M, 1, D) #[B, M, K,
        # D]
        if sparse_xyz is not None:
            localized_xyz = gathered_xyz - sparse_xyz.unsqueeze(dim=2)
        else:
            localized_xyz = gathered_xyz - dense_xyz.unsqueeze(dim=2)

        if self.USE_VI:
            gathered_norm = index_points(dense_xyz_norm, nei_inds)
            if sparse_xyz is not None:
                weightNetInput = VI_coordinate_transform(
                    localized_xyz, gathered_norm, sparse_xyz_norm, K)
            else:
                weightNetInput = VI_coordinate_transform(
                    localized_xyz, gathered_norm, dense_xyz_norm, K)
        else:
            weightNetInput = localized_xyz

        if self.USE_PE:
            additional_features = weightNetInput
            additional_features = additional_features.contiguous()
        else:
            additional_features = None

        weights = self.weightnet(weightNetInput)

        if self.USE_CUDA_KERNEL:
            dense_feats = dense_feats.contiguous()
            weights = weights.contiguous()
            # When the point cloud size drop under K, contiguous will make it int32
            # so we have to convert it back to avoid a bug
            nei_inds = nei_inds.contiguous().long()
            new_feat = self.pconv_linear_opt(
                                                dense_feats,
                                                nei_inds,
                                                inv_neighbors,
                                                inv_k,
                                                inv_idx,
                                                weights,
                                                additional_features
                                            )
        else:
            # Fallback
            gathered_feat = index_points(dense_feats, nei_inds)  # [B, M, K, in_ch]
            if self.USE_PE:
                gathered_feat = torch.cat([gathered_feat, weightNetInput], dim=-1)

            new_feat = torch.matmul(
                input=gathered_feat.permute(
                    0, 1, 3, 2), other=weights).view(
                B, M, -1)
            new_feat = self.linear(new_feat)

        if callable(self.norm_layer):
            new_feat = self.norm_layer(new_feat)

        # new_feat = F.relu(new_feat, inplace=True)
        new_feat = self.act_layer(new_feat)

        # Dropout
        new_feat = self.dropout(new_feat)

        if no_batch:
            new_feat = new_feat.squeeze(0)
            weightNetInput = weightNetInput.squeeze(0)
        return new_feat, weightNetInput

class PointConvTranspose(nn.Module):
    '''
    PointConvTranspose (upsampling) layer
    one needs to input dense_xyz (high resolution point coordinates after upsampling) and sparse_xyz (low-resolution) 
    and this layer would put features to the points at dense_xyz
    Parameters:
        in_channel: Number of input channels
        out_channel: Number of output channels
        weightnet: Number of input/output channels for weightnet
        mlp2: MLP after the PointConvTranspose

    Input:
        sparse_xyz: tensor (batch_size, num_points, 3). The coordinates of the points before upsampling
        sparse_feats: tensor (batch_size, num_points, num_dims). The features of the points before upsampling.
        nei_inds: tensor (batch_size, num_points2, K). The neighborhood indices of the K nearest neighbors of each 
                  point after upsampling. The indices should index into sparse_xyz and sparse_feats,
                  as during upsampling features at new coordinates are aggregated from the points before upsampling
        sparse_xyz_norm: tensor (batch_size, num_points, 3). The surface normals of the points before upsampling
        dense_xyz: tensor (batch_size, num_points2, 3). The coordinates of the points after upsampling (if there is no 
                   upsampling, just input None for this and the next)
        dense_xyz_norm: tensor (batch_size, num_points2, 3). The surface normals of the points after upsampling
        dense_feats: shortcut dense features
        vi_features: tensor (batch_size, num_points2, 12). VI features only needs to be computed once per stage. If it 
                     has been computed in a previous layer, it can be saved and directly inputted here.
        Note: batch_size is usually 1 since we are using the packed representation packing multiple point clouds into 
              one. However this dimension needs to be there for pyTorch to work properly.
    Output:
        new_feat: output features
        weightNetInput: the input to weightNet, which are relative coordinates or viewpoint-invariance aware transforms of it
    '''
    def __init__(
            self,
            in_channel,
            out_channel,
            weightnet=[9, 16],
            dropout_rate = 0.0,
            drop_path_rate=0.0,
            norm_layer = 'bn',
            USE_PE = True,
            USE_VI = True,
            USE_CUDA_KERNEL = True,
            mlp2=None):
        super(PointConvTranspose, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.USE_PE = USE_PE
        self.USE_VI = USE_VI
        self.USE_CUDA_KERNEL = USE_CUDA_KERNEL
        self.norm_layer = norm_layer

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
# This part can save a bit of memory, maybe with some performance drop or maybe no drop at all
#        self.unary1 = UnaryBlock(
#            in_channel,
#            out_channel,
#            use_bn=True,
#            bn_momentum=0.1)

        # positonal encoder
        self.pe_convs = nn.ModuleList()
        if self.USE_PE:
            self.pe_convs = WeightNet(
                3, min(out_channel // 4, 32), hidden_unit=[out_channel // 4], efficient=True)
            last_ch = min(out_channel // 4, 32)
        else:
            self.pe_convs = nn.ModuleList()
            last_ch = 0

        self.weightnet = WeightNet(weightnet[0], weightnet[1], efficient=True)

        if self.USE_CUDA_KERNEL:
            self.pconv_linear_opt = PConvLinearOpt((last_ch + in_channel) * weightnet[-1], out_channel)
            if self.norm_layer == 'bn':
                self.norm_layer = PermutedBN(out_channel, momentum=0.1)
        else:
            self.linear = nn.Linear((last_ch + in_channel) * weightnet[-1], out_channel)
                # self.linear = Linear_BN(
                #                 (last_ch + out_channel) * weightnet[-1], out_channel, bn_ver='1d')
    #            self.linear = nn.Linear(
    #                (last_ch + out_channel) * weightnet[-1], out_channel)

        self.dropout = nn.Dropout(
            p=dropout_rate) if dropout_rate > 0. else nn.Identity()

        self.mlp2_convs = nn.ModuleList()
        if mlp2 is not None:
            for i in range(1, len(mlp2)):
                if self.norm_layer == 'bn':
                    self.mlp2_convs.append(
                        Linear_BN(mlp2[i - 1], mlp2[i], bn_ver='1d'))
                else:
                    self.mlp2_convs.append(nn.Linear(mlp2[i - 1], mlp2[i]))

    def forward(
            self,
            sparse_xyz,
            sparse_feats,
            nei_inds,
            sparse_xyz_norm,
            dense_xyz,
            dense_xyz_norm,
            dense_feats=None,
            vi_features=None,
            inv_neighbors=None,
            inv_k=None,
            inv_idx=None
            ):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3)
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        sparse_xyz_norm: tensor (batch_size, num_points2, 3)
        dense_xyz_norm: tensor (batch_size, num_points, 3)
        norms are required if USE_VI is true
        inv_neighbors: tensor (batch_size, num_points2, K), inverse neighbors for the CUDA kernel
        inv_k: tensor (batch_size, num_points2), inverse k for the CUDA kernel
        inv_idx: tensor (batch_size, num_points2, K), inverse indices for the CUDA kernel
        """
        no_batch = False
        # Deal with no batch dimension case
        if dense_xyz.dim() == 2:
            dense_xyz = dense_xyz.unsqueeze(0)
        if sparse_xyz.dim() == 2:
            sparse_xyz = sparse_xyz.unsqueeze(0)
        if dense_feats is not None and len(dense_feats) > 0 and dense_feats.dim() == 2:
            dense_feats = dense_feats.unsqueeze(0)
        if sparse_feats.dim() == 2:
            sparse_feats = sparse_feats.unsqueeze(0)
            no_batch = True
        if nei_inds.dim() == 2:
            nei_inds = nei_inds.unsqueeze(0)
        if sparse_xyz_norm is not None and len(sparse_xyz_norm) > 0 and sparse_xyz_norm.dim() == 2:
            sparse_xyz_norm = sparse_xyz_norm.unsqueeze(0)
        if dense_xyz_norm is not None and len(dense_xyz_norm) > 0 and dense_xyz_norm.dim() == 2:
            dense_xyz_norm = dense_xyz_norm.unsqueeze(0)
        if vi_features is not None and vi_features.dim() == 3:
            vi_features = vi_features.unsqueeze(0)
        B, _, _ = sparse_xyz.shape
        _, M, _ = dense_xyz.shape
        _, _, K = nei_inds.shape

        gathered_xyz = index_points(sparse_xyz, nei_inds)
        localized_xyz = gathered_xyz - dense_xyz.unsqueeze(dim=2)
        gathered_norm = index_points(sparse_xyz_norm, nei_inds)

        if self.USE_PE:
            feat_pe = self.pe_convs(localized_xyz)
        if self.USE_VI is True:
            if vi_features is None:
                weightNetInput = VI_coordinate_transform(
                    localized_xyz, gathered_norm, dense_xyz_norm, K)
            else:
                weightNetInput = vi_features
        else:
            weightNetInput = localized_xyz

        # feats_x = self.unary1(sparse_feats)
        feats_x = sparse_feats

        weights = self.weightnet(weightNetInput)

        if not self.USE_CUDA_KERNEL:
            gathered_feat = index_points(feats_x, nei_inds)  # [B, M, K, in_ch]
            if self.USE_PE:
                gathered_feat = torch.cat([gathered_feat, feat_pe], dim=-1)

        if self.USE_CUDA_KERNEL:
            feats_x = feats_x.contiguous()

            # When the point cloud size drop under K, contiguous will make it int32
            # so we have to convert it back to avoid a bug
            nei_inds = nei_inds.contiguous().long()
            weights = weights.contiguous()

            if self.USE_PE:
                feat_pe = feat_pe.contiguous()
                new_feat = self.pconv_linear_opt(feats_x,nei_inds, inv_neighbors, inv_k, 
                                                        inv_idx, weights, feat_pe)
            else:
                new_feat = self.pconv_linear_opt(feats_x, nei_inds, inv_neighbors, inv_k, 
                                                        inv_idx, weights)
        else:
            new_feat = torch.matmul(
                input=gathered_feat.permute(
                    0, 1, 3, 2), other=weights).view(
                B, M, -1)
            new_feat = self.linear(new_feat)

        if callable(self.norm_layer):
            new_feat = self.norm_layer(new_feat)

        new_feat = F.relu(new_feat, inplace=True)

        if dense_feats is not None:
            new_feat = new_feat + dense_feats

        # Dropout
        new_feat = self.dropout(new_feat)

        for conv in self.mlp2_convs:
            new_feat = F.relu(conv(new_feat), inplace=True)

        if no_batch is True:
            new_feat = new_feat.squeeze(0)
            weightNetInput = weightNetInput.squeeze(0)
        if len(torch.nonzero(torch.isnan(new_feat) | torch.isinf(new_feat))):
            print('Infs/NaNs found in PointConvTranspose layer')
        return new_feat, weightNetInput
