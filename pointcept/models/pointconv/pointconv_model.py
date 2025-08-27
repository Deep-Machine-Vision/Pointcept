from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
from pointcept.models.utils.knn import compute_knn
from pointcept.models.utils.sampling import grid_sampling
from pointcept.models.modules import PointModule
from .pointconv import PointLinearLayer, PointConvResBlock, PointConvTranspose
import torch.nn as nn
import torch

try:
    import pcf_cuda
except ImportError:
    print("PCF CUDA kernel not available, using normal pyTorch version.")
    pcf_cuda = None


@MODELS.register_module("PointConvEncoder")
class PointConv_Encoder(PointModule):
    """
    Base class for PointConv backbone models.
    """
    def __init__(self, 
                 in_channels, # Number of input feature channels
                 point_dim=3, # Dimensionality of the point cloud
                 enc_depths = [2, 4, 6, 6, 2],
                 enc_channels=[32, 64, 128, 256, 512],
                 enc_patch_size = [16, 16, 16, 16, 16],
                 USE_PE=True, 
                 USE_VI=True, 
                 USE_CUDA_KERNEL=True,
                 weightnet_middim = [4, 4, 4, 4, 4],
                 act_layer = torch.nn.LeakyReLU(0.1, inplace=True),
                 norm_layer='bn', 
                 drop_out_rate=0.0, 
                 drop_path_rate=0.0
                 ):
        super(PointConv_Encoder, self).__init__()
        self.in_channels = in_channels
        self.USE_PE = USE_PE
        self.USE_VI = USE_VI
        self.enc_patch_size = enc_patch_size
        if pcf_cuda is not None and USE_CUDA_KERNEL:
            self.USE_CUDA_KERNEL = USE_CUDA_KERNEL
        else:
            self.USE_CUDA_KERNEL = False
        self.norm_layer = norm_layer
        self.drop_out_rate = drop_out_rate
        self.drop_path_rate = drop_path_rate

        if USE_VI is True:
            weightnet_input_dim = point_dim + 9
        else:
            weightnet_input_dim = point_dim
        weightnet = [weightnet_input_dim, weightnet_middim[0]]  # 2 hidden layers

        self.embedding = PointLinearLayer(
            in_channels=in_channels,
            out_channels=enc_channels[0],
            norm_layer=norm_layer,
            act_layer=act_layer, bn_ver='1d'
        )
        self.pointconv = nn.ModuleList()
        self.pointconv_res = nn.ModuleList()

        for i in range(1, len(enc_depths)):
            in_ch = enc_channels[i - 1]
            out_ch = enc_channels[i]
            weightnet = [weightnet_input_dim, weightnet_middim[i]]
            # Downsampling PointConv
            self.pointconv.append(
                    PointConvResBlock(
                        in_ch, out_ch, USE_VI, self.USE_CUDA_KERNEL, weightnet,norm_layer, act_layer,drop_out_rate, drop_path_rate))

            if enc_depths[i] == 0:
                self.pointconv_res.append(nn.ModuleList([]))
            else:
                res_blocks = nn.ModuleList()
                for _ in range(enc_depths[i]):
                    res_blocks.append(
                            PointConvResBlock(
                                out_ch, out_ch, USE_VI, self.USE_CUDA_KERNEL, weightnet, norm_layer, act_layer))
                self.pointconv_res.append(res_blocks)

    def forward(self, point):
        """
        Forward pass of the PointConv encoder.
        Args:
            point (Point): Input point cloud with features.
        Returns:
            point_list (list): List of Point objects after each encoding stage with features of each point cloud level computed.
        """
        point.coord = point.coord.float()
        
         # Initial embedding
        point.feat = self.embedding(point.feat)

        point_list = [point]

        if not hasattr(point, "neighbors") or len(point.neighbors) == 0:
            point.neighbors = compute_knn(
                point.coord, point.coord, K=self.enc_patch_size[0])


        for i, pointconv in enumerate(self.pointconv):
            # Downsampling (stride-2) PointConv
            down_point = grid_sampling(point, reduce='mean')
            down_point.neighbors = compute_knn(
                down_point.coord, down_point.coord, K=self.enc_patch_size[i + 1])
            if not hasattr(point, "neighbors_down") or len(point.neighbors_down) == 0:
                point.neighbors_down = compute_knn(
                    point.coord, down_point.coord, K=self.enc_patch_size[i])
            if self.USE_CUDA_KERNEL:
                add_dim_neighbors = point.neighbors_down.unsqueeze(0)
                inv_n, inv_k, inv_idx = pcf_cuda.compute_knn_inverse(add_dim_neighbors, point.coord.shape[0])
                inv_fwd_args = {
                                    "inv_neighbors": inv_n,
                                    "inv_k": inv_k,
                                    "inv_idx": inv_idx
                                }
                down_point.feat, _ = pointconv(
                    point.coord, point.feat, point.neighbors_down, point.normal, down_point.coord, down_point.normal, **inv_fwd_args)
            else:
                down_point.feat, _ = pointconv(
                    point.coord, point.feat, point.neighbors_down, point.normal, down_point.coord, down_point.normal)
            # print(sparse_feat.shape)
            # There is the need to recompute VI features from the neighbors at this level rather than from the previous level, hence need
            # to recompute VI features in the first residual block
            vi_features = None
            if self.USE_CUDA_KERNEL:
                add_dim_neighbors = down_point.neighbors.unsqueeze(0)
                print(down_point.coord.shape)
                inv_n, inv_k, inv_idx = pcf_cuda.compute_knn_inverse(add_dim_neighbors, down_point.coord.shape[0])

            for res_block in self.pointconv_res[i]:
                inv_self_args = {}
                if self.USE_CUDA_KERNEL:
                    inv_self_args = {
                                        "inv_neighbors": inv_n,
                                        "inv_k": inv_k,
                                        "inv_idx": inv_idx
                                    }
                if vi_features is not None:
                    down_point.feat, _ = res_block(
                        down_point.coord, down_point.feat, down_point.neighbors, down_point.normal, vi_features=vi_features, **inv_self_args)
                else:
                    down_point.feat, vi_features = res_block(
                        down_point.coord, down_point.feat, down_point.neighbors, down_point.normal, **inv_self_args)
            point_list.append(down_point)
            point = down_point

        return point_list

@MODELS.register_module("PointConvDecoder")
class PointConv_Decoder(PointModule):
    """
    Base class for PointConv decoder.
    """
    def __init__(self, 
                 point_dim=3, # Dimensionality of the point cloud
                 dec_depths=(0, 0, 0, 0, 0),
                 dec_channels=(32, 64, 128, 192, 256),
                 dec_patch_size=(16, 16, 16, 16),
                 USE_VI=True, 
                 USE_CUDA_KERNEL=True,
                 weightnet_middim=[4, 4, 4, 4],
                 act_layer=torch.nn.LeakyReLU(0.1, inplace=True),
                 norm_layer='bn', 
                 drop_out_rate=0.0, 
                 drop_path_rate=0.0
                 ):
        super(PointConv_Decoder, self).__init__()
        self.point_dim = point_dim
        self.dec_patch_size = dec_patch_size
        self.USE_VI = USE_VI
        self.USE_CUDA_KERNEL = USE_CUDA_KERNEL if pcf_cuda is not None and USE_CUDA_KERNEL else False
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.drop_out_rate = drop_out_rate
        self.drop_path_rate = drop_path_rate

        self.pointconv_transpose = nn.ModuleList()
        self.pointconv_res = nn.ModuleList()

        if USE_VI is True:
            weightnet_input_dim = point_dim + 9
        else:
            weightnet_input_dim = point_dim
        for i in reversed(range(1, len(dec_depths))):
            in_ch = dec_channels[i]
            out_ch = dec_channels[i - 1]
            weightnet = [weightnet_input_dim, weightnet_middim[i-1]]
            # Upsampling PointConvTranspose
            self.pointconv_transpose.append(
                PointConvTranspose(in_ch, out_ch, weightnet, 
                                    dropout_rate = 0.0,
                                    drop_path_rate=0.0,
                                    norm_layer = norm_layer,
                                    USE_PE = True,
                                    USE_VI = USE_VI,
                                    USE_CUDA_KERNEL = self.USE_CUDA_KERNEL
                )
            )
            if dec_depths[i - 1] == 0:
                self.pointconv_res.append(nn.ModuleList([]))
            else:
                res_blocks = nn.ModuleList()
                for _ in range(dec_depths[i - 1]):
                    res_blocks.append(
                        PointConvResBlock(
                            out_ch, out_ch, USE_VI, self.USE_CUDA_KERNEL, weightnet, norm_layer, act_layer
                        )
                    )
                self.pointconv_res.append(res_blocks)
    
        # pointwise_decode
        self.dropout = torch.nn.Dropout(
            p=drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, point_list):
        """
        Forward pass of the PointConv decoder.
        Args:
            point_list (list): List of Point objects from encoder (fine to coarse).
        Returns:
            torch.Tensor: Output tensor after decoding.
        """
        # Start from the coarsest point cloud
        point = point_list[-1]

        for i, pointconv_transpose in enumerate(self.pointconv_transpose):
            up_point = point_list[len(point_list) - i - 2]
            # Upsampling (stride-2) PointConvTranspose
            if not hasattr(point, "neighbors_up") or point.neighbors_up is None or len(point.neighbors_up) == 0:
                point.neighbors_up = compute_knn(
                    point.coord, up_point.coord, K=self.dec_patch_size[len(self.dec_patch_size) - i - 1])
            if self.USE_CUDA_KERNEL:
                add_dim_neighbors = point.neighbors_up.unsqueeze(0)
                inv_n, inv_k, inv_idx = pcf_cuda.compute_knn_inverse(add_dim_neighbors, up_point.coord.shape[0])
                inv_fwd_args = {
                    "inv_neighbors": inv_n,
                    "inv_k": inv_k,
                    "inv_idx": inv_idx
                }
                dense_feat, _ = pointconv_transpose(
                    point.coord, point.feat, point.neighbors_up, point.normal, up_point.coord, up_point.normal, up_point.feat, **inv_fwd_args)
            else:
                dense_feat, _ = pointconv_transpose(
                    point.coord, point.feat, point.neighbors_up, point.normal, up_point.coord, up_point.normal, up_point.feat)

            up_point.feat = dense_feat

            vi_features = None
            if self.USE_CUDA_KERNEL:
                add_dim_neighbors = up_point.neighbors.unsqueeze(0)
                inv_n, inv_k, inv_idx = pcf_cuda.compute_knn_inverse(add_dim_neighbors, up_point.coord.shape[0])

            for res_block in self.pointconv_res[i]:
                inv_self_args = {}
                if self.USE_CUDA_KERNEL:
                    inv_self_args = {
                        "inv_neighbors": inv_n,
                        "inv_k": inv_k,
                        "inv_idx": inv_idx
                    }
                if vi_features is not None:
                    up_point.feat, _ = res_block(
                        up_point.coord, up_point.feat, up_point.neighbors, up_point.normal, vi_features=vi_features, **inv_self_args)
                else:
                    up_point.feat, vi_features = res_block(
                        up_point.coord, up_point.feat, up_point.neighbors, up_point.normal, **inv_self_args)
            point = up_point

        return point

@MODELS.register_module("PointConvUNet")
class PointConvUNet(PointModule):
    """
    PointConv UNet model combining encoder and decoder.
    """
    def __init__(self,                 
                 in_channels, # Number of input feature channels
                 point_dim=3, # Dimensionality of the point cloud
                 enc_depths = [2, 2, 2, 6, 2],
                 enc_channels=[32, 64, 128, 256, 512],
                 enc_patch_size = [16, 16, 16, 16, 16],
                 dec_depths=(0,0, 0, 0),
                 dec_channels=(32, 64, 128, 192, 256),
                 dec_patch_size=(16, 16, 16, 16),
                 USE_PE=True, 
                 USE_VI=True, 
                 USE_CUDA_KERNEL=True,
                 weightnet_middim = [4,4,4,4,4],
                 act_layer = torch.nn.LeakyReLU(0.1, inplace=True),
                 norm_layer='bn', 
                 drop_out_rate=0.0, 
                 drop_path_rate=0.0):
        super(PointConvUNet, self).__init__()
        self.encoder = PointConv_Encoder(
            in_channels=in_channels,
            point_dim=point_dim,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_patch_size=enc_patch_size,
            USE_PE=USE_PE,
            USE_VI=USE_VI,
            USE_CUDA_KERNEL=USE_CUDA_KERNEL,
            weightnet_middim=weightnet_middim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_out_rate=drop_out_rate,
            drop_path_rate=drop_path_rate
        )
        self.decoder = PointConv_Decoder(
            point_dim=point_dim,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_patch_size=dec_patch_size,
            USE_VI=USE_VI,
            USE_CUDA_KERNEL=USE_CUDA_KERNEL,
            weightnet_middim=weightnet_middim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_out_rate=drop_out_rate,
            drop_path_rate=drop_path_rate
        )

    def forward(self, point):
        """
        Forward pass of the PointConv UNet.
        Args:
            point (Point): Input point cloud with features.
        Returns:
            output (Point): Output featurized point cloud after encoding and decoding.
        """
        point_list = self.encoder(point)
        output = self.decoder(point_list)
        return output