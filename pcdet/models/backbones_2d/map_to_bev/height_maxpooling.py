import torch
import torch.nn as nn
from ....utils.spconv_utils import replace_feature

class HeightMaxPooling(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.voxel_features_mlp_cfg = self.model_cfg.VOXEL_FEATURES_MLP
        
        voxel_features_mlp_layers = []
        c_in = 64
        for k in range(0, self.voxel_features_mlp_cfg.__len__()):
            voxel_features_mlp_layers.extend([
                nn.Linear(c_in, self.voxel_features_mlp_cfg[k], bias=False),
                nn.BatchNorm1d(self.voxel_features_mlp_cfg[k]),
                nn.ReLU(),
            ])
            c_in = self.voxel_features_mlp_cfg[k]
        self.voxel_features_mlp = nn.Sequential(*voxel_features_mlp_layers)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        encoded_spconv_tensor = replace_feature(encoded_spconv_tensor, self.voxel_features_mlp(encoded_spconv_tensor.features))

        spatial_features = encoded_spconv_tensor.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.max(dim=2)[0] # (N, C, H, W)

        batch_dict['spatial_features'] = spatial_features

        return batch_dict