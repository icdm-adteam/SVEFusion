import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from scipy.spatial import cKDTree

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate


class HeteroAlign(nn.Module):
    def __init__(self, lidar_dim, radar_dim, ada_dim=64, proj_dim=256):
        super().__init__()
        self.lidar_adapter  = nn.Sequential(
            nn.Linear(lidar_dim, ada_dim),
            nn.GELU()
        )
        self.radar_adapter  = nn.Linear(radar_dim, ada_dim)
        
        self.shared_proj  = nn.Linear(ada_dim, proj_dim)
        
    def forward(self, lidar, radar):
        lidar = self.lidar_adapter(lidar) 
        radar = self.radar_adapter(radar) 
        return self.shared_proj(lidar),  self.shared_proj(radar) 


class SNA(nn.Module):
    '''
    An attention mechanism that combines the information of two modalities 
    using neighborhood-based sparse attention.
    '''
    def __init__(self, uni_channels, channels):
        super(SNA, self).__init__()
        self.linear = nn.Linear(uni_channels, channels, bias=True)
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, neighbors_idx):
        x = self.linear(x).permute(0, 2, 1)
        y = self.linear(y).permute(0, 2, 1)
        
        x_q = self.q_conv(x).permute(2, 0, 1)
        y_k = self.k_conv(y).permute(2, 1, 0)
        y_v = self.v_conv(y).permute(2, 0, 1)

        y_k_sparse = torch.index_select(y_k, 2, neighbors_idx.view(-1)).view(y_k.shape[0], y_k.shape[1], neighbors_idx.shape[0], neighbors_idx.shape[1]).permute(2, 0, 1, 3)
        y_v_sparse = torch.index_select(y_v, 1, neighbors_idx.view(-1)).view(y_v.shape[0], neighbors_idx.shape[0], neighbors_idx.shape[1], y_v.shape[2]).permute(1, 0, 2, 3)

        energy = torch.einsum('pmc,mpck->pmk', x_q, y_k_sparse)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))

        y_r = torch.einsum('pmk,mpkc->pmc', attention, y_v_sparse)
        y_r = y_r.permute(1, 2, 0)

        y_r = self.act(self.after_norm(self.trans_conv(x - y_r)))
        x = x + y_r
        x = torch.max(x, dim=2, keepdim=True)[0]
        return x
    

class SNAFusion(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features_l = num_point_features[0]
        num_point_features_r = num_point_features[1]
        self.use_decomposed_velocity = self.model_cfg.USE_DECOMPOSED_VELOCITY

        num_point_features_l += 6 if self.use_absolute_xyz else 3
        num_point_features_r += 5 if self.use_absolute_xyz else 2

        if self.with_distance:
            num_point_features_l += 1
            num_point_features_r += 1

        if self.use_decomposed_velocity:
            num_point_features_r += 8

        self.uniform_features = self.model_cfg.UNIFORM_FEATURES

        self.lr_align = HeteroAlign(num_point_features_l, num_point_features_r, self.uniform_features, self.uniform_features * 4)
        
        self.r2l_neighbor_num = self.model_cfg.R2L_NEIGHBOR_NUM
        self.l2r_neighbor_num = self.model_cfg.L2R_NEIGHBOR_NUM

        self.vfe_dim = self.model_cfg.VFE_DIM
        self.inter_ral = SNA(self.uniform_features * 4, self.vfe_dim)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

        self.num_point_features_r = num_point_features_r
        self.encode_transform = nn.Linear(num_point_features_r, num_point_features_r)

    def get_output_feature_dim(self):
        return self.vfe_dim * len(self.r2l_neighbor_num)
    
    def time_embedding(self, t, embedding_dim):
        time_encoding = torch.zeros(t.shape[0], embedding_dim, device=t.device)

        for i, timestamp in enumerate(t):
            for j in range(0, embedding_dim, 2):
                time_encoding[i, j] = math.sin(timestamp * math.pow(10000, -j/embedding_dim))
                if j + 1 < embedding_dim:
                    time_encoding[i, j+1] = math.cos(timestamp * math.pow(10000, -(j+1)/embedding_dim))
        return time_encoding
    
    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        Args:
            actual_num: number of actual points per voxel
            max_num: the maximum number of voxel points
        Returns:
            paddings_indicator: Determine whether the data in the pillar is the real data or the filled value 0
        """

        # Extending a dimension
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def compute_voxel_neighbors(self, lidar_voxel_coords, radar_voxel_coords, k_neighbors=5):
        lidar_coords = lidar_voxel_coords[:, 1:]  # [M, 3]
        radar_coords = radar_voxel_coords[:, 1:]   # [N, 3]
        
        lidar_batch = lidar_voxel_coords[:, 0]     # [M]
        radar_batch = radar_voxel_coords[:, 0]     # [N]

        neighbors_idx = torch.full((len(lidar_coords), k_neighbors), -1, 
                            dtype=torch.long, device=lidar_voxel_coords.device)

        for batch_id in torch.unique(lidar_batch):
            # Mask for current batch
            lidar_mask = lidar_batch == batch_id
            radar_mask = radar_batch == batch_id
            
            radar_global_indices = torch.where(radar_mask)[0].cpu().numpy()
            
            batch_lidar = lidar_coords[lidar_mask].cpu().numpy()
            batch_radar = radar_coords[radar_mask].cpu().numpy()

            radar_tree = cKDTree(batch_radar)
            distances, local_indices = radar_tree.query(batch_lidar, k=k_neighbors)
            
            global_indices = np.full(local_indices.shape, -1)
            global_indices = radar_global_indices[local_indices]
            
            neighbors_idx[lidar_mask] = torch.tensor(global_indices, 
                                                device=lidar_voxel_coords.device)

        return neighbors_idx

    def forward(self, batch_dict, **kwargs):
        lidar_voxel_features, lidar_voxel_num_points, lidar_coords = batch_dict['lidar_voxels'], batch_dict['lidar_voxel_num_points'], batch_dict['lidar_voxel_coords']
        radar_voxel_features, radar_voxel_num_points, radar_coords = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points'], batch_dict['radar_voxel_coords']

        orig_xyz_l = lidar_voxel_features[:, :, :3]  # selecting x y z
        orig_xyz_r = radar_voxel_features[:, :, :3]  # selecting x y z

        lidar_points_mean = lidar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / lidar_voxel_num_points.type_as(lidar_voxel_features).view(-1, 1, 1)
        radar_points_mean = radar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / radar_voxel_num_points.type_as(radar_voxel_features).view(-1, 1, 1)
        lidar_f_cluster = lidar_voxel_features[:, :, :3] - lidar_points_mean
        radar_f_cluster = radar_voxel_features[:, :, :3] - radar_points_mean

        lidar_f_center = torch.zeros_like(lidar_voxel_features[:, :, :3])
        radar_f_center = torch.zeros_like(radar_voxel_features[:, :, :3])
        lidar_f_center[:, :, 0] = lidar_voxel_features[:, :, 0] - (lidar_coords[:, 3].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        lidar_f_center[:, :, 1] = lidar_voxel_features[:, :, 1] - (lidar_coords[:, 2].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        lidar_f_center[:, :, 2] = lidar_voxel_features[:, :, 2] - (lidar_coords[:, 1].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        radar_f_center[:, :, 0] = radar_voxel_features[:, :, 0] - (radar_coords[:, 3].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        radar_f_center[:, :, 1] = radar_voxel_features[:, :, 1] - (radar_coords[:, 2].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        radar_f_center[:, :, 2] = radar_voxel_features[:, :, 2] - (radar_coords[:, 1].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.with_distance:
            points_dist_l = torch.norm(orig_xyz_l, 2, 2, keepdim=True)
            points_dist_r = torch.norm(orig_xyz_r, 2, 2, keepdim=True)

        if self.use_decomposed_velocity:
            v_r = radar_voxel_features[:, :, 4]
            v_r_compensated = radar_voxel_features[:, :, 5]

            beta = torch.atan2(radar_voxel_features[:, :, 1], radar_voxel_features[:, :, 0])  # β = arctan(y / x)
            v_x = torch.cos(beta) * v_r
            v_y = torch.sin(beta) * v_r
            v_x_compensated = torch.cos(beta) * v_r_compensated
            v_y_compensated = torch.sin(beta) * v_r_compensated

            v_features = torch.stack([v_x, v_y, v_x_compensated, v_y_compensated], dim=-1)
            v_features_mean = v_features.sum(dim=1, keepdim=True) / radar_voxel_num_points.type_as(radar_voxel_features).view(-1, 1, 1)
            v_features_diff = v_features - v_features_mean

        if self.use_absolute_xyz:
            lidar_features = [lidar_voxel_features, lidar_f_cluster, lidar_f_center]
            radar_features = [radar_voxel_features[:, :, :-1], radar_f_cluster, radar_f_center]
        else:
            lidar_features = [lidar_voxel_features[:, :, 3:], lidar_f_cluster, lidar_f_center]
            radar_features = [radar_voxel_features[:, :, 3:-1], radar_f_cluster, radar_f_center]

        if self.with_distance:
            lidar_features.append(points_dist_l)
            radar_features.append(points_dist_r)
        
        if self.use_decomposed_velocity:
            radar_features.append(v_features)
            radar_features.append(v_features_diff)

        lidar_features = torch.cat(lidar_features, dim=-1)
        radar_features = torch.cat(radar_features, dim=-1)

        # cosine embedding for time
        t = [-4.0, -3.0, -2.0, -1.0, 0.0]
        t = torch.tensor(t, dtype=torch.float32).cuda()
        time_encoding = self.time_embedding(t, self.num_point_features_r)
        time_encoding = self.encode_transform(time_encoding)

        time = radar_voxel_features[:, :, -1].squeeze(-1).view(-1)
        time_encoding = time_encoding[time.int() + 4].view(radar_features.shape[0], radar_features.shape[1], -1)
        radar_features = radar_features + time_encoding

        lidar_features, radar_features = self.lr_align(lidar_features, radar_features)

        lidar_voxel_count = lidar_features.shape[1]
        radar_voxel_count = radar_features.shape[1]
        lidar_mask = self.get_paddings_indicator(lidar_voxel_num_points, lidar_voxel_count, axis=0)
        radar_mask = self.get_paddings_indicator(radar_voxel_num_points, radar_voxel_count, axis=0)
        lidar_mask = torch.unsqueeze(lidar_mask, -1).type_as(lidar_voxel_features)
        radar_mask = torch.unsqueeze(radar_mask, -1).type_as(radar_voxel_features)
        lidar_features *= lidar_mask
        radar_features *= radar_mask

        assert len(self.r2l_neighbor_num) == len(self.l2r_neighbor_num)
        lidar_features_output_list = []
        radar_features_output_list = []
        for i in range(len(self.r2l_neighbor_num)):
            neighbors_idx = self.compute_voxel_neighbors(lidar_coords, radar_coords, k_neighbors=self.r2l_neighbor_num[i])
            lidar_features_output = self.inter_ral(lidar_features, radar_features, neighbors_idx)
            lidar_features_output = lidar_features_output.view([lidar_features_output.size()[0], lidar_features_output.size()[1]])
            lidar_features_output_list.append(lidar_features_output)

            neighbors_idx = self.compute_voxel_neighbors(radar_coords, lidar_coords, k_neighbors=self.l2r_neighbor_num[i])
            radar_features_output = self.inter_ral(radar_features, lidar_features, neighbors_idx)
            radar_features_output = radar_features_output.view([radar_features_output.size()[0], radar_features_output.size()[1]])
            radar_features_output_list.append(radar_features_output)

        lidar_features_output = torch.cat(lidar_features_output_list, dim=-1)
        radar_features_output = torch.cat(radar_features_output_list, dim=-1)

        batch_dict['voxel_size'] = self.voxel_size

        batch_dict['lidar_features'] = lidar_features_output
        batch_dict['radar_features'] = radar_features_output

        return batch_dict