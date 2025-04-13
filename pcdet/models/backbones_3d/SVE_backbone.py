from functools import partial

import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
from pcdet.utils.loss_utils import SigmoidFocalClassificationLoss
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from ...utils import common_utils

from tools.svefusion_utils import SVSO, MambaBlock, Permute

from spconv.pytorch import functional as Fsp
from cumm.gemm.layout import to_stride
from typing import List
import numpy as np


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SVEBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.point_cloud_range = point_cloud_range
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 64, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(64),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        self.conv2 = spconv.SparseSequential(
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )
        self.conv3 = spconv.SparseSequential(
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        self.conv4 = spconv.SparseSequential(
            block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        self.svso = SVSO(features_in=256, down_mlp_channels=[256, 64, 1])
        self.pos_embed = nn.Sequential(
            nn.Linear(9, 128),
            Permute(0, 2, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            Permute(0, 2, 1),
            nn.Linear(128, 256),
        )
        self.vmamba = nn.ModuleList([
            MambaBlock(
                d_model=256,
                ssm_cfg=None,
                norm_epsilon=0.00001,
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True,
                layer_idx=i,
                device='cuda',
                dtype=torch.float32)
            for i in range (self.model_cfg.NUM_MAMBA_LAYER)
        ])
        
        self.upconv1 = spconv.SparseSequential(
            block(256, 256, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv'),
        )
        self.upconv2 = spconv.SparseSequential(
            block(256, 128, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv'),
        )
        self.upconv3 = spconv.SparseSequential(
            block(128, 64, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv'),
        )
        
        self.seg_mlp64_1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.seg_mlp64_2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.seg_mlp128 = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.seg_mlp256_1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.seg_mlp256_2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.convupconv1 = spconv.SparseSequential(
            block(64, 64, kernel_size=(3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=(1, 0, 0), indice_key='spconv6', conv_type='spconv'),
        )
        self.convupconv2 = spconv.SparseSequential(
            block(64, 64, kernel_size=(3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=(1, 0, 0), indice_key='spconv7', conv_type='spconv'),
        )
        self.convupconv3 = spconv.SparseSequential(
            block(64, 64, kernel_size=(3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=(0, 0, 0), indice_key='spconv8', conv_type='spconv'),
        )
            
        last_pad = (0, 0, 0)
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(64, 64, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(64),
            nn.ReLU(),
        )
        
        self.num_point_features = 64
        
        self.backbone_channels = {
            'x_conv1': 64,
            'x_conv2': 128,
            'x_conv3': 256,
            'x_conv4': 256
        }
        
        self.focal_loss = SigmoidFocalClassificationLoss()
        
    def _indice_to_scalar(self, indices: torch.Tensor, shape: List[int]):
        assert indices.shape[1] == len(shape)
        stride = to_stride(np.array(shape, dtype=np.int64))
        scalar_inds = indices[:, -1].clone()
        for i in range(len(shape) - 1):
            scalar_inds += stride[i] * indices[:, i]
        return scalar_inds.contiguous()

    def forward(self, batch_dict):
        self.voxel_size = batch_dict['voxel_size']
        batch_size = batch_dict['batch_size']

        lidar_features = batch_dict['lidar_features']   # [M, 64]
        radar_features = batch_dict['radar_features']   # [N, 64]
        lidar_coords = batch_dict['lidar_voxel_coords'] # [M, 4(bzyx)]
        radar_coords = batch_dict['radar_voxel_coords'] # [N, 4(bzyx)]

        lidar_sp_tensor = spconv.SparseConvTensor(
            features=lidar_features,
            indices=lidar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        radar_sp_tensor = spconv.SparseConvTensor(
            features=radar_features,
            indices=radar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        fusion_sp_tensor = Fsp.sparse_add(lidar_sp_tensor, radar_sp_tensor)
        input_sp_tensor = fusion_sp_tensor

        # [41, 320, 320] * 64 channels
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x) # [41, 320, 320] * 64 channels
        x_conv2 = self.conv2(x_conv1)   # [21, 160, 160] * 128 channels
        x_conv3 = self.conv3(x_conv2)   # [11, 80, 80] * 256 channels
        x_conv4 = self.conv4(x_conv3)   # [5, 40, 40] * 256 channels
        
        x_conv4_mamba_recover = []
        for batch_idx in range(batch_size):
            x_conv4_batch_features = x_conv4.features[x_conv4.indices[:, 0] == batch_idx]
            x_conv4_batch_coords = x_conv4.indices[x_conv4.indices[:, 0] == batch_idx][:, 1:]
            sorted_indices, sorted_embeddings = self.svso(x_conv4_batch_features.unsqueeze(0))
            
            for mamba_block in self.vmamba:
                sorted_embeddings = mamba_block(sorted_embeddings, self.pos_embed)
            inv_indices = sorted_indices.squeeze(0).argsort()
            x_conv4_mamba_recover.append(sorted_embeddings.squeeze(0).index_select(0, inv_indices))
            
        x_conv4_mamba_recover = torch.cat(x_conv4_mamba_recover, dim=0).index_select(0, x_conv4.indices[:, 0].argsort().argsort())
        x_conv4 = replace_feature(x_conv4, x_conv4.features * x_conv4_mamba_recover)    # [5, 40, 40] * 256 channels
        
        pred_logits_256_2 = self.seg_mlp256_2(x_conv4.features)
        x_conv4 = replace_feature(x_conv4, x_conv4.features * pred_logits_256_2.sigmoid())

        x_upconv1 = self.upconv1(x_conv4)
        x_conv3 = replace_feature(x_conv3, x_conv3.features * x_upconv1.features)   # [11, 80, 80] * 256 channels
        pred_logits_256_1 = self.seg_mlp256_1(x_conv3.features)
        x_conv3 = replace_feature(x_conv3, x_conv3.features * pred_logits_256_1.sigmoid())

        x_upconv2 = self.upconv2(x_conv3)
        x_conv2 = replace_feature(x_conv2, x_conv2.features * x_upconv2.features)   # [21, 160, 160] * 128 channels
        pred_logits_128 = self.seg_mlp128(x_conv2.features)
        x_conv2 = replace_feature(x_conv2, x_conv2.features * pred_logits_128.sigmoid())
        
        x_upconv3 = self.upconv3(x_conv2)
        x_conv1 = replace_feature(x_conv1, x_conv1.features * x_upconv3.features)   # [41, 320, 320] * 64 channels
        pred_logits_64_2 = self.seg_mlp64_2(x_conv1.features)
        x_conv1 = replace_feature(x_conv1, x_conv1.features * pred_logits_64_2.sigmoid())    
        
        x = replace_feature(x, x.features * x_conv1.features)   # [41, 320, 320] * 64 channels
        pred_logits_64_1 = self.seg_mlp64_1(x.features)
        x = replace_feature(x, x.features * pred_logits_64_1.sigmoid())

        conv_x_conv1 = self.convupconv1(x)
        conv_x_conv2 = self.convupconv2(conv_x_conv1)
        conv_x_conv3 = self.convupconv3(conv_x_conv2)
        out = self.conv_out(conv_x_conv3)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 1,
            'x_indices': x.indices,
            'x_conv1_indices': x_conv1.indices,
            'x_conv2_indices': x_conv2.indices,
            'x_conv3_indices': x_conv3.indices,
            'x_conv4_indices': x_conv4.indices,
            'pred_logits_64_1': pred_logits_64_1,
            'pred_logits_64_2': pred_logits_64_2,
            'pred_logits_128': pred_logits_128,
            'pred_logits_256_1': pred_logits_256_1,
            'pred_logits_256_2': pred_logits_256_2,
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        return batch_dict

    def get_fg_labels(self, voxel_centers, voxel_coords, gt_boxes):
        bs_idx = voxel_coords[:, 0]
        voxel_cls_labels = voxel_centers.new_zeros(voxel_centers.shape[0]).long()
        for k in range(gt_boxes.shape[0]):
            bs_mask = (bs_idx == k)
            voxels_single = voxel_centers[bs_mask]
            voxel_cls_labels_single = voxel_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = points_in_boxes_gpu(voxels_single.unsqueeze(0), gt_boxes[k:k + 1, :, 0:7].contiguous()).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            voxel_cls_labels_single[box_fg_flag] = 1
            voxel_cls_labels[bs_mask] = voxel_cls_labels_single
        return voxel_cls_labels

    def get_loss(self, batch_dict, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        voxel_centers_64_1 = common_utils.get_voxel_centers(
            voxel_coords=batch_dict['x_indices'][:, 1:4], 
            downsample_times=1, 
            voxel_size=self.voxel_size, 
            point_cloud_range=self.point_cloud_range
            )
        voxel_centers_64_2 = common_utils.get_voxel_centers(
            voxel_coords=batch_dict['x_conv1_indices'][:, 1:4], 
            downsample_times=1, 
            voxel_size=self.voxel_size, 
            point_cloud_range=self.point_cloud_range
            )
        voxel_centers_128 = common_utils.get_voxel_centers(
            voxel_coords=batch_dict['x_conv2_indices'][:, 1:4], 
            downsample_times=2, 
            voxel_size=self.voxel_size, 
            point_cloud_range=self.point_cloud_range
            )
        voxel_centers_256_1 = common_utils.get_voxel_centers(
            voxel_coords=batch_dict['x_conv3_indices'][:, 1:4], 
            downsample_times=4, 
            voxel_size=self.voxel_size, 
            point_cloud_range=self.point_cloud_range
            )
        voxel_centers_256_2 = common_utils.get_voxel_centers(
            voxel_coords=batch_dict['x_conv4_indices'][:, 1:4], 
            downsample_times=8, 
            voxel_size=self.voxel_size, 
            point_cloud_range=self.point_cloud_range
            )
        gt_boxes = batch_dict['gt_boxes']
        fg_labels_64_1 = self.get_fg_labels(voxel_centers_64_1, batch_dict['x_indices'], gt_boxes)
        fg_labels_64_1 = fg_labels_64_1.view(-1, 1)
        fg_labels_64_2 = self.get_fg_labels(voxel_centers_64_2, batch_dict['x_conv1_indices'], gt_boxes)
        fg_labels_64_2 = fg_labels_64_2.view(-1, 1)
        fg_labels_128 = self.get_fg_labels(voxel_centers_128, batch_dict['x_conv2_indices'], gt_boxes)
        fg_labels_128 = fg_labels_128.view(-1, 1)
        fg_labels_256_1 = self.get_fg_labels(voxel_centers_256_1, batch_dict['x_conv3_indices'], gt_boxes)
        fg_labels_256_1 = fg_labels_256_1.view(-1, 1)
        fg_labels_256_2 = self.get_fg_labels(voxel_centers_256_2, batch_dict['x_conv4_indices'], gt_boxes)
        fg_labels_256_2 = fg_labels_256_2.view(-1, 1)
        
        focal_loss_64_1 = self.focal_loss(batch_dict['pred_logits_64_1'], fg_labels_64_1, torch.ones_like(fg_labels_64_1.view(-1)))
        focal_loss_64_2 = self.focal_loss(batch_dict['pred_logits_64_2'], fg_labels_64_2, torch.ones_like(fg_labels_64_2.view(-1)))
        focal_loss_128 = self.focal_loss(batch_dict['pred_logits_128'], fg_labels_128, torch.ones_like(fg_labels_128.view(-1)))
        focal_loss_256_1 = self.focal_loss(batch_dict['pred_logits_256_1'], fg_labels_256_1, torch.ones_like(fg_labels_256_1.view(-1)))
        focal_loss_256_2 = self.focal_loss(batch_dict['pred_logits_256_2'], fg_labels_256_2, torch.ones_like(fg_labels_256_2.view(-1)))
        
        forward_ret_dict = {
            'loss_voxelseg_320v_1': focal_loss_64_1.sum() / batch_dict['x_indices'].shape[0],
            'loss_voxelseg_320v_2': focal_loss_64_2.sum() / batch_dict['x_conv1_indices'].shape[0],
            'loss_voxelseg_160v': focal_loss_128.sum() / batch_dict['x_conv2_indices'].shape[0],
            'loss_voxelseg_80v': focal_loss_256_1.sum() / batch_dict['x_conv3_indices'].shape[0],
            'loss_voxelseg_40v': focal_loss_256_2.sum() / batch_dict['x_conv4_indices'].shape[0],
        }
        
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        
        loss_320p_1 = forward_ret_dict['loss_voxelseg_320v_1']
        loss_320p_2 = forward_ret_dict['loss_voxelseg_320v_2']
        loss_160p = forward_ret_dict['loss_voxelseg_160v']
        loss_80p = forward_ret_dict['loss_voxelseg_80v']
        loss_40p = forward_ret_dict['loss_voxelseg_40v']
        
        loss_320p_1 = loss_320p_1 * loss_weights_dict['voxelseg_layer_weight'][0]
        loss_320p_2 = loss_320p_2 * loss_weights_dict['voxelseg_layer_weight'][1]
        loss_160p = loss_160p * loss_weights_dict['voxelseg_layer_weight'][2]
        loss_80p = loss_80p * loss_weights_dict['voxelseg_layer_weight'][3]
        loss_40p = loss_40p * loss_weights_dict['voxelseg_layer_weight'][4]
        
        loss = loss_320p_1 + loss_320p_2 + loss_160p + loss_80p + loss_40p

        tb_dict['loss_voxelseg_320v_1'] = loss_320p_1.item()
        tb_dict['loss_voxelseg_320v_2'] = loss_320p_2.item()
        tb_dict['loss_voxelseg_160v'] = loss_160p.item()
        tb_dict['loss_voxelseg_80v'] = loss_80p.item()
        tb_dict['loss_voxelseg_40v'] = loss_40p.item()
        tb_dict['loss_voxelseg'] = loss.item()
        return loss, tb_dict