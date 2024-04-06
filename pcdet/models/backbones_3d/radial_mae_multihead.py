from functools import partial
import random
import math
import numpy as np
import torch
import torch.nn as nn
import spconv
from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils
from .spconv_backbone import post_act_block



# Assuming norm_fn and post_act_block are defined and imported correctly

class SparseBasicBlockMultiHead(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size, stride=1, downsample=None, indice_key=None, norm_fn=None, heads=2, conv_type='subm'):
        super().__init__()
        self.heads = heads
        self.blocks = nn.ModuleList([
            post_act_block(inplanes, planes, kernel_size, indice_key=f"{indice_key}_{i}", stride=stride, padding=1, conv_type=conv_type, norm_fn=norm_fn)
            for i in range(heads)
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        out_features = []
        for block in self.blocks:
            out = block(x)
            out_features.append(out.features)
        
        # Aggregating features from all heads
        aggregated_features = torch.cat(out_features, dim=1)
        out = replace_feature(x, aggregated_features)
        return out

    

class Radial_MAE_multihead(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, heads=2, **kwargs):
        super(Radial_MAE_multihead, self).__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        self.masked_ratio = model_cfg.MASKED_RATIO
        self.angular_range = model_cfg.get('ANGULAR_RANGE', 5)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.heads = heads
        input_channels_multiplied = input_channels * heads  # Adjusted for multi-head

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16 * heads, 3, bias=False, indice_key='subm1'),
            norm_fn(16 * heads),
            nn.ReLU(),
        )

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlockMultiHead(16 * heads, 16 * heads, 3, norm_fn=norm_fn, indice_key='subm1', heads=heads),
        )

        self.conv2 = spconv.SparseSequential(
            SparseBasicBlockMultiHead(16 * heads, 32 * heads, 3, norm_fn=norm_fn, stride=2, indice_key='spconv2', heads=heads),
            SparseBasicBlockMultiHead(32 * heads, 32 * heads, 3, norm_fn=norm_fn, indice_key='subm2', heads=heads),
            SparseBasicBlockMultiHead(32 * heads, 32 * heads, 3, norm_fn=norm_fn, indice_key='subm2', heads=heads),
        )

        self.conv3 = spconv.SparseSequential(
            SparseBasicBlockMultiHead(32 * heads, 64 * heads, 3, norm_fn=norm_fn, stride=2, indice_key='spconv3', heads=heads),
            SparseBasicBlockMultiHead(64 * heads, 64 * heads, 3, norm_fn=norm_fn, indice_key='subm3', heads=heads),
            SparseBasicBlockMultiHead(64 * heads, 64 * heads, 3, norm_fn=norm_fn, indice_key='subm3', heads=heads),
        )

        self.conv4 = spconv.SparseSequential(
            SparseBasicBlockMultiHead(64 * heads, 64 * heads, 3, norm_fn=norm_fn, stride=2, indice_key='spconv4', heads=heads),
            SparseBasicBlockMultiHead(64 * heads, 64 * heads, 3, norm_fn=norm_fn, indice_key='subm4', heads=heads),
            SparseBasicBlockMultiHead(64 * heads, 64 * heads, 3, norm_fn=norm_fn, indice_key='subm4', heads=heads),
        )

        self.num_point_features = 16


        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(64 * heads, 32 * heads, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32 * heads),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(32 * heads, 8 * heads, 3, padding=1, output_padding=1, stride=(4, 2, 2), bias=False),
            nn.BatchNorm3d(8 * heads),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(8 * heads, 1, 3, padding=1, output_padding=1, stride=(3, 2, 2), bias=False),
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.forward_re_dict = {}


    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        pred = self.forward_re_dict['pred']
        target = self.forward_re_dict['target']
        loss = self.criterion(pred, target)
        tb_dict.update({'loss_rpn': loss.item()})
        return loss, tb_dict

    def forward(self, batch_dict):
        """
        Forward pass for the Radial Multi-Head MAE model.
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        nums = voxel_features.shape[0]
        # Calculate angles and select voxels based on radial grouping - same as single-head model
        # The radial selection logic here is unchanged, but make sure to adjust voxel_features processing if necessary
        selected_indices_tensor, voxel_features_partial, voxel_coords_partial = self.radial_selection(voxel_features, voxel_coords, batch_size)
        voxel_fratures_all_one = torch.ones(nums,1).to(voxel_features.device)


        # Create sparse tensor with selected voxels
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features_partial,
            indices=voxel_coords_partial.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        input_sp_tensor_ones = spconv.SparseConvTensor(
            features=voxel_fratures_all_one,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        # Encoder layers - multi-head logic already integrated in the conv* layers
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        # Assuming conv_out is the layer connecting encoder and decoder parts
        if self.conv_out:
            out = self.conv_out(x_conv4)
            out_features = out.dense()
        else:
            out_features = x_conv4.dense()

        self.forward_re_dict['target'] = input_sp_tensor_ones.dense()
        # Decoder layers
        x_up1 = self.deconv1(out_features)
        x_up2 = self.deconv2(x_up1)
        x_up3 = self.deconv3(x_up2)
        
        # Prepare output dict
        self.forward_re_dict['pred'] = x_up3

        # Creating target tensor for loss computation (considering binary occupancy as targets)
        # Assuming the target tensor preparation logic remains similar to the single-head model
        #self.prepare_targets(batch_dict, selected_indices_tensor)
        
        return batch_dict

    def radial_selection(self, voxel_features, voxel_coords, batch_size):
        # Implement the radial selection logic here, same as in the single-head model
        # Ensure to return selected_indices_tensor, voxel_features_partial, and voxel_coords_partial

        select_ratio = 1 - self.masked_ratio # ratio for select voxel
        
        # Calculate angles in radians for each voxel
        angles = torch.atan2(voxel_coords[:, 3], voxel_coords[:, 2])  # atan2(y, x)
        angles_deg = torch.rad2deg(angles) % 360  # Convert to degrees and ensure in range [0, 360)

        # Group indices based on angles (adjustable angular range)
        angular_range = self.angular_range  # degrees, can be adjusted
        radial_groups = {}
        for angle in range(0, 360, angular_range):
            mask = (angles_deg >= angle) & (angles_deg < angle + angular_range)
            group_indices = torch.where(mask)[0]
            if len(group_indices) > 0:  # Only consider non-empty groups
                radial_groups[angle] = group_indices
        
        # Ensure there are enough groups to select from
        if len(radial_groups) == 0:
            raise ValueError("No non-empty radial groups found. Consider adjusting the selection criteria.")
        
        # Randomly select a portion of radial groups
        select_ratio = 1 - self.masked_ratio  # Assuming masked_ratio is defined
        num_groups_to_select = min(int(select_ratio * len(radial_groups)), len(radial_groups))
        selected_group_angles = random.sample(list(radial_groups.keys()), num_groups_to_select)

        # Combine indices from the selected groups
        selected_indices = []
        for angle in selected_group_angles:
            selected_indices.extend(radial_groups[angle].tolist())
        
        # Convert list to 1-dimensional tensor
        selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=voxel_coords.device)

        # Use the selected indices to segment voxel_features and voxel_coords
        voxel_features_partial = voxel_features[selected_indices_tensor, :] # shape [N, C]
        voxel_coords_partial = voxel_coords[selected_indices_tensor, :] # shape [N, 4]

        return selected_indices_tensor, voxel_features_partial, voxel_coords_partial

    '''
    def prepare_targets(self, batch_dict, selected_indices_tensor):
        """
        Prepare target tensors for loss computation.
        """
        # Assuming binary occupancy as targets, prepare the target tensor
        # The logic for preparing targets will likely be similar to the single-head model,
        # but ensure it's compatible with the multi-head output dimensions if necessary.
        pass
    '''
    


