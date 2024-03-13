import torch
import torch.nn as nn
import spconv
from functools import partial
import random

from spconv.pytorch import SparseConvTensor
from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils
from .spconv_backbone import post_act_block

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = spconv.SubMConv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        out = self.conv1(x)
        out = self.bn1(out.features)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x).features

        out += identity
        out = self.relu(out)

        return spconv.SparseConvTensor(out, x.indices, x.spatial_shape, x.batch_size)


class MultiHeadEncoder(nn.Module):
    def __init__(self, input_channels, grid_size, voxel_size, point_cloud_range, num_heads=2):
        super(MultiHeadEncoder, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # Retrieve last_pad from model_cfg, default to 0 if not specified
        #last_pad = model_cfg.get('last_pad', 0)

        for _ in range(num_heads):
            head = spconv.SparseSequential(
                spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
                norm_fn(16),
                nn.ReLU(),
                post_act_block(16, 16, 3, stride=1, padding=1, norm_fn=norm_fn, indice_key='subm1'),
                # [1600, 1408, 41] <- [800, 704, 21]
                post_act_block(16, 32, 3, stride=2, padding=1, norm_fn=norm_fn, indice_key='spconv2', conv_type='spconv'),
                post_act_block(32, 32, 3, stride=1, padding=1, norm_fn=norm_fn, indice_key='subm2'),
                post_act_block(32, 32, 3, stride=1, padding=1, norm_fn=norm_fn, indice_key='subm2'),
                # [800, 704, 21] <- [400, 352, 11]
                post_act_block(32, 64, 3, stride=2, padding=1, norm_fn=norm_fn, indice_key='spconv3', conv_type='spconv'),
                post_act_block(64, 64, 3, stride=1, padding=1, norm_fn=norm_fn, indice_key='subm3'),
                post_act_block(64, 64, 3, stride=1, padding=1, norm_fn=norm_fn, indice_key='subm3'),
                # [400, 352, 11] <- [200, 176, 5]
                post_act_block(64, 64, 3, stride=2, padding=(0, 1, 1), norm_fn=norm_fn, indice_key='spconv4', conv_type='spconv'),
                post_act_block(64, 64, 3, stride=1, padding=1, norm_fn=norm_fn, indice_key='subm4'),
                post_act_block(64, 64, 3, stride=1, padding=1, norm_fn=norm_fn, indice_key='subm4'),
                # [200, 150, 5] -> [200, 150, 2]
                spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=0, bias=False, indice_key='spconv_down2'),
                norm_fn(128),
                nn.ReLU(),
            )
            self.heads.append(head)

    def forward(self, x):
        aggregated_features = []
        for head in self.heads:
            head_output = head(x)
            aggregated_features.append(head_output.features)

        # Aggregate features from all heads
        aggregated_features = torch.stack(aggregated_features, dim=0).mean(dim=0)

        # Replace features in the input sparse tensor with the aggregated features
        x = x.replace_feature(aggregated_features)

        return x



class Radial_MAE_multihead(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.masked_ratio = model_cfg.get('MASKED_RATIO', 0.15)
        self.angular_range = model_cfg.get('ANGULAR_RANGE', 5)

        #norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.encoder = MultiHeadEncoder(input_channels, grid_size, voxel_size, point_cloud_range, num_heads=2)
        self.num_point_features = 8


        # Initialize the decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=1),
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.forward_re_dict = {}
    
    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        pred = self.forward_re_dict['pred']
        target = self.forward_re_dict['target']
        loss = self.criterion(pred, target)

        tb_dict = {
            'loss_rpn': loss.item()
        }

        return loss, tb_dict


    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']

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


        # nums is set to voxel_features.shape[0], which is the number of voxels (the size of the first dimension of voxel_features)
        nums = voxel_features.shape[0]
        voxel_fratures_all_one = torch.ones(nums,1).to(voxel_features.device)
        # Use the selected indices to segment voxel_features and voxel_coords
        voxel_features_partial = voxel_features[selected_indices_tensor, :] # shape [N, C]
        voxel_coords_partial = voxel_coords[selected_indices_tensor, :] # shape [N, 4]

        batch_size = batch_dict['batch_size']

        # Prepare input sparse tensor
        input_sp_tensor = SparseConvTensor(
            features=voxel_features_partial,
            indices=voxel_coords_partial.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        input_sp_tensor_ones = SparseConvTensor(
            features=voxel_fratures_all_one,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        encoded_features = self.encoder(input_sp_tensor)

        # Decoder processing
        self.forward_re_dict['target'] = input_sp_tensor_ones.dense()
        decoded_features = self.decoder(encoded_features.dense())
        self.forward_re_dict['pred'] = decoded_features

        '''
        # Assuming the target is provided in the batch_dict for training
        if 'targets' in batch_dict:
            targets = batch_dict['targets']
            # Ensure targets are in the correct shape and device
            targets = targets.view_as(decoded_features)
            loss = self.criterion(decoded_features, targets)
            batch_dict['loss'] = loss
        '''
        

        # For inference or validation, you might want to include additional processing
        # to convert decoded_features into a more interpretable format or calculate metrics

        return batch_dict

