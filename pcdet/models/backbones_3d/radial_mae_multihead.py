import torch
import torch.nn as nn
import spconv
from functools import partial

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
    def __init__(self, input_channels, grid_size, voxel_size, point_cloud_range, num_heads=3):
        super(MultiHeadEncoder, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        for _ in range(num_heads):
            head = spconv.SparseSequential(
                spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
                norm_fn(16),
                nn.ReLU(),
                post_act_block(16, 16, 3, stride=1, padding=1, norm_fn=norm_fn, indice_key='subm1'),
                post_act_block(16, 32, 3, stride=2, padding=1, norm_fn=norm_fn, indice_key='spconv2', conv_type='spconv'),
                post_act_block(32, 32, 3, stride=1, padding=1, norm_fn=norm_fn, indice_key='subm2'),
                post_act_block(32, 32, 3, stride=1, padding=1, norm_fn=norm_fn, indice_key='subm2'),
                post_act_block(32, 64, 3, stride=2, padding=1, norm_fn=norm_fn, indice_key='spconv3', conv_type='spconv'),
                post_act_block(64, 64, 3, stride=1, padding=1, norm_fn=norm_fn, indice_key='subm3'),
                post_act_block(64, 64, 3, stride=1, padding=1, norm_fn=norm_fn, indice_key='subm3'),
                post_act_block(64, 64, 3, stride=2, padding=(0, 1, 1), norm_fn=norm_fn, indice_key='spconv4', conv_type='spconv'),
                post_act_block(64, 64, 3, stride=1, padding=1, norm_fn=norm_fn, indice_key='subm4'),
                post_act_block(64, 64, 3, stride=1, padding=1, norm_fn=norm_fn, indice_key='subm4'),
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



class Radial_MAE(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super(Radial_MAE, self).__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.masked_ratio = model_cfg.get('MASKED_RATIO', 0.15)
        self.angular_range = model_cfg.get('ANGULAR_RANGE', 5)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.encoder = MultiHeadEncoder(input_channels, norm_fn)

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

    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        # Prepare input sparse tensor
        input_sp_tensor = SparseConvTensor(voxel_features, voxel_coords.int(), self.grid_size, batch_size)
        encoded_features = self.encoder(input_sp_tensor)

        # Decoder processing
        decoded_features = self.decoder(encoded_features.dense())

        # Assuming the target is provided in the batch_dict for training
        if 'targets' in batch_dict:
            targets = batch_dict['targets']
            # Ensure targets are in the correct shape and device
            targets = targets.view_as(decoded_features)
            loss = self.criterion(decoded_features, targets)
            batch_dict['loss'] = loss

        # For inference or validation, you might want to include additional processing
        # to convert decoded_features into a more interpretable format or calculate metrics

        return batch_dict

