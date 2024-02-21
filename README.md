## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of [OpenPCDet(v0.6)](https://github.com/open-mmlab/OpenPCDet).

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) .

## Usage

### SSL Pre-training Radial MAE

### KITTI:

* Pretrain with multiple GPUs:
```shell
bash ./scripts/dist_train_mae.sh ${NUM_GPUS} \
  --cfg_file cfgs/kitti_models/radial_mae_kitti.yaml
```
* Pretrain with a single GPU:
```shell
python3 train_ssl.py ${NUM_GPUS} \
  --cfg_file cfgs/kitti_models/radial_mae_kitti.yaml --batch_size ${BATCH_SIZE}
```

### Waymo:

```shell
bash ./scripts/dist_train_mae.sh ${NUM_GPUS} \
  --cfg_file cfgs/waymo_models/radial_mae_waymo.yaml
```

### nuScenes:

```shell
bash ./scripts/dist_train_mae.sh ${NUM_GPUS} \
  --cfg_file cfgs/nuscenes_models/radial_mae_nuescenes.yaml
```

### Traing Detection Head

Train with multiple GPUs:
```shell
bash ./scripts/dist_train.sh ${NUM_GPUS} \
  --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
  --pretrained_model ../output/kitti_models/radial_mae_pretrain_kitti_0.9/default/ckpt/checkpoint_epoch_30.pth
```

## Performance

### KITTI Dataset

Performance comparison on the kitti val split evaluated by the ap with 40 recall positions at moderate difficulty level.

|                                             | Car@R40 | Pedestrian@R40 | Cyclist@R40  | download | 
|---------------------------------------------|:-------:|:--------------:|:------------:|:--------:|
| [PointPillar](tools/cfgs/kitti_models/pointpillar.yaml) | 75.60 | 41.98 | 60.27 | [ckpt]() | 
| [SECOND](tools/cfgs/kitti_models/second.yaml)       | - | - | - | [ckpt]() |
| [PV-RCNN](tools/cfgs/kitti_models/pv_rcnn.yaml) | - | - | - | [ckpt]() |
| [Radia-MAE]() | 82.73 | 52.77 | 72.85 | [ckpt]() |




### Waymo Open Dataset

All models are trained with **a single frame** of **20% data (~32k frames)** of all the training samples on 2 RTX A6000 GPUs, and the results of each cell here are mAP/mAPH calculated by the official Waymo evaluation metrics on the **whole** validation set (version 1.2).    

|    Performance@(train with 20\% Data)            | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |  
|---------------------------------------------|----------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| [SECOND](tools/cfgs/waymo_models/second.yaml) | 70.96/70.34|62.58/62.02|65.23/54.24	|57.22/47.49|	57.13/55.62 |	54.97/53.53 | 
| [PointPillar](tools/cfgs/waymo_models/pointpillar_1x.yaml) | 70.43/69.83 |	62.18/61.64 | 66.21/46.32|58.18/40.64|55.26/51.75|53.18/49.80 |
[CenterPoint-Pillar](tools/cfgs/waymo_models/centerpoint_pillar_1x.yaml)| 70.50/69.96|62.18/61.69|73.11/61.97|65.06/55.00|65.44/63.85|62.98/61.46| 
[CenterPoint-Dynamic-Pillar](tools/cfgs/waymo_models/centerpoint_dyn_pillar_1x.yaml)| 70.46/69.93|62.06/61.58|73.92/63.35|65.91/56.33|66.24/64.69|63.73/62.24| 
[CenterPoint](tools/cfgs/waymo_models/centerpoint_without_resnet.yaml)| 71.33/70.76|63.16/62.65|	72.09/65.49	|64.27/58.23|	68.68/67.39	|66.11/64.87|
| [CenterPoint (ResNet)](tools/cfgs/waymo_models/centerpoint.yaml)|72.76/72.23|64.91/64.42	|74.19/67.96	|66.03/60.34|	71.04/69.79	|68.49/67.28 |
| [Part-A2-Anchor](tools/cfgs/waymo_models/PartA2.yaml) | 74.66/74.12	|65.82/65.32	|71.71/62.24	|62.46/54.06	|66.53/65.18	|64.05/62.75 |
| [PV-RCNN (AnchorHead)](tools/cfgs/waymo_models/pv_rcnn.yaml) | 75.41/74.74	|67.44/66.80	|71.98/61.24	|63.70/53.95	|65.88/64.25	|63.39/61.82 | 
| [PV-RCNN (CenterHead)](tools/cfgs/waymo_models/pv_rcnn_with_centerhead_rpn.yaml) | 75.95/75.43	|68.02/67.54	|75.94/69.40	|67.66/61.62	|70.18/68.98	|67.73/66.57|
| [Radial-MAE + PV-RCNN (AnchorHead)]() | 75.70/75.05 |	67.16/66.56|	73.40/63.54| 64.47/55.63 | 67.91/66.45	|	65.40/63.99|


### nuScenes Dataset 


##  License
Our codes are released under the Apache 2.0 license.

## Acknowledgement

This project is mainly based on the following codebases. Thanks for their great works!

* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
* [SPCONV](https://github.com/traveller59/spconv)

