## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of [OpenPCDet(v0.6)](https://github.com/open-mmlab/OpenPCDet).

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) .

## Usage

### Pre-training Radial MAE

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
  --cfg_file cfgs/nuscenes_models/radial_mae_res_nuescenes.yaml
```

### Finetuning

Train with multiple GPUs:
* example of fintetuning Radial_MAE checkpoint on Waymo using PVRCNN
```shell
bash ./scripts/dist_train.sh ${NUM_GPUS} \
  --cfg_file cfgs/waymo_models/pv_rcnn.yaml \
  --pretrained_model ../output/waymo_models/radial_mae_waymo/default/ckpt/checkpoint_epoch_30.pth
```

## Performance

### KITTI Dataset

Performance comparison on the kitti val split evaluated by the ap with 40 recall positions at moderate difficulty level.

|                                             | Car@R40 | Pedestrian@R40 | Cyclist@R40  | download | 
|---------------------------------------------|:-------:|:--------------:|:------------:|:--------:|
| [SECOND](tools/cfgs/kitti_models/second.yaml)       | 79.08 | 44.52 | 64.49 | [ckpt 71]() |
| [SECOND + Radial-MAE (0.8)]()       | 79.64 | 47.33 | 65.65 | [ckpt 75]() |
| [SECOND + Radial-MAE (0.9)]()       | 79.01 | 46.93 | 67.75 | [ckpt 73]() |
| [SECOND + Occupancy-MAE]()       | 79.12 | 45.35 | 63.27 | [ckpt 80]() |
| [SECOND + ALSO]()       | 78.98 | 45.33 | 66.53 | [ckpt 71]() |
| [PV-RCNN](tools/cfgs/kitti_models/pv_rcnn.yaml) | 82.28 | 51.51 | 69.45 | [ckpt 70]() |
| [PV-RCNN + Radial-MAE (0.8) ]() | 83.00 | 52.08 | 71.16 | [ckpt 78]() |
| [PV-RCNN + Radial-MAE (0.9)]() | 82.82 | 51.61 | 73.82 | [ckpt 78]() |


Performance comparison on the kitti val split evaluated by the ap with 11 recall positions at moderate difficulty level.

|                                             | Car@R11 | Pedestrian@R11 | Cyclist@R11  | download | 
|---------------------------------------------|:-------:|:--------------:|:------------:|:--------:|
| [SECOND](tools/cfgs/kitti_models/second.yaml)       | 77.81 | 46.33 | 63.65 | [ckpt]() |
| [SECOND + Radial-MAE (0.8)]()       | 78.23 | 48.70 | 65.72 | [ckpt 75]() |
| [SECOND + Radial-MAE (0.9)]()       | 77.64 | 48.52 | 67.94 | [ckpt 73]() |
| [SECOND + Occupancy-MAE]()       | 77.75 | 47.63 | 63.82 | [ckpt]() |
| [SECOND + ALSO]()       | 77.75 | 46.18 | 66.60 | [ckpt 71]() |
| [PV-RCNN](tools/cfgs/kitti_models/pv_rcnn.yaml) | 78.81 | 52.84 | 69.10 | [ckpt 70]() |
| [PV-RCNN + Radial-MAE (0.8)]() | 79.42 | 53.24 | 71.33 | [ckpt 78]() |
| [PV-RCNN + Radial-MAE (0.9)]() | 79.25 | 53.10 | 72.99 | [ckpt 78]() |


"Performance Comparison of Radial-MAE Variations with 80% Masking and Angular Ranges of 1°, 5°, and 10° Fine-Tuned on SECOND, Evaluated on KITTI Validation Split by AP with 40/11 Recall Positions at Moderate Difficulty Level"


|                                             | Car @40/@R11 | Pedestrian @40/@R11 | Cyclist @40/@R11  |
|---------------------------------------------|:-------:|:--------------:|:------------:|
| [SECOND](tools/cfgs/kitti_models/second.yaml)       | 79.08/77.81 | 44.52/46.33 | 64.49/63.65 |
| [SECOND + Radial-MAE (0.8) 1d]()       | 79.64/78.23 | 47.33/48.70 | 65.65/65.72 |
| [SECOND + Radial-MAE (0.8) 5d]()       | 79.38/78.05 | 46.81/48.00 | 63.62/64.48 |
| [SECOND + Radial-MAE (0.8) 10d]()       | 79.41/78.04 | 46.23/47.57 | 65.18/65.21s |






### Waymo Open Dataset

All models are trained with **a single frame** of **20% data (~32k frames)** of all the training samples on 2 RTX 6000 ADA GPUs, and the results of each cell here are mAP/mAPH calculated by the official Waymo evaluation metrics on the **whole** validation set (version 1.2).    

|    Performance@(train with 20\% Data)            | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |  
|---------------------------------------------|----------:|:-------:|:-------:|:-------:|:-------:|:-------:|
[CenterPoint](tools/cfgs/waymo_models/centerpoint_without_resnet.yaml)| 71.33/70.76|63.16/62.65|	72.09/65.49	|64.27/58.23|	68.68/67.39	|66.11/64.87|
| [CenterPoint (ResNet)](tools/cfgs/waymo_models/centerpoint.yaml)|72.76/72.23|64.91/64.42	|74.19/67.96	|66.03/60.34|	71.04/69.79	|68.49/67.28 |
| [CenterPoint (ResNet) + Radial-MAE](tools/cfgs/waymo_models/centerpoint.yaml)| 73.38/72.85 | 65.28/64.79	| 74.84/68.68	| 66.90/61.24 |	72.05/70.84	| 69.43/68.26 |
| [CenterPoint + Occupancy-MAE]()| 71.89/71.33 | 64.05/63.53	| 73.85/67.12	| 65.78/59.62 |	70.29/69.03	| 67.76/66.53 |
| [CenterPoint + GCC-3D]()| -/- | 63.97/63.47	| -/-	| 64.23/58.47 |	-/-	| 67.68/66.44 |
| [Voxel R-CNN (CenterHead)-Dynamic-Voxel](tools/cfgs/waymo_models/voxel_rcnn_with_centerhead_dyn_voxel.yaml) | 76.13/75.66	|68.18/67.74	|78.20/71.98	|69.29/63.59	| 70.75/69.68	|68.25/67.21|
| [Voxel R-CNN (CenterHead)-Dynamic-Voxel + Radial-MAE]() | 76.35/75.88	| 67.99/67.56 | 78.60/72.56	| 69.93/64.35	| 71.74/70.65	| 69.13/68.08 |
| [PV-RCNN (AnchorHead)](tools/cfgs/waymo_models/pv_rcnn.yaml) | 75.41/74.74	|67.44/66.80	|71.98/61.24	|63.70/53.95	|65.88/64.25	|63.39/61.82 | 
| [PV-RCNN (AnchorHead) + Radial-MAE]() | 75.70/75.05 |	67.16/66.56|	73.40/63.54| 64.47/55.63 | 67.91/66.45	|	65.40/63.99|
| [PV-RCNN (CenterHead)](tools/cfgs/waymo_models/pv_rcnn_with_centerhead_rpn.yaml) | 75.95/75.43	|68.02/67.54	|75.94/69.40	|67.66/61.62	|70.18/68.98	|67.73/66.57|
| [PV-RCNN (CenterHead) + Radial-MAE]() | 76.72/76.22 |	68.38/67.92|	78.19/71.74 | 69.63/63.68 | 72.44/70.32	|	68.84/67.76|
| [PV-RCNN + Occupancy-MAE]() | 75.94/75.28 |	67.94/67.34| 74.02/63.48 | 64.94/55.57 | 67.21/66.49 |	65.62/63.02|
| [PVRCNN + MV-JAR]() | -/- |	61.88/61.45| -/- | 66.98/59.02 | -/- |	57.98/57.00|
| [PVRCNN + MAELi]() | -/- |	-/67.34 | -/- | -/56.32 | -/- |	-/62.76 |
| [PVRCNN + PropCont]() | -/- |	-/65.47 | -/- | -/49.51 | -/- |	-/62.86 |


Here we also provide the performance of several models trained and finetuned on 100% training set while pretraining has been the same on 20% of the data:


| Performance@(train with 100\% Data)                                                       | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |  
|-------------------------------------------------------------------------------------------|----------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| [PV-RCNN (CenterHead)](tools/cfgs/waymo_models/pv_rcnn_with_centerhead_rpn.yaml)          | 78.00/77.50 | 69.43/68.98 | 79.21/73.03 | 70.42/64.72 | 71.46/70.27 | 68.95/67.79 |
| [PV-RCNN (CenterHead + Radial-MAE)]()          | 78.10/77.65 | 69.69/69.25 | 79.61/73.69 | 71.26/65.72 | 71.94/70.87 | 69.32/68.28 |



### nuScenes Dataset 

All models are trained with 2 RTX 6000 ADA GPUs and are available for download.

|                                                                                                    | Modality |  mATE |  mASE  |  mAOE  | mAVE  | mAAE  |  mAP  |  NDS   |                                              download                                              | 
|----------------------------------------------------------------------------------------------------|----------|------:|:------:|:------:|:-----:|:-----:|:-----:|:------:|:--------------------------------------------------------------------------------------------------:|
| [PointPillar-MultiHead](tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml)                         | LiDAR    | 33.87 | 26.00  | 32.07  | 28.74 | 20.15 | 44.63 | 58.23  |  [model-23M](https://drive.google.com/file/d/1p-501mTWsq0G9RzroTWSXreIMyTUUpBM/view?usp=sharing)   | 
| [SECOND-MultiHead (CBGS)](tools/cfgs/nuscenes_models/cbgs_second_multihead.yaml)                   | LiDAR    | 31.15 | 25.51  | 26.64  | 26.26 | 20.46 | 50.59 | 62.29  |  [model-35M](https://drive.google.com/file/d/1bNzcOnE3u9iooBFMk2xK7HqhdeQ_nwTq/view?usp=sharing)   |
| [CenterPoint](tools/cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint.yaml)     | LiDAR    | 30.11 | 25.55  | 38.28  | 21.94 | 18.87 | 56.03 | 64.54  |  [model-34M](https://drive.google.com/file/d/1Cz-J1c3dw7JAWc25KRG1XQj8yCaOlexQ/view?usp=sharing)   |
| [CenterPoint + Radial-MAE]() | LiDAR    | 29.73 | 25.71  | 34.16  | 20.02 | 17.91 | 59.20 | 66.85  |  [model-34M]()   |
| [TransFusion-L](tools/cfgs/nuscenes_models/transfusion_lidar.yaml) | LiDAR    | - | - | - | - | - | - | -  | [model-32M]() |
| [TransFusion-L + Radial-MAE]() | LiDAR    | 29.88 | 25.49  | 29.02  | 29.10 | 19.04 | 62.80 | 68.15  | [model-32M]() |
| [BEVFusion](tools/cfgs/nuscenes_models/bevfusion.yaml)     | LiDAR + Camera    |  28.26  |  25.43  |  28.88  |  26.80  | 18.67  |  65.91  |  70.20  | [model-157M]() |
| [BEVFusion + Radial-MAE](tools/cfgs/nuscenes_models/bevfusion.yaml)                                | LiDAR + Camera    |  28.31  |  25.54  |  29.57  |  25.87  | 18.60  |  66.40  |  70.41  | [model-157M]() |



##  License
Our codes are released under the Apache 2.0 license.

## Acknowledgement

This project is mainly based on the following codebases. Thanks for their great works!

* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
* [SPCONV](https://github.com/traveller59/spconv)

