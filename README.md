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


### Waymo Open Dataset


### nuScenes Dataset 


##  License
Our codes are released under the Apache 2.0 license.

## Acknowledgement

This project is mainly based on the following codebases. Thanks for their great works!

* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
* [SPCONV](https://github.com/traveller59/spconv)

