from tensorboard.backend.event_processing import event_accumulator

def list_scalar_tags(log_dir):
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()
    tags = event_acc.Tags()['scalars']
    return tags

if __name__ == "__main__":
    # Replace this with the actual path to your TensorBoard log directory
    log_dir = '/hdd_10tb/public/datasets/Occupancy-MAE/output/kitti_models/pretrain_mae_radial_kitti_90/default/tensorboard/events.out.tfevents.1704144777.aeonlab2024'

    # Print all available scalar tags in the specified log directory
    scalar_tags = list_scalar_tags(log_dir)
    print("Available Scalar Tags:")
    print(scalar_tags)
