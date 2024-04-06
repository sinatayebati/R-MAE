from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

def load_tensorboard_data(log_dir):
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()
    return event_acc

def plot_tensorboard_scalars(log_dirs, labels, tags, save_path=None):
    plt.figure(figsize=(10, 6))

    for i, log_dir in enumerate(log_dirs):
        event_acc = load_tensorboard_data(log_dir)

        for tag in tags:
            if tag not in event_acc.Tags()['scalars']:
                raise KeyError(f"Key {tag} was not found in Reservoir")

            scalar_data = event_acc.Scalars(tag)

            steps = [entry.step for entry in scalar_data]
            values = [entry.value for entry in scalar_data]

            plt.plot(steps, values, label=f'{labels[i]} - {tag}')

    plt.xlabel('Steps or Epochs')
    plt.ylabel('Values')
    plt.legend()
    plt.title('Comparison of Scalar Metrics Between Runs')

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    log_dir1 = '/hdd_10tb/sina/Radial_MAE/output/nuscenes_models/radial_mae_res_nuscenes/default/tensorboard/events.out.tfevents.1711039832.aeonlab2024'
    #log_dir2 = 'path/to/log_dir2'
    #log_dir3 = 'path/to/log_dir3'

    #labels = ['Run 1', 'Run 2', 'Run 3']
    labels = ['Run 1']
    tags = ['train/loss', 'train/loss_rpn']  # Add your tags here

    save_path = '/hdd_10tb/sina/Radial_MAE/output/nuscenes_models/radial_mae_res_nuscenes/default2'

    #plot_tensorboard_scalars([log_dir1, log_dir2, log_dir3], labels, tags, save_path)
    plot_tensorboard_scalars([log_dir1], labels, tags, save_path)
