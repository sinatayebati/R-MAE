from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os

def load_tensorboard_data(log_dir):
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()
    return event_acc

def tensorboard_logs_to_csv(log_dirs, output_csv_path):
    data = []

    for i, log_dir in enumerate(log_dirs):
        event_acc = load_tensorboard_data(log_dir)

        for tag in event_acc.Tags()['scalars']:
            scalar_data = event_acc.Scalars(tag)
            steps = [entry.step for entry in scalar_data]
            values = [entry.value for entry in scalar_data]

            run_name = f'Run_{i+1}'
            data.extend(list(zip(steps, values, [tag] * len(steps), [run_name] * len(steps))))

    df = pd.DataFrame(data, columns=['Step', 'Value', 'Tag', 'Run'])
    df.to_csv(output_csv_path, index=False)
    print(f"CSV file saved to: {output_csv_path}")

if __name__ == "__main__":
    # Replace these with the actual paths to your TensorBoard log directories
    log_dir1 = '/hdd_10tb/public/datasets/Occupancy-MAE/output/kitti_models/second/default/eval/eval_with_train/tensorboard_val/events.out.tfevents.1703749923.aeonlab2024'
    # log_dir2 = 'experiments/finetune_modelnet/cfgs/TFBoard/experiments/train/events.out.tfevents.1702264077.aeonlab2024'

    # Specify the path where you want to save the CSV file
    output_csv_path = '/hdd_10tb/public/datasets/Occupancy-MAE/output/kitti_models/second/default/eval/eval_with_train/tensorboard_val/'

    # Provide the directories and output_csv_path to the function
    #tensorboard_logs_to_csv([log_dir1, log_dir2], output_csv_path)
    tensorboard_logs_to_csv([log_dir1], output_csv_path)
