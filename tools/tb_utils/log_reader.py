import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

# Step 1: Read Tensorboard Log
def load_tensorboard_data(log_dir):
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()
    return event_acc

# Step 2: Extract Data
def extract_scalars(event_acc, scalar_tags):
    data = {}
    for tag in scalar_tags:
        scalar_data = event_acc.Scalars(tag)
        steps = [entry.step for entry in scalar_data]
        values = [entry.value for entry in scalar_data]
        data[tag] = pd.Series(values, index=steps)
    return pd.DataFrame(data)

# Step 3: Create a Table
def create_table(data_frame):
    # You can also export this DataFrame to a CSV file if needed
    data_frame.to_csv(save_data_frame_path)
    return data_frame

# Main Execution
log_dir = '/hdd_10tb/public/datasets/Occupancy-MAE/output/kitti_models/second_mae_90/default/eval/eval_with_train/tensorboard_val/events.out.tfevents.1704237165.aeonlab2024'  # Replace with your Tensorboard log path
save_data_frame_path = '/hdd_10tb/public/datasets/Occupancy-MAE/output/kitti_models/second_mae_90/default/eval/eval_with_train/data_frame.csv'  # Replace with your desired save path for data_frame
event_acc = load_tensorboard_data(log_dir)

# Include all provided scalar tags
scalar_tags = [
    'recall/roi_0.3', 'recall/rcnn_0.3', 'recall/roi_0.5', 'recall/rcnn_0.5', 
    'recall/roi_0.7', 'recall/rcnn_0.7', 'Car_aos/easy_R40', 'Car_aos/moderate_R40', 
    'Car_aos/hard_R40', 'Car_3d/easy_R40', 'Car_3d/moderate_R40', 'Car_3d/hard_R40', 
    'Car_bev/easy_R40', 'Car_bev/moderate_R40', 'Car_bev/hard_R40', 'Car_image/easy_R40', 
    'Car_image/moderate_R40', 'Car_image/hard_R40', 'Pedestrian_aos/easy_R40', 
    'Pedestrian_aos/moderate_R40', 'Pedestrian_aos/hard_R40', 'Pedestrian_3d/easy_R40', 
    'Pedestrian_3d/moderate_R40', 'Pedestrian_3d/hard_R40', 'Pedestrian_bev/easy_R40', 
    'Pedestrian_bev/moderate_R40', 'Pedestrian_bev/hard_R40', 'Pedestrian_image/easy_R40', 
    'Pedestrian_image/moderate_R40', 'Pedestrian_image/hard_R40', 'Cyclist_aos/easy_R40', 
    'Cyclist_aos/moderate_R40', 'Cyclist_aos/hard_R40', 'Cyclist_3d/easy_R40', 
    'Cyclist_3d/moderate_R40', 'Cyclist_3d/hard_R40', 'Cyclist_bev/easy_R40', 
    'Cyclist_bev/moderate_R40', 'Cyclist_bev/hard_R40', 'Cyclist_image/easy_R40', 
    'Cyclist_image/moderate_R40', 'Cyclist_image/hard_R40'
]

data_frame = extract_scalars(event_acc, scalar_tags)

# Save the data_frame as a CSV file
table = create_table(data_frame)
