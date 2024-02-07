import pandas as pd
import numpy as np
from matplotlib.path import Path
import json
import os

def is_point_in_roi(point, roi):
    """
    Check if a given point is inside the specified ROI.
    """
    roi_path = Path(roi)
    return roi_path.contains_point(point)

def final_correct_surrounding_vehicles_using_x_coordinate(df, target_id, rois, roi_size=(300, 50)):
    target_vehicle = df[df['ID'] == target_id]
    surrounding_vehicles_data = []

    for _, target in target_vehicle.iterrows():
        frame = target['frame']
        lane = target['lane']
        center_x, center_y = target['center_x'], target['center_y']

        # Define new ROI boundaries based on x coordinate
        roi_left = center_x - roi_size[0] / 2
        roi_right = center_x + roi_size[0] / 2

        # Find vehicles in the same frame and within the new ROI
        frame_vehicles = df[(df['frame'] == frame) &
                            (df['center_x'] >= roi_left) & (df['center_x'] <= roi_right)]

        surrounding_vehicles = {
            'frame': frame,
            'leading': None,
            'right_leading': None,
            'left_leading': None,
            'following': None,
            'right_following': None,
            'left_following': None
        }

        for _, vehicle in frame_vehicles.iterrows():
            if vehicle['ID'] == target_id:
                continue

            vehicle_lane = None
            for lane_idx, roi in enumerate(rois, start=1):
                if is_point_in_roi((vehicle['center_x'], vehicle['center_y']), roi):
                    vehicle_lane = lane_idx
                    break

            if vehicle_lane is None:
                continue

            # Correctly determine the position of the vehicle relative to the target vehicle using x coordinate
            if vehicle['center_x'] > center_x:  # Following vehicles
                if vehicle_lane == lane:
                    surrounding_vehicles['following'] = vehicle['ID']
                elif vehicle_lane < lane:  # Right side
                    surrounding_vehicles['right_following'] = vehicle['ID']
                elif vehicle_lane > lane:  # Left side
                    surrounding_vehicles['left_following'] = vehicle['ID']
            elif vehicle['center_x'] < center_x:  # Leading vehicles
                if vehicle_lane == lane:
                    surrounding_vehicles['leading'] = vehicle['ID']
                elif vehicle_lane < lane:  # Right side
                    surrounding_vehicles['right_leading'] = vehicle['ID']
                elif vehicle_lane > lane:  # Left side
                    surrounding_vehicles['left_leading'] = vehicle['ID']

        surrounding_vehicles_data.append(surrounding_vehicles)

    return pd.DataFrame(surrounding_vehicles_data)

# Load ROIs from JSON file
roi_file_path = '/mnt/data/roi.json'
with open(roi_file_path, 'r') as file:
    rois = json.load(file)

# Vehicle ID mapping
vehicle_id_mapping = {1:38, 2:42, 3:12, 4:10, 5:20, 6:34, 10:7, 11:2, 12:11, 13:26, 14:13, 15:39, 16:28, 17:2, 18:4, 19:8, 20:30,
                      21:5, 22:31, 23:17, 24:26, 25:11, 26:15, 28:23, 29:1, 30:8, 31:24, 32:2, 33:21, 35:13, 36:18, 37:15, 38:29,
                      39:13, 40:9, 41:10, 42:8, 43:16, 44:13, 45:8, 46:27, 47:45, 48:9, 49:25, 50:6, 51:4, 52:5, 53:14, 54:31, 55:3,
                      56:52, 59:17, 60:8, 61:17, 62:41, 63:5, 64:64, 66:35, 67:5, 68:10, 69:9, 70:9, 71:7, 72:24, 73:13, 74:16, 75:27,
                      76:42, 77:6, 78:51, 79:8, 80:2, 81:17, 82:7, 83:45, 85:12, 86:26, 87:49, 88:59, 89:24, 90:93, 91:54, 92:21, 93:56,
                      94:15, 95:15, 96:48, 97:58, 98:29, 99:35, 100:7, 101:13, 102:19, 103:15, 104:2, 106:1, 107:6, 108:8, 110:5, 111:2,
                      112:25, 118:4}

# Process each clip
base_path = '/Users/user/Desktop/drone_vision_local/LC'
for clip_number in range(1, 120):
    try:
        csv_file_path = f'{base_path}/clip{clip_number}/clip{clip_number}_sv.csv'
        if not os.path.exists(csv_file_path):
            continue  # Skip if file does not exist
        
        df = pd.read_csv(csv_file_path)
        target_vehicle_id = vehicle_id_mapping.get(clip_number)
        if target_vehicle_id is not None:
            surrounding_vehicles = final_correct_surrounding_vehicles_using_x_coordinate(df, target_vehicle_id, rois, roi_size=(300, 50))
            output_csv_path = f'/mnt/data/surrounding_vehicles_for_clip{clip_number}.csv'
            surrounding_vehicles.to_csv(output_csv_path, index=False)
            print(f"Results saved to {output_csv_path}")
    except Exception as e:
        print(f"Error processing clip {clip_number}: {e}")
