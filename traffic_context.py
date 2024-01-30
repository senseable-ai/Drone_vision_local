import pandas as pd
import numpy as np
from matplotlib.path import Path

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

# Define the ROIs for each lane
rois = [
    [(0, 430), (745, 495), (1430, 520), (1430, 535), (1050, 530), (740, 515), (0, 447)], #lane 1
    [(0, 447), (740, 515), (1050, 530), (1430, 535), (1430, 551), (1030, 545), (735, 530), (0, 462)], #lane 2
    [(0, 462), (735, 530), (1030, 545), (1430, 551), (1430, 565), (1020, 560), (735, 545), (0, 480)] #lane 3
]

# Load the CSV file
csv_file_path = '/Users/user/Desktop/drone_vision_local/clip/clip1.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file_path)

# Find surrounding vehicles using x coordinate for ID *
surrounding_vehicles = final_correct_surrounding_vehicles_using_x_coordinate(df, 8, rois, roi_size=(300, 50))

# Save the result to a CSV file
output_csv_path = 'surrounding_vehicles_for_id_8.csv'
surrounding_vehicles.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")
