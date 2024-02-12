import pandas as pd
import numpy as np
from matplotlib.path import Path
import json
import os

def is_point_in_roi(point, roi):
    roi_path = Path(roi)
    return roi_path.contains_point(point)

def final_correct_surrounding_vehicles_using_x_coordinate(df, target_id, rois, roi_size=(300, 50)):
    target_vehicle = df[df['ID'] == target_id]
    surrounding_vehicles_data = []

    for _, target in target_vehicle.iterrows():
        frame = target['frame']
        lane = target['lane']
        center_x, center_y = target['center_x'], target['center_y']

        roi_left = center_x - roi_size[0] / 2
        roi_right = center_x + roi_size[0] / 2

        frame_vehicles = df[(df['frame'] == frame) & (df['center_x'] >= roi_left) & (df['center_x'] <= roi_right)]

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

            if vehicle['center_x'] > center_x:
                if vehicle_lane == lane:
                    surrounding_vehicles['following'] = vehicle['ID']
                elif vehicle_lane < lane:
                    surrounding_vehicles['right_following'] = vehicle['ID']
                elif vehicle_lane > lane:
                    surrounding_vehicles['left_following'] = vehicle['ID']
            elif vehicle['center_x'] < center_x:
                if vehicle_lane == lane:
                    surrounding_vehicles['leading'] = vehicle['ID']
                elif vehicle_lane < lane:
                    surrounding_vehicles['right_leading'] = vehicle['ID']
                elif vehicle_lane > lane:
                    surrounding_vehicles['left_leading'] = vehicle['ID']

        surrounding_vehicles_data.append(surrounding_vehicles)

    return pd.DataFrame(surrounding_vehicles_data)

# JSON 파일에서 ROI 로드
roi_file_path = '/Users/user/Desktop/drone_vision_local/roi/roi.json'
with open(roi_file_path, 'r') as file:
    rois_data = json.load(file)

base_path = '/Users/user/Desktop/drone_vision_local/LC'
vehicle_id_mapping = {1: 2, 2: 25, 3: 35, 4: 11, 5: 17, 6: 18, 10: 4, 11: 5, 12: 6, 13: 7, 14: 2, 15: 2, 16: 11, 17: 3,
                      18: 2, 19: 5, 20: 16, 21: 20, 22: 26, 23: 11, 24: 19, 25: 4, 26: 4, 27: 4, 28: 35, 29: 8, 30: 4, 
                      31: 40, 32: 1, 33: 14, 34: 38, 35: 14, 36: 25, 37: 19, 38: 16, 39: 15, 40: 16, 41: 4, 42: 14, 
                      43: 11, 44: 14, 45: 21, 46: 7, 47: 34, 48: 16, 49: 43, 50: 2, 51: 10, 52: 9, 53: 10, 54: 14, 55: 6,
                      56: 11, 59: 23, 60: 1, 61: 8, 62: 7, 63: 6, 64: 6, 65: 4, 66: 26, 67: 2, 68: 12, 69: 2, 70: 25, 
                      71: 11, 72: 21, 73: 16, 74: 1, 75: 26, 76: 6, 77: 11, 78: 21, 79: 1, 80: 10, 81: 22, 82: 18, 83: 10, 
                      85: 1, 86: 4, 87: 2, 88: 7, 89: 4, 90: 11, 91: 24, 92: 2, 93: 7, 94: 3, 95: 1, 96: 19, 97: 2, 98: 1, 
                      99: 15, 100: 12, 101: 1, 102: 4, 103: 4, 104: 3, 105: 1, 106: 3, 107: 1, 108: 1, 109: 9, 110: 9, 111: 45, 
                      112: 27, 113: 7, 114: 31, 115: 6, 116: 4, 118: 8}

for clip_number, target_id in vehicle_id_mapping.items():
    try:
        csv_file_path = f'{base_path}/clip{clip_number}/clip{clip_number}_speed.csv'
        if not os.path.exists(csv_file_path):
            continue
        
        df = pd.read_csv(csv_file_path)
        rois = rois_data[str(clip_number)]  # 클립 번호에 맞는 ROI 로드
        
        surrounding_vehicles = final_correct_surrounding_vehicles_using_x_coordinate(df, target_id, rois, roi_size=(300, 50))
        output_csv_path = f'/Users/user/Desktop/drone_vision_local/LK/clip{clip_number}/clip{clip_number}_sv.csv'
        surrounding_vehicles.to_csv(output_csv_path, index=False)
        print(f"Results saved to {output_csv_path}")
    except Exception as e:
        print(f"Error processing clip {clip_number}: {e}")
