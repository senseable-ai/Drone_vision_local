import pandas as pd
import numpy as np


file_path = '/Users/user/Desktop/drone_vision_local/clip/clip1.csv'  
data = pd.read_csv(file_path)


id_8_data = data[data['ID'] == 8]


roi_width, roi_height = 125, 50
id_8_rois = []
for index, row in id_8_data.iterrows():
    center_x, center_y = row['center_x'], row['center_y']
    x1, y1 = center_x - roi_width / 2, center_y - roi_height / 2
    x2, y2 = center_x + roi_width / 2, center_y + roi_height / 2
    id_8_rois.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

def find_neighboring_ids(frame_data, target_id, roi):

    frame_data = frame_data[frame_data['ID'] != target_id]


    preceding_id = preceding_right_id = preceding_left_id = None
    following_id = following_left_id = following_right_id = None


    x1_target, y1_target, x2_target, y2_target = roi[0][0], roi[0][1], roi[2][0], roi[2][1]
    center_x_target, center_y_target = (x1_target + x2_target) / 2, (y1_target + y2_target) / 2


    for _, row in frame_data.iterrows():
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        center_x, center_y = row['center_x'], row['center_y']

        # 차량이 대상 차량의 전방, 후방, 좌측 또는 우측에 있는지 확인
        if center_x < center_x_target:  # 차량이 앞쪽에 존재
            if center_y < center_y_target:  # 차량이 좌측에 존재
                preceding_left_id = row['ID']
            elif center_y > center_y_target:  # 차량이 우측에 존재
                preceding_right_id = row['ID']
            else:
                preceding_id = row['ID']
        elif center_x > center_x_target:  # 차량이 후방에 존재
            if center_y < center_y_target:  # 차량이 좌측에 존재
                following_left_id = row['ID']
            elif center_y > center_y_target:  # 차량이 우측에 존재
                following_right_id = row['ID']
            else:
                following_id = row['ID']

    return {
        'preceding_id': preceding_id,
        'preceding_right_id': preceding_right_id,
        'preceding_left_id': preceding_left_id,
        'following_id': following_id,
        'following_left_id': following_left_id,
        'following_right_id': following_right_id
    }


output_file_path = 'output_id_8_neighboring_ids.csv'


frame_to_roi_map = {row.frame: roi for row, roi in zip(id_8_data.itertuples(), id_8_rois)}

with open(output_file_path, 'w') as file:
    for frame, roi in frame_to_roi_map.items():
        frame_data = data[data['frame'] == frame]
        neighboring_ids = find_neighboring_ids(frame_data, 8, roi)


        file.write(f"Frame: {frame}, ")
        file.write(f"Preceding ID: {neighboring_ids['preceding_id']}, ")
        file.write(f"Preceding Right ID: {neighboring_ids['preceding_right_id']}, ")
        file.write(f"Preceding Left ID: {neighboring_ids['preceding_left_id']}, ")
        file.write(f"Following ID: {neighboring_ids['following_id']}, ")
        file.write(f"Following Left ID: {neighboring_ids['following_left_id']}, ")
        file.write(f"Following Right ID: {neighboring_ids['following_right_id']}\n")
