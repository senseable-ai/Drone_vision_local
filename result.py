import pandas as pd
import numpy as np

vehicle_id_mapping = {1:38, 2:42, 3:12, 4:10, 5:20, 6:34, 10:7, 11:2, 12:11, 13:26, 14:13, 15:39, 16:28, 17:2, 18:4, 19:8, 20:30,
                      21:5, 22:31, 23:17, 24:26, 25:11, 26:15, 28:23, 29:1, 30:8, 31:24, 32:2, 33:21, 35:13, 36:18, 37:15, 38:29,
                      39:13, 40:9, 41:10, 42:8, 43:16, 44:13, 45:8, 46:27, 47:45, 48:9, 49:25, 50:6, 51:4, 52:5, 53:14, 54:31, 55:3,
                      56:52, 59:17, 60:8, 61:17, 62:41, 63:5, 64:64, 66:35, 67:5, 68:10, 69:9, 70:9, 71:7, 72:24, 73:13, 74:16, 75:27,
                      76:42, 77:6, 78:51, 79:8, 80:2, 81:17, 82:7, 83:45, 85:12, 86:26, 87:49, 88:59, 89:24, 90:93, 91:54, 92:21, 93:56,
                      94:15, 95:15, 96:48, 97:58, 98:29, 99:35, 100:7, 101:13, 102:19, 103:15, 104:2, 106:1, 107:6, 108:8, 110:5, 111:2,
                      112:25, 118:4}  

for i in range(1, 120):
    vehicle_id = vehicle_id_mapping.get(i)  
    if vehicle_id is None:
        print(f"No vehicle ID mapping for clip {i}. Skipping...")
        continue
    
    try:
        clip_speed_path = f'C:/Users/user/Desktop/drone_vision_local/LC/clip{i}/clip{i}_speed.csv'
        surrounding_vehicles_path = f'C:/Users/user/Desktop/drone_vision_local/LC/clip{i}/clip{i}_sv.csv'
        
        clip_speed_data = pd.read_csv(clip_speed_path)
        surrounding_vehicles_data = pd.read_csv(surrounding_vehicles_path)
        
        id_data = clip_speed_data[clip_speed_data['ID'] == vehicle_id]

        merged_data = pd.merge(surrounding_vehicles_data, id_data, on='frame', how='left')

        merged_data.rename(columns={
            'leading': 'preceding',
            'right_leading': 'right_preceding',
            'left_leading': 'left_preceding',
            'leading_rel_v_x': 'preceding_rel_v_x',
            'leading_rel_v_y': 'preceding_rel_v_y',
            'leading_distance': 'preceding_distance',
            'right_leading_rel_v_x': 'right_preceding_rel_v_x',
            'right_leading_rel_v_y': 'right_preceding_rel_v_y',
            'right_leading_distance': 'right_preceding_distance',
            'left_leading_rel_v_x': 'left_preceding_rel_v_x',
            'left_leading_rel_v_y': 'left_preceding_rel_v_y',
            'left_leading_distance': 'left_preceding_distance'
        }, inplace=True)

        final_columns = [
            'frame', 'ID', 'center_x', 'center_y', 'v_x', 'v_y', 'a_x', 'a_y', 'lane', 
            'preceding_rel_v_x', 'preceding_rel_v_y', 'right_preceding_rel_v_x', 'right_preceding_rel_v_y',
            'left_preceding_rel_v_x', 'left_preceding_rel_v_y', 'following_rel_v_x', 'following_rel_v_y',
            'right_following_rel_v_x', 'right_following_rel_v_y', 'left_following_rel_v_x', 'left_following_rel_v_y',
            'preceding_distance', 'right_preceding_distance', 'left_preceding_distance', 
            'following_distance', 'right_following_distance', 'left_following_distance'
        ]
        final_data = merged_data[final_columns]

        output_csv_path = f'C:/Users/user/Desktop/drone_vision_local/LC/clip{i}/clip{i}_ID{vehicle_id}_result.csv'
        final_data.to_csv(output_csv_path, index=False)
        
    except FileNotFoundError:
        print(f"File not found for clip {i}. Skipping...")
