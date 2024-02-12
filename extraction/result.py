import pandas as pd
import numpy as np
import os

vehicle_id_mapping = {1:38, 2:42, 3:12, 4:10, 5:20, 6:34, 10:7, 11:2, 12:11, 13:26, 14:13, 15:39, 16:28, 17:2, 18:4, 19:8, 20:30,
                      21:5, 22:31, 23:17, 24:26, 25:11, 26:15, 28:23, 29:1, 30:8, 31:24, 32:2, 33:21, 35:13, 36:18, 37:15, 38:29,
                      39:13, 40:9, 41:10, 42:8, 43:16, 44:13, 45:8, 46:27, 47:45, 48:9, 49:25, 50:6, 51:4, 52:5, 53:14, 54:31, 55:3,
                      56:52, 59:17, 60:8, 61:17, 62:41, 63:5, 64:64, 66:35, 67:5, 68:10, 69:9, 70:9, 71:7, 72:24, 73:13, 74:16, 75:27,
                      76:42, 77:6, 78:51, 79:8, 80:2, 81:17, 82:7, 83:45, 85:12, 86:26, 87:49, 88:59, 89:24, 90:93, 91:54, 92:21, 93:56,
                      94:15, 95:15, 96:48, 97:58, 98:29, 99:35, 100:7, 101:13, 102:19, 103:15, 104:2, 106:1, 107:6, 108:8, 110:5, 111:2,
                      112:25, 118:4}  

base_path = '/Users/user/Desktop/drone_vision_local/LC'  # Adjusted to use the current environment's file path

# Function to check file existence
def file_exists(path):
    return os.path.exists(path)

# Function to calculate euclidean distance
def calculate_euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Function to process each clip
def process_clip(clip_num, vehicle_id_mapping):
    clip_speed_path = f'{base_path}/clip{clip_num}/clip{clip_num}_speed.csv'
    surrounding_vehicles_path = f'{base_path}/clip{clip_num}/clip{clip_num}_sv.csv'
    
    # Check if files exist
    if not file_exists(clip_speed_path) or not file_exists(surrounding_vehicles_path):
        print(f"Files for clip {clip_num} not found. Skipping...")
        return
    
    # Load data
    clip_speed_data = pd.read_csv(clip_speed_path)
    surrounding_vehicles_data = pd.read_csv(surrounding_vehicles_path)
    
    # Apply vehicle ID mapping
    target_vehicle_id = vehicle_id_mapping.get(clip_num, None)
    if target_vehicle_id is None:
        print(f"No vehicle ID mapping for clip {clip_num}. Skipping...")
        return
    
    # Filter data based on the vehicle ID mapping
    id_data = clip_speed_data[clip_speed_data['ID'] == target_vehicle_id]
    
    # Merge data
    merged_data = pd.merge(surrounding_vehicles_data, id_data, on='frame', how='left')
    
    # Rename columns
    merged_data.rename(columns={
        'leading': 'preceding', 'leading_rel_v_x': 'preceding_rel_v_x', 'leading_rel_v_y': 'preceding_rel_v_y', 'leading_distance': 'preceding_distance',
        'right_leading': 'right_preceding', 'right_leading_rel_v_x': 'right_preceding_rel_v_x', 'right_leading_rel_v_y': 'right_preceding_rel_v_y', 'right_leading_distance': 'right_preceding_distance',
        'left_leading': 'left_preceding', 'left_leading_rel_v_x': 'left_preceding_rel_v_x', 'left_leading_rel_v_y': 'left_preceding_rel_v_y', 'left_leading_distance': 'left_preceding_distance'}, inplace=True)
    
    # Calculate relative speeds and distances
    for col in ['preceding', 'right_preceding', 'left_preceding', 'following', 'right_following', 'left_following']:
        merged_data[f'{col}_rel_v_x'] = np.nan
        merged_data[f'{col}_rel_v_y'] = np.nan
        merged_data[f'{col}_distance'] = np.nan

        for index, row in merged_data.iterrows():
            surr_vehicle_id = row[col]
            if pd.notnull(surr_vehicle_id):
                surr_data = clip_speed_data[clip_speed_data['ID'] == surr_vehicle_id]
                if not surr_data.empty:
                    surr_v_x, surr_v_y = surr_data.iloc[0]['v_x'], surr_data.iloc[0]['v_y']
                    merged_data.at[index, f'{col}_rel_v_x'] = surr_v_x - row['v_x']
                    merged_data.at[index, f'{col}_rel_v_y'] = surr_v_y - row['v_y']
                    # Calculate distance if position data is available
                    if 'front_x' in row and 'front_y' in row and 'back_x' in row and 'back_y' in row:
                        if col.endswith('preceding'):
                            surr_front_x, surr_front_y = surr_data.iloc[0]['front_x'], surr_data.iloc[0]['front_y']
                            distance = calculate_euclidean_distance(row['front_x'], row['front_y'], surr_front_x, surr_front_y)
                        else:
                            surr_back_x, surr_back_y = surr_data.iloc[0]['back_x'], surr_data.iloc[0]['back_y']
                            distance = calculate_euclidean_distance(row['back_x'], row['back_y'], surr_back_x, surr_back_y)
                        merged_data.at[index, f'{col}_distance'] = distance

    # Select specific columns to form the final dataframe
    final_columns = [
        'frame', 'ID', 'center_x', 'center_y', 'v_x', 'v_y', 'a_x', 'a_y', 'lane',
        'preceding_rel_v_x', 'preceding_rel_v_y', 'right_preceding_rel_v_x', 'right_preceding_rel_v_y',
        'left_preceding_rel_v_x', 'left_preceding_rel_v_y', 'following_rel_v_x', 'following_rel_v_y',
        'right_following_rel_v_x', 'right_following_rel_v_y', 'left_following_rel_v_x', 'left_following_rel_v_y',
        'preceding_distance', 'right_preceding_distance', 'left_preceding_distance', 
        'following_distance', 'right_following_distance', 'left_following_distance'
    ]
    final_data = merged_data[final_columns]

    # Save to CSV
    output_csv_path = f'{base_path}/clip{clip_num}/clip{clip_num}_result.csv'
    final_data.to_csv(output_csv_path, index=False)
    print(f"Processed and saved clip {clip_num}.")

# Loop through clips 1 to 119 with exception handling
for i in range(1, 120):
    try:
        process_clip(i, vehicle_id_mapping)
    except Exception as e:
        print(f"An error occurred processing clip {i}: {e}")