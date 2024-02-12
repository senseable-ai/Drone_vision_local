import pandas as pd
import numpy as np
import os

vehicle_id_mapping = {1: 2, 2: 25, 3: 35, 4: 11, 5: 17, 6: 18, 10: 4, 11: 5, 12: 6, 13: 7, 14: 2, 15: 2, 16: 11, 17: 3,
                      18: 2, 19: 5, 20: 16, 21: 20, 22: 26, 23: 11, 24: 19, 25: 4, 26: 4, 27: 4, 28: 35, 29: 8, 30: 4, 
                      31: 40, 32: 1, 33: 14, 34: 38, 35: 14, 36: 25, 37: 19, 38: 16, 39: 15, 40: 16, 41: 4, 42: 14, 
                      43: 11, 44: 14, 45: 21, 46: 7, 47: 34, 48: 16, 49: 43, 50: 2, 51: 10, 52: 9, 53: 10, 54: 14, 55: 6,
                      56: 11, 59: 23, 60: 1, 61: 8, 62: 7, 63: 6, 64: 6, 65: 4, 66: 26, 67: 2, 68: 12, 69: 2, 70: 25, 
                      71: 11, 72: 21, 73: 16, 74: 1, 75: 26, 76: 6, 77: 11, 78: 21, 79: 1, 80: 10, 81: 22, 82: 18, 83: 10, 
                      85: 1, 86: 4, 87: 2, 88: 7, 89: 4, 90: 11, 91: 24, 92: 2, 93: 7, 94: 3, 95: 1, 96: 19, 97: 2, 98: 1, 
                      99: 15, 100: 12, 101: 1, 102: 4, 103: 4, 104: 3, 105: 1, 106: 3, 107: 1, 108: 1, 109: 9, 110: 9, 111: 45, 
                      112: 27, 113: 7, 114: 31, 115: 6, 116: 4, 118: 8} 

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
    surrounding_vehicles_path = f'/Users/user/Desktop/drone_vision_local/LK/clip{clip_num}/clip{clip_num}_sv.csv'
    
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
    output_csv_path = f'/Users/user/Desktop/drone_vision_local/LK/clip{clip_num}/clip{clip_num}_result.csv'
    final_data.to_csv(output_csv_path, index=False)
    print(f"Processed and saved clip {clip_num}.")

# Loop through clips 1 to 119 with exception handling
for i in range(1, 120):
    try:
        process_clip(i, vehicle_id_mapping)
    except Exception as e:
        print(f"An error occurred processing clip {i}: {e}")