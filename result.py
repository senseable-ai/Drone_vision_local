import pandas as pd
import numpy as np

# 데이터 로드
clip1_speed_path = 'C:/Users/user/Desktop/drone_vision_local/clip/clip1/clip1_speed.csv'
surrounding_vehicles_path = 'C:/Users/user/Desktop/drone_vision_local/clip/clip1/surrounding_vehicles_for_id_8.csv'
clip1_speed_data = pd.read_csv(clip1_speed_path)
surrounding_vehicles_data = pd.read_csv(surrounding_vehicles_path)

# ID 8번 차량 데이터 필터링
id_8_data = clip1_speed_data[clip1_speed_data['ID'] == 8]

# 데이터 병합
merged_data = pd.merge(surrounding_vehicles_data, id_8_data, on='frame', how='left')

# 'leading'을 'preceding'으로, 'right_leading'을 'right_preceding'으로, 'left_leading'을 'left_preceding'으로 이름 변경
merged_data.rename(columns={
    'leading': 'preceding', 'leading_rel_v_x': 'preceding_rel_v_x', 'leading_rel_v_y': 'preceding_rel_v_y', 'leading_distance': 'preceding_distance',
    'right_leading': 'right_preceding', 'right_leading_rel_v_x': 'right_preceding_rel_v_x', 'right_leading_rel_v_y': 'right_preceding_rel_v_y', 'right_leading_distance': 'right_preceding_distance',
    'left_leading': 'left_preceding', 'left_leading_rel_v_x': 'left_preceding_rel_v_x', 'left_leading_rel_v_y': 'left_preceding_rel_v_y', 'left_leading_distance': 'left_preceding_distance'}, inplace=True)

# 상대 속도 계산 함수
def get_speed(vehicle_id, frame, data):
    vehicle_data = data[(data['ID'] == vehicle_id) & (data['frame'] == frame)]
    if not vehicle_data.empty:
        return vehicle_data.iloc[0]['v_x'], vehicle_data.iloc[0]['v_y']
    else:
        return None, None

# 상대 속도 계산 및 새로운 열 추가
surrounding_columns = ['preceding', 'right_preceding', 'left_preceding', 'following', 'right_following', 'left_following']
for col in surrounding_columns:
    merged_data[f'{col}_rel_v_x'] = 0.0
    merged_data[f'{col}_rel_v_y'] = 0.0

for index, row in merged_data.iterrows():
    id_8_v_x = row['v_x']
    id_8_v_y = row['v_y']

    for col in surrounding_columns:
        surr_vehicle_id = row[col]
        if not pd.isna(surr_vehicle_id):
            surr_v_x, surr_v_y = get_speed(surr_vehicle_id, row['frame'], clip1_speed_data)
            if surr_v_x is not None and surr_v_y is not None:
                merged_data.at[index, f'{col}_rel_v_x'] = surr_v_x - id_8_v_x
                merged_data.at[index, f'{col}_rel_v_y'] = surr_v_y - id_8_v_y

# 좌표 및 거리 계산 함수
def get_coordinates(vehicle_id, frame, data, position):
    vehicle_data = data[(data['ID'] == vehicle_id) & (data['frame'] == frame)]
    if not vehicle_data.empty:
        if position == 'front':
            return vehicle_data.iloc[0]['front_x'], vehicle_data.iloc[0]['front_y']
        elif position == 'back':
            return vehicle_data.iloc[0]['back_x'], vehicle_data.iloc[0]['back_y']
    return None, None

def calculate_euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# 상대 거리 계산 및 새로운 열 추가
for col in surrounding_columns:
    merged_data[f'{col}_distance'] = 0.0

for index, row in merged_data.iterrows():
    id_8_front_x, id_8_front_y = row['front_x'], row['front_y']
    id_8_back_x, id_8_back_y = row['back_x'], row['back_y']

    for col in surrounding_columns:
        surr_vehicle_id = row[col]
        if not pd.isna(surr_vehicle_id):
            if col in ['preceding', 'right_preceding', 'left_preceding']:
                surr_x, surr_y = get_coordinates(surr_vehicle_id, row['frame'], clip1_speed_data, 'back')
            else:  # 'following', 'right_following', 'left_following'
                surr_x, surr_y = get_coordinates(surr_vehicle_id, row['frame'], clip1_speed_data, 'front')

            if surr_x is not None and surr_y is not None:
                if col in ['preceding', 'right_preceding', 'left_preceding']:
                    distance = calculate_euclidean_distance(id_8_front_x, id_8_front_y, surr_x, surr_y)
                else:  # 'following', 'right_following', 'left_following'
                    distance = calculate_euclidean_distance(id_8_back_x, id_8_back_y, surr_x, surr_y)
                merged_data.at[index, f'{col}_distance'] = distance

# 필요한 컬럼 선택하여 최종 데이터프레임 생성
final_columns = ['frame', 'ID', 'center_x', 'center_y', 'v_x', 'v_y', 'a_x', 'a_y', 'lane', 
                 'preceding_rel_v_x', 'preceding_rel_v_y', 'right_preceding_rel_v_x', 'right_preceding_rel_v_y',
                 'left_preceding_rel_v_x', 'left_preceding_rel_v_y', 'following_rel_v_x', 'following_rel_v_y',
                 'right_following_rel_v_x', 'right_following_rel_v_y', 'left_following_rel_v_x', 'left_following_rel_v_y',
                 'preceding_distance', 'right_preceding_distance', 'left_preceding_distance', 
                 'following_distance', 'right_following_distance', 'left_following_distance']
final_data = merged_data[final_columns]

# CSV 파일로 저장
output_csv_path = 'C:/Users/user/Desktop/drone_vision_local/clip/clip1/result.csv'
final_data.to_csv(output_csv_path, index=False)
