import pandas as pd
import numpy as np

vehicle_id_mapping = {1:38, 2:42, 3:12, 4:10, 5:20, 6:34, 10:7, 11:2, 12:11, 13:26, 14:13, 15:39, 16:28, 17:2, 18:4, 19:8, 20:30,
                      21:5, 22:31, 23:17, 24:26, 25:11, 26:15, 28:23, 29:1, 30:8, 31:24, 32:2, 33:21, 35:13, 36:18, 37:15, 38:29,
                      39:13, 40:9, 41:10, 42:8, 43:16, 44:13, 45:8, 46:27, 47:45, 48:9, 49:25, 50:6, 51:4, 52:5, 53:14, 54:31, 55:3,
                      56:52, 59:17, 60:8, 61:17, 62:41, 63:5, 64:64, 66:35, 67:5, 68:10, 69:9, 70:9, 71:7, 72:24, 73:13, 74:16, 75:27,
                      76:42, 77:6, 78:51, 79:8, 80:2, 81:17, 82:7, 83:45, 85:12, 86:26, 87:49, 88:59, 89:24, 90:93, 91:54, 92:21, 93:56,
                      94:15, 95:15, 96:48, 97:58, 98:29, 99:35, 100:7, 101:13, 102:19, 103:15, 104:2, 106:1, 107:6, 108:8, 110:5, 111:2,
                      112:25, 118:4}  

# 상대 속도 계산 함수
def get_relative_speed(id_v_x, id_v_y, surr_v_x, surr_v_y):
    return surr_v_x - id_v_x, surr_v_y - id_v_y

# 거리 계산 함수
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# i 값에 대한 루프 (1부터 119까지)
for i in range(1, 120):
    vehicle_id = vehicle_id_mapping.get(i)  # i에 대응하는 vehicle_id
    if vehicle_id is None:
        print(f"No vehicle ID mapping for clip {i}. Skipping...")
        continue
    
    try:
        # 데이터 로드 경로 설정
        clip_speed_path = f'C:/Users/user/Desktop/drone_vision_local/LC/clip{i}/clip{i}_speed.csv'
        surrounding_vehicles_path = f'C:/Users/user/Desktop/drone_vision_local/LC/clip{i}/clip{i}_sv.csv'
        
        # 데이터 로드 시도
        clip_speed_data = pd.read_csv(clip_speed_path)
        surrounding_vehicles_data = pd.read_csv(surrounding_vehicles_path)
        
        # ID에 해당하는 차량 데이터 필터링
        id_data = clip_speed_data[clip_speed_data['ID'] == vehicle_id]

        # 데이터 병합
        merged_data = pd.merge(surrounding_vehicles_data, id_data, on='frame', how='left')

        # 컬럼 이름 변경
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

        # 상대 속도 및 거리 계산
        for index, row in merged_data.iterrows():
            id_v_x, id_v_y = row['v_x'], row['v_y']

            for col in ['preceding', 'right_preceding', 'left_preceding', 'following', 'right_following', 'left_following']:
                surr_id = row[col]
                if pd.notnull(surr_id):
                    surr_data = clip_speed_data[clip_speed_data['ID'] == surr_id].iloc[0]
                    surr_v_x, surr_v_y = surr_data['v_x'], surr_data['v_y']
                    rel_v_x, rel_v_y = get_relative_speed(id_v_x, id_v_y, surr_v_x, surr_v_y)
                    merged_data.at[index, f'{col}_rel_v_x'] = rel_v_x
                    merged_data.at[index, f'{col}_rel_v_y'] = rel_v_y
                    
                    # 거리 계산을 위해 필요한 좌표 정보 추가 가정
                    id_x, id_y, surr_x, surr_y = row['center_x'], row['center_y'], surr_data['center_x'], surr_data['center_y']
                    distance = calculate_distance(id_x, id_y, surr_x, surr_y)
                    merged_data.at[index, f'{col}_distance'] = distance

        # 최종 데이터프레임 선택
        final_columns = [ 
            'frame', 'ID', 'v_x', 'v_y', 'a_x', 'a_y', 'lane', 
            'preceding_rel_v_x', 'preceding_rel_v_y', 'right_preceding_rel_v_x', 'right_preceding_rel_v_y',
            'left_preceding_rel_v_x', 'left_preceding_rel_v_y', 'following_rel_v_x', 'following_rel_v_y',
            'right_following_rel_v_x', 'right_following_rel_v_y', 'left_following_rel_v_x', 'left_following_rel_v_y',
            'preceding_distance', 'right_preceding_distance', 'left_preceding_distance', 
            'following_distance', 'right_following_distance', 'left_following_distance'
        ]
        final_data = merged_data[final_columns]

        # CSV 파일로 저장
        output_csv_path = f'C:/Users/user/Desktop/drone_vision_local/LC/clip{i}/clip{i}_ID{vehicle_id}_result.csv'
        final_data.to_csv(output_csv_path, index=False)
        
    except FileNotFoundError:
        print(f"File not found for clip {i}. Skipping...")
