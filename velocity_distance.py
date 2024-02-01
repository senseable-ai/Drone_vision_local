import pandas as pd

# 파일 로드
df_speed = pd.read_csv("/Users/user/Desktop/drone_vision_local/clip/clip1/clip1_speed.csv")
df_surrounding = pd.read_csv("/Users/user/Desktop/drone_vision_local/clip/clip1/surrounding_vehicles_for_id_8.csv")

# 유클리드 거리 계산 함수
def calculate_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# 수정된 상대 거리 계산 함수: 차량의 전방 및 후방 좌표를 사용
def calculate_relative_distance(front_x1, front_y1, back_x1, back_y1, front_x2, front_y2, back_x2, back_y2):
    distance_front = calculate_distance(front_x1, front_y1, front_x2, front_y2)
    distance_back = calculate_distance(back_x1, back_y1, back_x2, back_y2)
    return (distance_front + distance_back) / 2

# 상대 속도 계산 함수: x 및 y 좌표를 사용
def calculate_relative_speed_x(vx1, vx2):
    return abs(vx1 - vx2)

def calculate_relative_speed_y(vy1, vy2):
    return abs(vy1 - vy2)

# 주변 차량 유형
surrounding_types = ['leading', 'right_leading', 'left_leading', 'following', 'right_following', 'left_following']

# 주변 차량 정보와 상대 거리 및 속도 정보를 포함할 새 데이터프레임 생성
df_final = df_surrounding.copy()

# 상대 거리 및 속도를 저장할 새 열 추가
for s_type in surrounding_types:
    df_final[f'{s_type}_relative_distance'] = None
    df_final[f'{s_type}_relative_speed_x'] = None
    df_final[f'{s_type}_relative_speed_y'] = None

# 주변 차량의 상대 거리 및 속도 계산 및 저장
for index, row in df_final.iterrows():
    frame = row['frame']
    # ID 8의 전방 및 후방 좌표, 속도 정보
    id_8_info = df_speed[(df_speed['frame'] == frame) & (df_speed['ID'] == 8)].iloc[0]
    front_x8, front_y8, back_x8, back_y8, vx8, vy8 = id_8_info['front_x'], id_8_info['front_y'], id_8_info['back_x'], id_8_info['back_y'], id_8_info['v_x'], id_8_info['v_y']
    
    for s_type in surrounding_types:
        vid = row[s_type]
        if pd.notna(vid):
            # 주변 차량의 전방 및 후방 좌표, 속도 정보
            surrounding_info = df_speed[(df_speed['frame'] == frame) & (df_speed['ID'] == vid)].iloc[0]
            front_x_vid, front_y_vid, back_x_vid, back_y_vid, vx_vid, vy_vid = surrounding_info['front_x'], surrounding_info['front_y'], surrounding_info['back_x'], surrounding_info['back_y'], surrounding_info['v_x'], surrounding_info['v_y']
            
            # 상대 거리 및 속도 계산
            relative_distance = round(calculate_relative_distance(front_x8, front_y8, back_x8, back_y8, front_x_vid, front_y_vid, back_x_vid, back_y_vid),3)
            relative_speed_x = round(calculate_relative_speed_x(vx8, vx_vid),3)
            relative_speed_y = round(calculate_relative_speed_y(vy8, vy_vid),3)
            
            # 결과 저장
            df_final.at[index, f'{s_type}_relative_distance'] = relative_distance
            df_final.at[index, f'{s_type}_relative_speed_x'] = relative_speed_x
            df_final.at[index, f'{s_type}_relative_speed_y'] = relative_speed_y

# 수정된 결과를 새로운 CSV 파일로 저장
save_path_with_corrected_distances_speeds = "relative_distance_speed_corrected_with_surroundings_id_8.csv"
df_final.to_csv(save_path_with_corrected_distances_speeds, index=False)
