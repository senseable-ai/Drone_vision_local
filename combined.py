import pandas as pd
import os
from tqdm import tqdm

# 원본 디렉토리 설정
source_directories = ["C:\\Users\\user\\Desktop\\drone_vision_local\\LC", "C:\\Users\\user\\Desktop\\drone_vision_local\\LK"]

# 데이터를 저장할 빈 DataFrame 생성
combined_data = pd.DataFrame()

# 파일 병합
for directory in source_directories:
    for i in tqdm(range(2, 119), desc=f"Processing files in {directory}"):
        result_file_path = os.path.join(directory, f"clip{i}", f"clip{i}_result.csv")
        sv_file_path = os.path.join(directory, f"clip{i}", f"clip{i}_sv.csv")
        
        # 결과 파일과 SV 파일이 모두 존재하는지 확인
        if os.path.exists(result_file_path) and os.path.exists(sv_file_path):
            try:
                # CSV 파일 읽기
                result_data = pd.read_csv(result_file_path)
                sv_data = pd.read_csv(sv_file_path)
                
                # 데이터가 비어있지 않은 경우에만 병합
                if not result_data.empty and not sv_data.empty:
                    # frame을 기준으로 데이터 병합
                    merged_data = pd.merge(result_data, sv_data, on="frame")
                    combined_data = pd.concat([combined_data, merged_data], ignore_index=True)
            except pd.errors.EmptyDataError:
                print(f"Warning: Empty data found in files: {result_file_path}, {sv_file_path}")

# 병합된 데이터를 CSV 파일로 저장
combined_data.to_csv("C:\\Users\\user\\Desktop\\drone_vision_local\\combined_data.csv", index=False)
