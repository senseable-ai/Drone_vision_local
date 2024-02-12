import pandas as pd
from pathlib import Path

id_results = {}

for clip_number in range(1, 120):
    file_path = f'C:/Users/user/Desktop/drone_vision_local/LC/clip{clip_number}/clip{clip_number}_coordinate.csv'
    
    if Path(file_path).is_file():
        try:
            data = pd.read_csv(file_path)
            
            stable_lane_ids = data.groupby('ID').filter(lambda x: x['lane'].nunique() == 1)
            
            longest_frame_id = stable_lane_ids.groupby('ID').size().idxmax()
            
            id_results[clip_number] = longest_frame_id
        except Exception as e:
            id_results[f'clip{clip_number}'] = f"Error: {e}"
    else:
        continue

# 결과 출력
print(id_results)
