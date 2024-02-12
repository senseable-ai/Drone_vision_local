import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_preprocessed_and_grouped_data(file_path='/home/user/drone-vision/drone_vision_local/combined_data.csv'):
    # 데이터 로드
    data = pd.read_csv(file_path)

    # 결측치 처리 (예: 0으로 채우기, 평균으로 채우기, 행 삭제 등)
    data.fillna(0, inplace=True)

    # 데이터 정규화
    scaler = StandardScaler()
    data.drop = ['center_x', 'center_y']
    features_to_scale = [ 'v_x', 'v_y', 'a_x', 'a_y'] # 정규화할 특성 목록
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

    # 시퀀스 아이디별로 그룹화하여 각 그룹을 하나의 시퀀스로 처리
    grouped = data.groupby(data['ID'])
    sequences = []
    targets = []
    for _, group in grouped:
        sequences.append(group[features_to_scale + ['LC']].values) # 'LC' 타겟도 포함하여 시퀀스 생성
        targets.append(group['LC'].iloc[-1]) # 시퀀스의 마지막 LC 값을 타겟으로 사용

    sequences = np.array(sequences, dtype=object)
    targets = np.array(targets)

    # 학습, 테스트 세트 분할 (여기서는 검증 세트를 생성하지 않음)
    X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.2, random_state=42) # test_size를 조절하여 테스트 세트 비율 조정
    
    # 결과 확인
    print(f"X_train shape: {len(X_train)} sequences")
    print(f"X_test shape: {len(X_test)} sequences")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_preprocessed_and_grouped_data()
    