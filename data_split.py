
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_preprocessed_data(file_path='C:/Users/user/Desktop/drone_vision_local/combined_data.csv'):
    # 데이터 로드
    data = pd.read_csv(file_path)

    # 결측치 처리 (예: 0으로 채우기, 평균으로 채우기, 행 삭제 등)
    data.fillna(0, inplace=True)

    # 데이터 정규화
    scaler = StandardScaler()
    features_to_scale = ['center_x', 'center_y', 'v_x', 'v_y', 'a_x', 'a_y'] # 정규화할 특성 목록
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

    # 고정 길이 시퀀스 생성 (여기서는 각 시퀀스 길이를 10으로 가정)
    sequence_length = 10
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        seq = data.iloc[i:i + sequence_length].values
        sequences.append(seq)

    sequences = np.array(sequences)

    # 학습, 검증, 테스트 세트 분할
    targets = data['lane'][sequence_length-1:].values # 예제로 'lane' 열을 타겟으로 사용
    X_train, X_temp, y_train, y_temp = train_test_split(sequences, targets, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 결과 확인
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test
