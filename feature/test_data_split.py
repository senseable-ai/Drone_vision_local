import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_preprocessed_data(file_path='C:/Users/user/Desktop/drone_vision_local/combined_data.csv'):
    # 데이터 로드
    data = pd.read_csv(file_path)

    # 결측치 처리 (예: 0으로 채우기, 평균으로 채우기, 행 삭제 등)
    data.fillna(0, inplace=True)

    # 데이터 정규화 (타겟 열 'LC' 제외하고 전체 열 정규화)
    scaler = StandardScaler()
    features_to_scale = data.columns.drop('LC')  # 타겟 열 'LC'를 제외한 전체 열을 정규화 대상으로 설정
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

    # 데이터와 타겟 분리
    X = data.drop(columns=['LC','center_x', 'center_y']).values  # 'LC' 열을 제외한 데이터
    y = data['LC'].values  # 타겟 데이터

    # 학습, 테스트 세트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 결과 확인
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_preprocessed_data()
