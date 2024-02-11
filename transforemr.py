import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MultiHeadAttention, LayerNormalization, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

# 모델 파라미터 정의
input_shape = (X_train.shape[1],)  # 입력 데이터의 형태
num_heads = 4  # 멀티-헤드 어텐션의 헤드 수
model_dim = 128  # 모델의 차원
dropout_rate = 0.1  # 드롭아웃 비율

# 모델 구성
model = Sequential([
    # 멀티-헤드 셀프 어텐션 레이어
    MultiHeadAttention(num_heads=num_heads, key_dim=model_dim, input_shape=input_shape),
    Dropout(dropout_rate),
    LayerNormalization(),  # 층 정규화
    # 완전 연결 레이어 (Dense Layers)
    Flatten(),  # 멀티-헤드 어텐션 출력을 평탄화
    Dense(64, activation='relu'),
    Dense(1)  # 예측을 위한 출력 레이어
])

# 모델 컴파일
model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

# 모델 요약 출력
model.summary()
