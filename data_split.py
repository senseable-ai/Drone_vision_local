from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 데이터셋을 특성(X)과 타겟(y)으로 분할 (여기서는 예시로 'center_x'를 타겟으로 가정)
X = data_preprocessed_df.drop(['ID'], axis=1)  # '[]'를 제외한 모든 특성을 사용
y = data_preprocessed_df['']  # '[]'를 타겟으로 사용

# 학습, 검증, 테스트 세트로 분할
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 시퀀스 패딩 예시 (여기서는 패딩을 적용할 실제 시퀀스 데이터가 필요함)
# 예시 시퀀스 데이터 (가정)
sequences = [[1, 2, 3], [4, 5], [6]]

# 시퀀스 패딩 (최대 길이를 5로 가정)
padded_sequences = pad_sequences(sequences, maxlen=5, padding='post', truncating='post')

print("Padded Sequences:")
print(padded_sequences)

# 참고: 실제 데이터에 맞게 'sequences'를 해당 데이터의 시퀀스 형태로 준비해야 합니다.
# 시퀀스 데이터는 일반적으로 시간에 따른 데이터 포인트의 연속적인 배열입니다.
