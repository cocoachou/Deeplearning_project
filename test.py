#test + csv 파일 생성
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#주석 처리된 부분에 사용한 모델 구조 그대로 넣기
'''
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
'''
# 모델 생성
input_shape = (380, 380, 3)
model = create_cnn_model(input_shape)

# 모델 저장 경로 설정(임의 선택)
model_save_path = '/content/gdrive/MyDrive/DL/deepfake_1st/gyeongjin_scale70/model_checkpoints/model_epoch_13.h5'

# 모델 저장 (훈련 후)
model.save(model_save_path)

# 모델 로드
model = tf.keras.models.load_model(model_save_path)

# 테스트 데이터 디렉토리 경로 설정 (임의 선택)
test_dir = '/content/gdrive/MyDrive/DL/deepfake_1st/test_data/test_ela/test_ela_50'

# 결과 저장을 위한 리스트 초기화
results = []

# 테스트 이미지 파일 리스트 가져오기
test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Total test files found: {len(test_files)}")  # 전체 파일 수 확인

# 각 이미지에 대해 예측 수행
for idx, file in enumerate(test_files):
    try:
        # 이미지 경로 설정
        img_path = os.path.join(test_dir, file)
        
        # 이미지 로드 및 전처리
        img = load_img(img_path, target_size=(380, 380))  # 모델에 맞는 입력 크기로 조정
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # 배치를 추가
        
        # 예측 수행
        prediction = model.predict(img_array)
        
        # 결과 저장 (0: real, 1: fake)
        label = 1 if prediction[0] > 0.5 else 0
        results.append([f"leaderboard/{file}", label])
        
        # 현재 처리된 파일 수 출력
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} / {len(test_files)} files")

    except Exception as e:
        print(f"Error processing file {file}: {e}")

# 처리된 파일 수 확인
print(f"Total processed files: {len(results)}")

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(results, columns=['path', 'y'])

# 파일명을 기준으로 오름차순 정렬
results_df = results_df.sort_values(by='path')

# 결과를 CSV 파일로 저장
results_csv_path = '/content/gdrive/MyDrive/DL/deepfake_1st/final_model_70_test_50_results.csv'
results_df.to_csv(results_csv_path, index=False)

print(f"CSV 파일로 결과 저장 완료! 총 {len(results)} 개의 파일이 처리되었습니다.")
