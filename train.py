import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras import layers, models, Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# 파일 경로 설정
base_dir = '/content/gdrive/MyDrive/DL/team_project/datasets/deepfake_1st/train_ela/new_ela_50'
real_dir = os.path.join(base_dir, 'real')
fake_dir = os.path.join(base_dir, 'fake')
csv_path = '/content/gdrive/MyDrive/DL/team_project/datasets/deepfake_1st/selected_data_scale50_5000.csv'  # 파일 이름 변경

checkpoint_dir = '/content/gdrive/MyDrive/DL/team_project/datasets/deepfake_1st/ted/ela50_5000_method3'  # 임의로 설정
log_path = os.path.join(checkpoint_dir, 'training_log.csv') # 로그 파일 경로 설정

# 경로와 파일 존재 여부 확인
print(f"Real directory exists: {os.path.exists(real_dir)}")
print(f"Fake directory exists: {os.path.exists(fake_dir)}")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    print("CSV 파일이 존재하지 않습니다.")
    exit()

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='label',
    target_size=(380, 380),
    batch_size=32,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='filename',
    y_col='label',
    target_size=(380, 380),
    batch_size=32,
    class_mode='binary'
)

# EfficientNetB4 모델 불러오기
input_shape = (380, 380, 3)
efficientnet_base = EfficientNetB4(weights='imagenet', include_top=False, input_shape=input_shape)

# 특정 레이어 이후의 모든 레이어를 trainable로 설정
set_trainable = False
for layer in efficientnet_base.layers:
    if layer.name == 'block7a_expand_conv':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# 모델 구성
finetune_model = models.Sequential()
finetune_model.add(efficientnet_base)
finetune_model.add(layers.GlobalAveragePooling2D())
finetune_model.add(layers.Dense(256, activation='relu'))
finetune_model.add(BatchNormalization())
finetune_model.add(Dropout(0.5))
finetune_model.add(layers.Dense(1, activation='sigmoid'))

# 모델 컴파일
finetune_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 콜백 설정
checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, 'best_model.h5'), monitor='val_loss', save_best_only=True, mode='min')
csv_logger = CSVLogger(log_path, append=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        print(f"Learning rate for epoch {epoch + 1} is {lr.numpy()}")

lr_logger = LearningRateLogger()

callbacks = [checkpoint, csv_logger, reduce_lr, lr_logger]

# 모델 학습
history = finetune_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=callbacks
)
