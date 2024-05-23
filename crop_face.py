import os
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
import matplotlib.pyplot as plt
import hashlib

def generate_unique_filename(image_path, index):
    # 이미지 경로와 인덱스를 기반으로 해시를 생성하여 고유한 파일 이름 생성
    unique_string = f"{image_path}_{index}"
    unique_hash = hashlib.md5(unique_string.encode()).hexdigest()
    return unique_hash

def crop_faces_from_txt(input_txt_path, output_directory, target_size=(380, 380), min_size=20, scale_factor=0.709):
    # MTCNN 초기화 (min_size와 scale_factor만 사용)
    detector = MTCNN(min_face_size=min_size, scale_factor=scale_factor)

    # 결과를 저장할 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # txt 파일에서 경로 읽기
    with open(input_txt_path, 'r') as file:
        image_paths = file.readlines()

    # 각 이미지 경로 처리
    for image_path in image_paths:
        image_path = image_path.strip()  # 줄 끝의 개행 문자 제거
        print(f"Processing file: {image_path}")

        # 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR 형식을 사용하므로 RGB로 변환

        # 얼굴 검출
        results = detector.detect_faces(image_rgb)

        # 얼굴 부분 crop & 크기 조정
        for i, result in enumerate(results):
            x, y, w, h = result['box']
            x, y = max(0, x), max(0, y)  # 좌표가 음수가 되는 것 방지

            # 얼굴 중심점 계산
            center_x = x + w // 2
            center_y = y + h // 2

            # 얼굴 width와 height 중 최댓값 계산
            face_size = max(w, h)

            # 새로운 Bounding Box의 좌상단 점 계산
            new_x = max(0, center_x - face_size // 2)
            new_y = max(0, center_y - face_size // 2)

            # 새로운 Bounding Box의 우하단 점 계산
            new_w = min(image.shape[1], center_x + face_size // 2)
            new_h = min(image.shape[0], center_y + face_size // 2)

            # 얼굴 부분 crop
            face = image_rgb[new_y:new_h, new_x:new_w]

            # 얼굴을 target_size로 resize
            face_resized = cv2.resize(face, target_size)

            # zero padding을 실시하여 target_size에 맞게 조정
            padded_face = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            padded_face[:face_resized.shape[0], :face_resized.shape[1], :] = face_resized

            # 고유한 파일 이름 생성
            unique_filename = generate_unique_filename(image_path, i + 1)
            output_path = os.path.join(output_directory, f"{unique_filename}.jpg")
            
            # 조정된 얼굴을 저장
            cv2.imwrite(output_path, cv2.cvtColor(padded_face, cv2.COLOR_RGB2BGR))

            # crop된 얼굴을 출력(옵션)
            plt.imshow(padded_face)
            plt.axis('off')
            plt.show()

# 경로 설정
input_txt_path = '/content/gdrive/MyDrive/DL/team_project/datasets/deepfake_1st/real_paths.txt'  # txt 파일 경로
output_directory = '/content/crop'   # 디렉토리는 개별적으로 변경!!!!

# 함수 사용 예시
crop_faces_from_txt(input_txt_path, output_directory)
