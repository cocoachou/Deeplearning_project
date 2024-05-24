# data_preprocessing
FaceNet에 기반한 MTCNN 패키지를 통해 face detection을 두가지 방법을 이용해 전처리 진행

crop_face_1-a.py

안면비율이 유지되도록 특정 Width, Height 마진 입력 후, resize를 진행


crop_face_1-b.py

MTCNN으로부터 얻은 Bounding Box의 중심점을 찾고, Bounding Box의 Width, Height 중 최댓값을 기준으로 새 Bounding Box를 그림.
이후, 빈 공간에 대해 zero padding을 실시

crop_face.py

기존 raw 데이터가 각 동영상별 1초에 2장씩 샘플링된 이미지라 중복으로 여길 수 있는 이미지가 많음.
각 동영상 별로 한 장의 사진만 샘플링한 방법으로 데이터 크기를 줄이고, 1-b 방법을 통해 전처리


new_crop_face.py

raw data 파일이 디렉토리는 서로 다르지만 파일명이 00001, 00002 등으로 중복되어
output 과정에서 덮어쓰기 되는 문제가 생겨, 파일 디렉토리를 해싱하여 새로운 파일명으로 저장되도록 함.
