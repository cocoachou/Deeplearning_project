#수정
import os
from PIL import Image, ImageChops, ImageEnhance

def ELA_analysis_and_save(input_path, output_path, quality):
    original = Image.open(input_path)
    original.save('temp.jpg', 'JPEG', quality=quality)
    temporary = Image.open('temp.jpg')

    ela_image = ImageChops.difference(original, temporary)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    ela_image.save(output_path)

def process_and_save_ela_images(input_dir, output_dir, quality):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category in ['real', 'fake']:
        sub_input_dir = os.path.join(input_dir, category)
        sub_output_dir = os.path.join(output_dir, category)
        if not os.path.exists(sub_output_dir):
            os.makedirs(sub_output_dir)
        for filename in os.listdir(sub_input_dir):
            if filename.endswith('.jpg'):
                input_path = os.path.join(sub_input_dir, filename)
                output_path = os.path.join(sub_output_dir, filename)
                if not os.path.exists(output_path):  # 파일이 존재하지 않는 경우에만 처리
                    ELA_analysis_and_save(input_path, output_path, quality)

# 디렉토리 설정 및 함수 호출
input_train_dir = '/content/gdrive/MyDrive/DL/deepfake_1st/train_crop'
output_train_dir = '/content/gdrive/MyDrive/DL/deepfake_1st/train_ela/new_ela_30'

#test 데이터는 나중에
#input_test_dir = '/content/gdrive/MyDrive/DL/team_project/crop/ela_test_crop/test'
#output_test_dir = '/content/gdrive/MyDrive/DL/team_project/crop/ela_test_crop/test_ela_30'

quality = 30

process_and_save_ela_images(input_train_dir, output_train_dir, quality)
process_and_save_ela_images(input_test_dir, output_test_dir, quality)
