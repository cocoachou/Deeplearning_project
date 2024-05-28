import os
from pathlib import Path
import concurrent.futures

# 폴더 경로 설정 ; 상위 경로는 개인마다 다를 수 있음
fake_folder_base_path = Path("/content/gdrive/MyDrive/DL/team_project/datasets/deepfake_1st/fake")
real_folder_base_path = Path("/content/gdrive/MyDrive/DL/team_project/datasets/deepfake_1st/real")

# 모든 파일 경로 가져오기
def get_first_file_path(folder_path):
    files = sorted(folder_path.glob("**/*"))
    if files:
        return files[0]
    return None

# fake paths 병렬로 수집
with concurrent.futures.ThreadPoolExecutor() as executor:
    fake_folder_paths = sorted(fake_folder_base_path.glob("*/*/*/*/*"))
    fake_paths = list(executor.map(get_first_file_path, fake_folder_paths))

# None 값 제거
fake_paths = [str(path) for path in fake_paths if path]

# real paths 병렬로 수집
with concurrent.futures.ThreadPoolExecutor() as executor:
    real_folder_paths = sorted(real_folder_base_path.glob("*/*/*/*"))
    real_paths = list(executor.map(get_first_file_path, real_folder_paths))

# None 값 제거
real_paths = [str(path) for path in real_paths if path]

# 경로 수
print(f"len(fake_paths): {len(fake_paths)}")
print(f"len(real_paths): {len(real_paths)}")

# txt 파일로 저장
with open('fake_paths.txt', 'w') as f:
    f.write('\n'.join(fake_paths))

with open('real_paths.txt', 'w') as f:
    f.write('\n'.join(real_paths))
