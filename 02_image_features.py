# 02. 사진에서 매력도 점수 뽑기 (YOLO + CLIP)
# 입력: output/train_preprocessed.csv, data/train_images/ (사진)
# 출력: output/image_features.csv (동물별), output/image_features_photo_level.csv (사진별)
# 실행: 구글 코랩 GPU 활용 (run_02_colab.ipynb)
#
# 변수 (사진마다 0~1 → 같은 동물 사진끼리 평균)
#   eyes            카메라를 쳐다보는지          (동물 부분만 잘라서 CLIP)
#   face            얼굴이 잘 보이는지           (동물 부분만 잘라서 CLIP)
#   action          움직이는 모습인지            (동물 부분만 잘라서 CLIP)
#   accessory       리본·옷·장난감이 있는지      (동물 부분만 잘라서 CLIP)
#   subject_focus   동물이 사진의 주인공인지     (사진 전체로 CLIP)
#   occlusion       가리는 것 없이 잘 보이는지   (사진 전체로 CLIP)
#   near            동물이 사진에서 차지하는 크기 (YOLO 박스)
#   has_human_photo 사람과 같이 찍힌 사진이 있는지 (YOLO)
#   blur            사진 선명도 (OpenCV)

import os
import glob
import cv2
import pandas as pd
import torch
import open_clip
from PIL import Image
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 1. 모델 불러오기
#    YOLO: 사진 속 개·고양이·사람 위치 찾기 / CLIP: 사진과 문장이 얼마나 비슷한지
yolo = YOLO('yolo11s.pt')
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model = clip_model.to(device).eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')


# 2. CLIP 채점 문장 설정 (좋은 쪽 문장들, 나쁜 쪽 문장들, 동물만 잘라서 볼지)
criteria = {
    'eyes': (
        ["A photo of a pet looking directly at the camera.",
         "A pet making eye contact with the camera.",
         "A close-up of a pet's face staring straight at the camera."],
        ["A photo of a pet looking away from the camera.",
         "A pet looking down or up at something, not at the camera.",
         "A side profile photo of a pet looking elsewhere."],
        True),
    'face': (
        ["A clear photo of the pet's face.",
         "A photo showing the pet's full face clearly."],
        ["A photo where the pet's face is not visible.",
         "A photo showing only the pet's body, face turned away or hidden."],
        True),
    'action': (
        ["A photo of a pet in motion, running or jumping.",
         "An active pet caught mid-action, playful and lively."],
        ["A photo of a pet sitting or lying still.",
         "A stationary pet not moving, completely passive."],
        True),
    'accessory': (                                  # 목줄은 너무 흔해서 뺌
        ["A pet wearing a bow or costume.",
         "A pet with a toy visible."],
        ["A pet with no toy, bow, or costume.",
         "A plain photo of a pet with nothing else visible."],
        True),
    'subject_focus': (
        ["A photo where the pet is the clear main subject.",
         "A photo focused entirely on the pet with minimal distractions."],
        ["A cluttered photo where the pet is hard to notice.",
         "A photo where the pet is a small part of a busy scene."],
        False),
    'occlusion': (
        ["A pet clearly visible with nothing blocking the view.",
         "An unobstructed clear view of the entire pet."],
        ["A pet partially blocked by cage bars or an object.",
         "A pet obscured by something in front of it."],
        False),
}


# 문장 여러 개 → 벡터 1개 (문장마다 길이 1로 맞춘 뒤 평균)
def text_vector(sentences):
    with torch.no_grad():
        v = clip_model.encode_text(tokenizer(sentences).to(device))
    v = v / v.norm(dim=-1, keepdim=True)
    return v.mean(dim=0, keepdim=True)


good_vec = {}
bad_vec = {}
for name in criteria:
    good_vec[name] = text_vector(criteria[name][0])
    bad_vec[name] = text_vector(criteria[name][1])


# 사진 → 벡터 (길이 1로 맞춤)
def image_vector(img):
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        v = clip_model.encode_image(x)
    return v / v.norm(dim=-1, keepdim=True)


# 점수: 좋은 쪽 문장에 더 가까울수록 1에 가깝게 (유사도 × 100 → 소프트맥스)
def clip_score(img_vec, name):
    with torch.no_grad():
        sim_good = (img_vec @ good_vec[name].T).squeeze(-1)
        sim_bad = (img_vec @ bad_vec[name].T).squeeze(-1)
        prob = torch.softmax(torch.stack([sim_good, sim_bad], dim=1) * 100, dim=1)[:, 0]
    return round(prob.item(), 4)


# 3. 대상 사진 목록 (사진 파일 이름 = "PetID-번호.jpg")
pets = pd.read_csv('output/train_preprocessed.csv', dtype={'PetID': str})
target = set(pets[pets['PhotoAmt'] > 0]['PetID'])

# 01번에서 번호를 바꾼 PetID는 사진 파일 이름이 원래 PetID라서 연결표 사용
pid = pd.read_csv('data/manual/petid_team_mapping.csv', dtype=str)
orig_to_team = dict(zip(pid['PetID_원본'], pid['PetID_팀']))

# 옆으로 누운 사진을 바로 세운 사진 폴더 사용 (data/train_images/ 안에 압축 풀기)
photo_list = []
seen = set()
for path in sorted(glob.glob('data/train_images/**/*.jpg', recursive=True)):
    name = os.path.basename(path)[:-4]                       # '045543730-1.jpg' → '045543730-1'
    if '-' not in name:
        continue
    pet_id, photo_no = name.rsplit('-', 1)                   # → '045543730', '1'
    pet_id = orig_to_team.get(pet_id, pet_id)
    if pet_id in target and (pet_id, photo_no) not in seen:  # 같은 사진은 한 번만
        seen.add((pet_id, photo_no))
        photo_list.append((pet_id, path))
print('사진 수:', len(photo_list))


# 4. 사진 채점 (YOLO는 32장씩 묶어서 실행)
BATCH_SIZE = 32
rows = []
for start in range(0, len(photo_list), BATCH_SIZE):
    batch = photo_list[start:start + BATCH_SIZE]
    batch_paths = [path for _, path in batch]

    # 4-1. YOLO: 사진마다 가장 큰 개·고양이 박스 + 사람 있는지
    results = yolo.predict(batch_paths, imgsz=640, conf=0.25, verbose=False)

    for (pet_id, path), result in zip(batch, results):
        img = Image.open(path).convert('RGB')
        W, H = img.size

        best_box = None
        best_area = 0
        human = 0
        for box, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
            label = result.names[int(cls)]
            if label in ['dog', 'cat']:
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > best_area:
                    best_box = box
                    best_area = area
            if label == 'person':
                human = 1

        # 4-2. 동물 부분만 자르기 (주변 15% 여유, 못 찾으면 원본 그대로)
        if best_box is None:
            crop = img
        else:
            x1, y1, x2, y2 = best_box
            pad_x = (x2 - x1) * 0.15
            pad_y = (y2 - y1) * 0.15
            crop = img.crop((max(0, x1 - pad_x), max(0, y1 - pad_y),
                             min(W, x2 + pad_x), min(H, y2 + pad_y)))

        # 4-3. 선명도: 라플라시안 분산 (경계선이 뚜렷할수록 큼)
        bgr = cv2.imread(path)
        if bgr is None:
            sharpness = None
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            sharpness = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)

        # 4-4. 점수 기록
        row = {'PetID': pet_id, 'photo_path': path,
               'has_human_photo': human,
               'near': round(best_area / (W * H), 4),      # 동물 박스 크기 / 사진 크기
               'blur_raw': sharpness}
        crop_vec = image_vector(crop)
        full_vec = image_vector(img)
        for name in criteria:
            use_crop = criteria[name][2]
            if use_crop:
                row[name] = clip_score(crop_vec, name)
            else:
                row[name] = clip_score(full_vec, name)
        rows.append(row)

    if start % 3200 == 0:
        print(start, '/', len(photo_list), flush=True)

photo_df = pd.DataFrame(rows)

# 선명도를 0~1로 (전체 사진 최솟값 = 0, 최댓값 = 1)
photo_df['blur'] = ((photo_df['blur_raw'] - photo_df['blur_raw'].min())
                    / (photo_df['blur_raw'].max() - photo_df['blur_raw'].min()))
photo_df.to_csv('output/image_features_photo_level.csv', index=False)


# 5. 같은 동물 사진끼리 합치기 (점수는 평균, 사람 사진은 한 장이라도 있으면 1)
score_cols = ['eyes', 'face', 'action', 'accessory', 'subject_focus', 'occlusion', 'near', 'blur']
pet_df = photo_df.groupby('PetID')[score_cols].mean()
pet_df['has_human_photo'] = photo_df.groupby('PetID')['has_human_photo'].max()
pet_df['n_photos_used'] = photo_df.groupby('PetID').size()
pet_df = pet_df.reset_index()

pet_df.to_csv('output/image_features.csv', index=False)
print('저장 완료:', pet_df.shape)
