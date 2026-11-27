# 05. 정형 + 사진 + 글 변수를 하나의 표로 합치기
# 입력: 01·02번 결과 + 친밀도·글 점수 파일
# 출력: output/train_joined_final.csv
#
# 결측 처리
#   - 사진이 없는 공고     → 사진 변수 빈칸 (HasPhoto=0으로 구분)
#   - 입양 완료 글         → 글 변수 7개 모두 빈칸
#   - 설명글이 아예 없는 글 → 단어 수 0, 나머지 글 점수는 빈칸

import re
import jieba
import numpy as np
import pandas as pd

df = pd.read_csv('output/train_preprocessed.csv', dtype={'PetID': str})      # 01번
image = pd.read_csv('output/image_features.csv', dtype={'PetID': str})       # 02번

# AI 점수(03·04번)는 돌릴 때마다 조금씩 달라서 채점해 둔 결과 파일 사용
close = pd.read_csv('data/manual/closeness_team.csv', dtype={'PetID': str})      # 03번 결과
scores = pd.read_csv('data/manual/text_scores_team.csv', dtype={'PetID': str})   # 04번 결과


# 단어 수: 팀이 채점해 둔 점수 파일에는 없어서 여기서 직접 계산 (04번과 같은 방법)
def word_count(text):
    if pd.isna(text) or str(text).strip() == '':
        return 0
    text = str(text)
    chinese = 0
    for part in re.findall(r'[\u4e00-\u9fff]+', text):
        chinese += len([w for w in jieba.lcut(part) if w.strip()])
    text = re.sub(r'[\u4e00-\u9fff]+', ' ', text)
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)) + chinese


text = df[['PetID']].copy()
text['WordCount'] = df['Description'].apply(word_count)
text = text.merge(scores, on='PetID', how='left')


# 1. PetID 기준으로 합치기
df = df.merge(image, on='PetID', how='left')
df = df.merge(close, on='PetID', how='left')
df = df.merge(text, on='PetID', how='left')


# 2. 입양 완료 글은 소개글이 아니라서 글 변수를 빈칸으로
text_cols = ['WordCount', 'Urgency', 'appearance_praise', 'friendliness',
             'direct_appeal', 'contact', 'training_mention']
df.loc[df['Adopted'] == 1, text_cols] = np.nan


# 3. 저장
cols = ['PetID', 'RescuerID', 'RescuerID_PostCount', 'Type', 'Quantity', 'Name', 'HasName',
        'Age', 'Gender', 'Breed1', 'Breed1Name', 'Breed2', 'Breed2Name', 'IsMixed',
        'Color1', 'Color1Name', 'Color2', 'Color2Name', 'Color3', 'Color3Name',
        'MaturitySize', 'FurLength', 'Health', 'Vaccinated', 'Dewormed', 'Sterilized',
        'MedicalCare_score', 'Fee', 'HasFee', 'State', 'StateName', 'VideoAmt',
        'PhotoAmt', 'HasPhoto', 'n_photos_used', 'has_human_photo',
        'eyes', 'face', 'action', 'accessory', 'subject_focus', 'occlusion', 'near',
        'blur', 'Closeness', 'Description', 'Description_en_clean', 'Adopted',
        'WordCount', 'Urgency', 'appearance_praise', 'friendliness',
        'direct_appeal', 'contact', 'training_mention', 'AdoptionSpeed']
df = df[cols]
df.to_csv('output/train_joined_final.csv', index=False, encoding='utf-8-sig')
print('저장 완료:', df.shape)
