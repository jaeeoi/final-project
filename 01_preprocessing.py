# 01. 정형 데이터 전처리
# 입력: data/train.csv + 라벨 파일 3개 + data/manual/ 수작업 파일 4개
# 출력: output/train_preprocessed.csv

import os
import re
import pandas as pd

os.makedirs('output', exist_ok=True)


# 1. 데이터 불러오기
train = pd.read_csv('data/train.csv')
breeds = pd.read_csv('data/PetFinder-BreedLabels.csv')
colors = pd.read_csv('data/PetFinder-ColorLabels.csv')
states = pd.read_csv('data/PetFinder-StateLabels.csv')
print('원본:', train.shape)


# 2. 한 마리만 올라온 공고만 사용
#    여러 마리 공고는 사진·털색 등을 한 마리 기준으로 볼 수 없어서 제외
df = train[train['Quantity'] == 1].copy()
df = df.reset_index(drop=True)
print('Quantity=1 :', df.shape)


# 3. 품종 정리
# 이름 유무 판단에 쓸 품종 이름 목록 (품종 정리 전에 만들어 둠)
breed_dict = dict(zip(breeds['BreedID'], breeds['BreedName']))
breed_list = []
for code in list(df['Breed1']) + list(df['Breed2']):
    if code in breed_dict:
        breed_list.append(breed_dict[code].lower().strip())
breed_list = set(breed_list)

# 3-1. 주 품종이 0이고 부 품종에만 값이 있으면 → 주 품종으로 옮기기
cond = df['Breed1'] == 0
df.loc[cond, 'Breed1'] = df.loc[cond, 'Breed2']
df.loc[cond, 'Breed2'] = 0

# 3-2. 고양이인데 개 품종이 들어간 공고 → 사진 확인 후 수정
#      1건은 실제로 개 → 종을 개, 품종을 Mixed Breed(307)로
#      나머지는 고양이가 맞음 → Domestic Short Hair(266)로
df.loc[df['PetID'] == '6c399cb06', 'Type'] = 1
df.loc[df['PetID'] == '6c399cb06', 'Breed1'] = 307

dog_breeds = breeds[breeds['Type'] == 1]['BreedID']
cond = (df['Type'] == 2) & (df['Breed1'].isin(dog_breeds))
print('고양이인데 개 품종:', cond.sum())
df.loc[cond, 'Breed1'] = 266

# 3-3. 부 품종이 의미 없으면 0으로 (주·부 품종이 같거나, 주 품종이 이미 잡종일 때)
cond = (df['Breed1'] == df['Breed2']) | (df['Breed1'].isin([307, 264, 265, 266]))
print('부 품종 지움:', (cond & (df['Breed2'] != 0)).sum())
df.loc[cond, 'Breed2'] = 0

# 3-4. 번호 → 이름
color_dict = dict(zip(colors['ColorID'], colors['ColorName']))
state_dict = dict(zip(states['StateID'], states['StateName']))

df['Breed1Name'] = df['Breed1'].map(breed_dict)
df['Breed2Name'] = df['Breed2'].map(breed_dict)
df['Color1Name'] = df['Color1'].map(color_dict)
df['Color2Name'] = df['Color2'].map(color_dict)
df['Color3Name'] = df['Color3'].map(color_dict)
df['StateName'] = df['State'].map(state_dict)


# 4. 이름 유무 HasName (있으면 1, 없으면 0)
#    규칙으로 먼저 거르고 → 이름이라고 나온 것을 직접 검토
def has_name(name):
    if pd.isna(name):                                         # 빈칸
        return 0
    n = str(name).strip()

    if n.lower() in breed_list:                               # 품종 이름  "Poodle"
        return 0
    if re.search(r'no\s*name', n, re.I):                      # "No Name"
        return 0
    if re.search(r'\d+\s*(mths?|months?|weeks?|wks?|days?)\s*old', n, re.I):   # "2 months old"
        return 0
    if re.search(r'\b\d+\s*(cute\s+|little\s+|female\s+|male\s+|baby\s+)*'
                 r'(kitten|puppy|puppies|kittens|kitties|pups?)\b', n, re.I):   # "3 puppies"
        return 0
    if re.search(r'\b(puppy|kitten|kitty|kittens|puppies)\s*\d+\b|\bthe\s*\d+\b', n, re.I):  # "Kitten 2"
        return 0
    if re.search(r'for\s*adoption|need.*home|to\s*be\s*adopted', n, re.I):   # "For Adoption"
        return 0
    if re.match(r'^(boy|girl|male|female|kitten|kittens|puppy|puppies|cats?|dogs?|'
                r'unknown|stray|n/?a|tbd|none|null)$', n, re.I):           # "Puppy" 한 단어
        return 0
    if re.match(r'^[A-Za-z]\d+$', n):                         # "A1" 같은 코드
        return 0
    if len(n) <= 2 and not re.match(r'^([A-Za-z])\1$', n):    # 두 글자 이하 ("Bb"는 인정)
        return 0
    if re.match(r'^[A-Za-z]{1,5}[\s-]*(male|female|f|m)?[\s-]*\d{2,}', n, re.I):  # "PF123 ..."
        # 코드 뒤에 진짜 이름이 붙어 있으면 인정  "PF123 Lucky"
        rest = re.sub(r'\(\d+\)|\b\d+\b|\b(pf|io|ck|mpsp|male|female|f|m)\b|[-:;,.!]', ' ', n, flags=re.I)
        for word in rest.split():
            if len(word) >= 3 and word.isalpha():
                return 1
        return 0
    return 1


df['HasName'] = df['Name'].apply(has_name)
print('규칙으로 HasName=1:', df['HasName'].sum())

# 직접 검토해서 이름이 아니라고 표시한(X) 것 → 0
review = pd.read_csv('data/manual/hasname_manual_review.csv')
x_list = review[review['manual_check'] == 'X']['PetID']
df.loc[df['PetID'].isin(x_list), 'HasName'] = 0
print('검토 후 HasName=1:', df['HasName'].sum())


# 5. 설명글 영어 번역본
#    말레이어·중국어가 섞인 글을 영어로 통일한 결과 (번역은 미리 ai를 활용해 작업한 파일 사용)
trans = pd.read_csv('data/manual/description_en_clean.csv')
df = df.merge(trans, on='PetID', how='left')


# 6. 입양 완료 표시 Adopted
#    이미 입양돼서 설명글이 "입양됐어요, 감사합니다"로 바뀐 공고 → 직접 읽고 표시
adopted = pd.read_csv('data/manual/adopted_manual.csv')
df['Adopted'] = 0
df.loc[df['PetID'].isin(adopted['PetID']), 'Adopted'] = 1
print('Adopted=1:', df['Adopted'].sum())


# 7. 파생변수
# 사진 유무
df['HasPhoto'] = (df['PhotoAmt'] > 0).astype(int)

# 입양비 유무
df['HasFee'] = (df['Fee'] > 0).astype(int)

# 의료관리 점수: 예방접종·구충·중성화 중 한 것의 개수 (0~3)
df['MedicalCare_score'] = ((df['Vaccinated'] == 1).astype(int)
                           + (df['Dewormed'] == 1).astype(int)
                           + (df['Sterilized'] == 1).astype(int))

# 잡종 여부: Mixed Breed(307) / 두 품종이 다름 / 고양이 Domestic 품종(264~266) 중 하나라도 해당
mixed = (df['Breed1'] == 307) | (df['Breed2'] == 307)
two_breeds = (df['Breed2'] != 0) & (df['Breed1'] != df['Breed2'])
domestic = (df['Breed1'].isin([264, 265, 266])) | (df['Breed2'].isin([264, 265, 266]))
df['IsMixed'] = (mixed | two_breeds | domestic).astype(int)

# 구조자별 공고 수
df['RescuerID_PostCount'] = df.groupby('RescuerID')['PetID'].transform('count')


# 8. PetID·이름 정리
#    엑셀에서 깨지던 PetID는 우리가 정한 번호로 바꿔서 사용
pid = pd.read_csv('data/manual/petid_team_mapping.csv', dtype=str)
for i in range(len(pid)):
    df.loc[df['PetID'] == pid['PetID_원본'][i], 'PetID'] = pid['PetID_팀'][i]

df.loc[df['Name'] == 'October 09', 'Name'] = '09-Oct'    # 이름 표기 통일


# 9. 저장
cols = ['PetID', 'RescuerID', 'RescuerID_PostCount', 'Type', 'Quantity', 'Name', 'HasName',
        'Age', 'Gender', 'Breed1', 'Breed1Name', 'Breed2', 'Breed2Name', 'IsMixed',
        'Color1', 'Color1Name', 'Color2', 'Color2Name', 'Color3', 'Color3Name',
        'MaturitySize', 'FurLength', 'Health', 'Vaccinated', 'Dewormed', 'Sterilized',
        'MedicalCare_score', 'Fee', 'HasFee', 'State', 'StateName', 'VideoAmt',
        'PhotoAmt', 'HasPhoto', 'Description', 'Description_en_clean', 'Adopted', 'AdoptionSpeed']
df = df[cols]
df.to_csv('output/train_preprocessed.csv', index=False, encoding='utf-8-sig')
print('저장 완료:', df.shape)
