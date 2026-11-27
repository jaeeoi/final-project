# 03. 친밀도(Closeness): 사진 속 동물이 사람에게 친근해 보이는 정도 (0~1)
# 1부. 방법 고르기: 팀원 5명이 직접 라벨링하여 매긴 점수와 AI 3종(CLIP, Claude, Gemini) 비교
# 2부. 고른 방법(Gemini, Vertex AI)으로 전체 사진 채점 → RUN_GEMINI = True 일 때만 실행
#
# 준비 (2부)
#   pip install google-genai pandas
#   gcloud auth application-default login
#   환경변수 GCP_PROJECT_ID 에 구글 클라우드 프로젝트 ID 넣기

import os
import re
import sys
import time
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from concurrent.futures import ThreadPoolExecutor

RUN_GEMINI = False          # True로 바꾸면 2부까지 실행 (API 비용 발생)


# ===== 1부. 방법 고르기 =====
# 팀원 5명이 같은 기준표로 사진 4000장을 직접 채점 (0, 0.25, 0.5, 0.75, 1)
# AI 3종과 비교 - 기준: 팀원 평균과 순위가 얼마나 비슷한지 (스피어만 순위상관)
label = pd.read_csv('data/manual/closeness_label_compare.csv')
print('비교한 사진 수:', len(label))

# 1-1. 팀원끼리 얼마나 일치하는지 (2명씩 짝지어 순위상관 → 평균)
raters = ['팀원1', '팀원2', '팀원3', '팀원4', '팀원5']
pairs = []
for a, b in itertools.combinations(raters, 2):
    pairs.append(spearmanr(label[a], label[b])[0])
print('팀원끼리 일치도:', round(np.mean(pairs), 3))

# 1-2. AI 3종 vs 팀원 평균
for ai in ['CLIP', 'Claude', 'Gemini']:
    r = spearmanr(label['팀평균'], label[ai])[0]
    print(f'{ai:7s} vs 팀원 평균:', round(r, 3))

# → Gemini가 가장 높고 팀원끼리 일치도보다도 높음 → Gemini 선택
if not RUN_GEMINI:
    print('\n1부 끝. Gemini 전체 채점은 RUN_GEMINI = True 로 바꾸고 실행')
    sys.exit()


# ===== 2부. Gemini로 전체 사진 채점 =====
from google import genai
from google.genai import types

# 1. Gemini 연결 (Vertex AI, gemini-2.5-flash)
client = genai.Client(vertexai=True,
                      project=os.environ['GCP_PROJECT_ID'],
                      location='us-central1')


# 2. 채점 질문 (팀 라벨링 기준표와 같은 기준)
prompt = """You are analyzing a pet photo to rate how comfortable/sociable the animal appears \
toward people, on a scale from 0.0 to 1.0. Use these anchor points as your guide:

1.0 (매우 높음): 매우 편안하고 개방적인 자세 / 적극적으로 다가오는 모습 / 몸을 완전히 맡긴 자세 / \
사람에게 기대기·비비기·편안하게 안겨 있는 등 매우 강한 친밀 신호

0.75 (높음): 편안하게 눕거나 몸을 펼침 / 접근하려는 자세 / 호기심 있고 친근한 태도 / 사람과 자연스럽게 접촉

0.5 (보통): 자연스럽게 앉거나 서 있음 / 특별한 친밀·경계 신호가 없는 중립 상태

0.25 (낮음): 몸을 웅크림 / 시선·몸 방향 회피 / 다소 긴장되거나 위축된 자세

0.0 (매우 낮음): 강한 경계·공포 / 몸이 굳음 / 숨거나 도망가려는 모습 / 명확한 방어 자세

You may use any value between these anchors (e.g. 0.6, 0.85) if the photo falls between categories.
People do not need to be visible in the photo - judge purely from the animal's body language and expression.

Respond with ONLY a number between 0 and 1 (e.g. 0.05, 0.42, 0.91). No explanation, no other text."""


# 3. 사진 1장 채점 (답에서 0~1 숫자를 찾음, "판단 불가" 답이면 0점, 실패하면 빈칸)
def score_photo(path):
    try:
        with open(path, 'rb') as f:
            image = f.read()
        res = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[types.Part.from_bytes(data=image, mime_type='image/jpeg'), prompt])
        answer = res.text.strip()
        number = re.search(r'[01](\.\d+)?', answer)
        if number:
            return float(number.group())
        if re.search(r'\b(cannot|unable)\b', answer, re.I):
            return 0.0
        return None
    except Exception as e:
        print('실패:', os.path.basename(path), str(e)[:50])
        return None


# 4. 모든 사진 채점 (20장씩 동시에 요청, 사진 목록은 02번 결과 사용)
photos = pd.read_csv('output/image_features_photo_level.csv', dtype={'PetID': str})
photos = photos[['PetID', 'photo_path']].copy()
photos['filename'] = photos['photo_path'].apply(os.path.basename)

with ThreadPoolExecutor(max_workers=20) as ex:
    photos['gemini_closeness_score'] = list(ex.map(score_photo, photos['photo_path']))

print('채점 실패:', photos['gemini_closeness_score'].isna().sum(), '장')

# 사진별 결과 저장
raw = photos.dropna(subset=['gemini_closeness_score'])[['PetID', 'filename', 'gemini_closeness_score']]
raw.to_csv('output/closeness_raw.csv', index=False)


# 5. 같은 동물 사진끼리 평균
pet = raw.groupby('PetID')['gemini_closeness_score'].mean().reset_index()
pet = pet.rename(columns={'gemini_closeness_score': 'Closeness'})
pet.to_csv('output/closeness.csv', index=False)
print('저장 완료:', pet.shape)
