# 04. 설명글에서 변수 7개 만들기
# 입력: output/train_preprocessed.csv
# 출력: output/text_features.csv
#
# 변수
#   WordCount          설명글 단어 수 (원문 사용)                  - 직접 계산
#   Urgency            진짜 위급한 상황인지 (안락사, 마감일 등)     ┐
#   appearance_praise  외모 칭찬                                  │
#   friendliness       성격·친화성 묘사                           │ Claude API
#   direct_appeal      "입양해 주세요" 같은 직접 호소              │ (영어 번역본 사용)
#   contact            연락 방법 안내                             │
#   training_mention   배변·훈련 언급                             ┘
#
# 방법 선택: 키워드 → 다의어 오류 ("pretty": 예쁜/꽤) / 제로샷 → 글 길이 편향 / Claude API → 최종
#
# 준비
#   pip install anthropic jieba pandas
#   환경변수 ANTHROPIC_API_KEY 에 API 키 넣기

import re
import json
import jieba
import pandas as pd
import anthropic
from concurrent.futures import ThreadPoolExecutor

df = pd.read_csv('output/train_preprocessed.csv', dtype={'PetID': str})


# 1. WordCount (단어 수)
#    영어·말레이어는 알파벳 단어 1개씩 (dog, months, don't), 중국어는 jieba로 나눠서 세기
def word_count(text):
    if pd.isna(text) or str(text).strip() == '':
        return 0
    text = str(text)

    # 중국어 단어 수 ([\u4e00-\u9fff] = 한자 범위)
    chinese = 0
    for part in re.findall(r'[\u4e00-\u9fff]+', text):
        words = [w for w in jieba.lcut(part) if w.strip()]
        chinese += len(words)

    # 중국어를 지우고 알파벳 단어 수
    text = re.sub(r'[\u4e00-\u9fff]+', ' ', text)
    english = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))

    return english + chinese


df['WordCount'] = df['Description'].apply(word_count)


# 2. Claude API 연결 (claude-haiku-4-5, 글은 앞 1500자까지만)
client = anthropic.Anthropic()


def ask_claude(prompt, max_tokens):
    res = client.messages.create(model='claude-haiku-4-5', max_tokens=max_tokens,
                                 messages=[{'role': 'user', 'content': prompt}])
    return res.content[0].text.strip()


# 3. Urgency (말버릇 같은 재촉이 아니라 진짜 위급할 때만 높은 점수)
urgency_prompt = """You are analyzing pet adoption descriptions to rate genuine urgency.

Rate from 0 to 1 how much this description indicates a GENUINE urgent need for adoption \
- meaning explicit risk of euthanasia, a concrete deadline, the pet being taken away or killed soon, \
or similarly severe time pressure.

Do NOT give a high score just because the text uses persuasive phrases like "please call urgently" \
or "adopt soon" as a generic sales pitch. Only score high when there is a real, concrete threat or \
deadline described in the text. Rhetorical urgency with no concrete threat should score low-to-moderate \
at most; genuine crisis situations (euthanasia risk, being taken away, a specific deadline) should score high.

Respond with ONLY a number between 0 and 1 (e.g. 0.05, 0.42, 0.91). No explanation, no other text.

Description: {text}"""


def score_urgency(text):
    try:
        answer = ask_claude(urgency_prompt.format(text=str(text)[:1500]), 10)
        return float(re.search(r'[01](\.\d+)?', answer).group())
    except Exception as e:
        print('Urgency 실패:', str(e)[:50])
        return None


# 4. 나머지 5개 변수도 채점
category_prompt = """You are analyzing a pet adoption description. Rate the following 5 categories, \
each from 0 to 1, based ONLY on this specific definition for each:

1. AppearancePraise: praise of the PET's physical appearance/cuteness (e.g. cute, adorable, beautiful, \
handsome, fluffy). Do NOT count "pretty" used as an intensifier (e.g. "pretty active", "pretty good") \
or praise directed at something other than the pet's looks.

2. Friendliness: describes the PET's own temperament/personality (playful, affectionate, calm, sociable, \
gentle, energetic, etc). Do NOT count "kind" describing a person/adopter (e.g. "kind soul"), or "warm" \
describing a home/family rather than the pet's own personality.

3. DirectAppeal: an emotional plea or persuasive call to adopt this pet (e.g. "please give her a home", \
"looking for a loving home", "please adopt"). This is about rhetorical style, not about verifying urgency.

4. Contact: mentions a way to reach the poster (call, message, email, whatsapp, sms, etc). Do NOT count \
the pet's own behavioral responsiveness (e.g. "responds to her name" is about training, not contact).

5. TrainingMention: describes the pet's training/obedience status (toilet trained, obedient, disciplined, \
house trained, etc).

Respond with ONLY a JSON object in this exact format, no other text:
{{"appearance_praise": 0.0, "friendliness": 0.0, "direct_appeal": 0.0, "contact": 0.0, "training_mention": 0.0}}

Description: {text}"""

category_names = ['appearance_praise', 'friendliness', 'direct_appeal', 'contact', 'training_mention']


def score_categories(text):
    try:
        answer = ask_claude(category_prompt.format(text=str(text)[:1500]), 150)
        return json.loads(re.search(r'\{.*\}', answer, re.DOTALL).group())
    except Exception as e:
        print('5개 항목 실패:', str(e)[:50])
        return {}


# 5. 글 1개 채점 → 점수 6개 출력 (글자가 하나도 없는 글은 API 없이 0점)
score_cols = ['Urgency'] + category_names


def score_text(text):
    if not re.search(r'[A-Za-z\u4e00-\u9fff]', str(text)):
        return {col: 0.0 for col in score_cols}
    result = {'Urgency': score_urgency(text)}
    result.update(score_categories(text))
    return result


# 6. 채점 실행 (8개씩 동시에 요청, 영어 번역본이 있는 글만)
target = df[df['Description_en_clean'].notna()]
with ThreadPoolExecutor(max_workers=8) as ex:
    scores = list(ex.map(score_text, target['Description_en_clean']))

scores = pd.DataFrame(scores, index=target.index)
for col in score_cols:
    df[col] = scores[col]


# 7. 저장
out = df[['PetID', 'WordCount'] + score_cols]
out.to_csv('output/text_features.csv', index=False)
print('저장 완료:', out.shape)
