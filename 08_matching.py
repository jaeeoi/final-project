# 08. 마할라노비스 매칭: 조건이 비슷한 동물끼리 짝지어서 사진·글의 효과만 보기
# 입력: output/train_joined_final.csv
# 출력: output/matching_전체.csv, matching_개.csv, matching_고양이.csv
#
# SHAP에서는 나이·품종·구조자처럼 바꿀 수 없는 조건이 예측을 주도함
# → 조건이 비슷한 동물끼리 짝을 지어 "사진·글 점수가 높은 쪽 vs 낮은 쪽" 입양률 비교
#
# 방법
#   1. 정형 조건 13개로 동물마다 벡터 만들기 (범주형은 원-핫, 숫자형은 표준화)
#   2. 마할라노비스 거리 = 변수끼리 상관까지 고려한 거리
#   3. 확인할 변수를 중앙값으로 높은 쪽(처치군) / 낮은 쪽(대조군) 나누기
#      (값이 0에 몰려 있으면 '0 vs 0보다 큼')
#   4. 처치군 동물마다 가장 가까운 대조군 1마리와 짝짓기
#   5. 거리가 먼 하위 50% 짝은 버림
#   6. 짝끼리 입양률 차이 → 대응표본 t검정 → FDR 보정 (여러 변수를 동시에 검정하므로)
#
# pip install pandas numpy scipy scikit-learn statsmodels

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


# 1. 데이터 준비 (입양 완료 글은 글 변수가 빈칸이라 그 변수에서만 자동으로 빠짐)
df = pd.read_csv('output/train_joined_final.csv')

df['adopted'] = (df['AdoptionSpeed'] < 4).astype(int)               # 100일 안 입양

top5 = df['Breed1'].value_counts().head(5).index                    # 품종은 상위 5개 + 기타
df['Breed1_grouped'] = df['Breed1'].where(df['Breed1'].isin(top5), -1)

df['RescuerID_PostCount_log'] = np.log1p(df['RescuerID_PostCount'])  # 한쪽으로 쏠린 값 완화

# 확인할 변수: 이미지 9개 + 텍스트 7개 + 공고에서 바꿀 수 있는 정형 2개
check_vars = ['has_human_photo', 'eyes', 'face', 'action', 'accessory',
              'subject_focus', 'occlusion', 'near', 'Closeness',
              'WordCount', 'Urgency', 'appearance_praise', 'friendliness',
              'direct_appeal', 'contact', 'training_mention',
              'PhotoAmt', 'VideoAmt']

# 짝지을 때 비교하는 조건 13개
cat_vars = ['Type', 'Gender', 'Health', 'MaturitySize', 'Breed1_grouped',
            'FurLength', 'State', 'IsMixed', 'HasFee', 'HasName']          # 범주형 10개
num_vars = ['Age', 'MedicalCare_score', 'RescuerID_PostCount_log']        # 숫자형 3개


# 2. 변수 1개 매칭
def match_one(data, X, inv_cov, var):
    # 이 변수 값이 있는 동물만
    ok = data[var].notna()
    d = data[ok]
    Xd = X[ok]

    # 처치군(높은 쪽) / 대조군(낮은 쪽)
    if var == 'has_human_photo':
        treat = d[var] > 0
    elif d[var].median() == 0:                    # 0이 절반 넘으면 '0 vs 0보다 큼'
        treat = d[var] > 0
    else:
        treat = d[var] >= d[var].median()
    if treat.sum() < 20 or (~treat).sum() < 20:
        return None

    # 처치군마다 가장 가까운 대조군 1마리 찾기
    nn = NearestNeighbors(n_neighbors=1, metric='mahalanobis',
                          metric_params={'VI': inv_cov}, algorithm='brute')
    nn.fit(Xd[~treat].values)
    dist, idx = nn.kneighbors(Xd[treat].values)
    dist = dist.ravel()
    idx = idx.ravel()

    # 거리가 먼 하위 50% 짝은 버림
    keep = dist <= np.percentile(dist, 50)
    t_result = d[treat]['adopted'].values[keep]
    c_result = d[~treat]['adopted'].values[idx[keep]]

    # 짝끼리 입양률 차이 + t검정
    _, p = stats.ttest_rel(t_result, c_result)
    return {'변수': var, '짝 수': len(t_result),
            '입양률 차이(%p)': round((t_result.mean() - c_result.mean()) * 100, 2),
            'p_value': p}


# 3. 그룹별 실행 (전체 / 개 / 고양이)
def run(data, label, cat_list):
    data = data.reset_index(drop=True)

    # 조건 → 숫자 벡터 (범주형 원-핫 + 숫자형 표준화)
    X_cat = pd.get_dummies(data[cat_list].astype(str), drop_first=True).astype(float)
    X_num = pd.DataFrame(StandardScaler().fit_transform(data[num_vars]), columns=num_vars)
    X = pd.concat([X_num, X_cat], axis=1)

    # 마할라노비스 거리에 쓰는 공분산 역행렬 (안정적인 pinv 사용)
    inv_cov = np.linalg.pinv(np.cov(X.values, rowvar=False))

    rows = []
    for var in check_vars:
        r = match_one(data, X, inv_cov, var)
        if r is not None:
            rows.append(r)
    result = pd.DataFrame(rows)

    # 여러 변수를 동시에 검정 → FDR 보정
    result['p_fdr'] = multipletests(result['p_value'], method='fdr_bh')[1]
    result['유의'] = result['p_fdr'] < 0.05
    result = result.sort_values('p_fdr')

    print('\n=====', label, len(data), '마리 =====')
    print(result.round(4).to_string(index=False))
    result.to_csv(f'output/matching_{label}.csv', index=False, encoding='utf-8-sig')


run(df, '전체', cat_vars)

# 종별 (종 안에서는 Type이 다 같으니까 조건에서 뺌)
cat_no_type = [c for c in cat_vars if c != 'Type']
run(df[df['Type'] == 1], '개', cat_no_type)
run(df[df['Type'] == 2], '고양이', cat_no_type)
