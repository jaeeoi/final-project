# 07. SHAP: CatBoost가 각 변수를 어느 방향으로 썼는지 해석
# 입력: output/train_joined_final.csv, output/catboost_final.cbm (06번)
# 출력: output/shap_summary.png, output/shap_importance.png, output/shap_direction.csv
#
# 그래프 읽는 법: 오른쪽(+) = 입양 쪽 / 왼쪽(-) = 미입양 쪽, 빨강 = 값이 큼 / 파랑 = 작음
# ※ Breed1, State는 번호라서 '값이 크다/작다'로 해석하면 안 됨
#
# pip install shap catboost

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# 1. 06번과 똑같이 데이터 준비 (같은 test 데이터 사용)
df = pd.read_csv('output/train_joined_final.csv')
df = df[df['Adopted'] == 0]
y = (df['AdoptionSpeed'] < 4).astype(int)

df = pd.get_dummies(df, columns=['Color1', 'Color2', 'Color3'], dtype=int)
color_cols = [c for c in df.columns if c.startswith(('Color1_', 'Color2_', 'Color3_'))]

basic = ['Type', 'Age', 'Gender', 'MaturitySize', 'FurLength', 'Health',
         'State', 'VideoAmt', 'PhotoAmt', 'Breed1', 'Breed2']
derived = ['RescuerID_PostCount', 'MedicalCare_score', 'HasFee',
           'IsMixed', 'HasName', 'HasPhoto']
image = ['has_human_photo', 'eyes', 'face', 'action', 'accessory',
         'subject_focus', 'occlusion', 'near', 'Closeness']
text = ['WordCount', 'Urgency', 'appearance_praise', 'friendliness',
        'direct_appeal', 'contact', 'training_mention']
X = df[basic + color_cols + derived + image + text]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)


# 2. 06번에서 저장한 CatBoost 불러와서 SHAP 계산
model = CatBoostClassifier()
model.load_model('output/catboost_final.cbm')

shap_values = shap.TreeExplainer(model).shap_values(X_test)


# 3. 그래프 2개
# 3-1. 점 그래프: 변수마다 영향 크기 + 방향
shap.summary_plot(shap_values, X_test, max_display=20, show=False)
plt.title('SHAP - CatBoost (오른쪽 = 입양 확률 ↑)')
plt.tight_layout()
plt.savefig('output/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

# 3-2. 막대 그래프: 평균 영향 크기 순위
shap.summary_plot(shap_values, X_test, plot_type='bar', max_display=20, show=False)
plt.title('SHAP 변수 중요도')
plt.tight_layout()
plt.savefig('output/shap_importance.png', dpi=150, bbox_inches='tight')
plt.close()


# 4. 변수별 방향 요약표
#    변수값과 SHAP값이 같이 커지면 '클수록 입양 ↑', 반대면 '작을수록 입양 ↑'
rows = []
for i, col in enumerate(X_test.columns):
    value = X_test[col].values.astype(float)
    ok = ~np.isnan(value)                                  # 빈칸 빼고 계산
    if ok.sum() > 10 and value[ok].std() > 0:
        corr = np.corrcoef(value[ok], shap_values[ok, i])[0, 1]
    else:
        corr = 0

    if corr > 0.05:
        direction = '클수록 입양 ↑'
    elif corr < -0.05:
        direction = '작을수록 입양 ↑'
    else:
        direction = '방향 약함'

    rows.append({'변수': col,
                 '평균 영향 크기': round(np.abs(shap_values[:, i]).mean(), 4),
                 '방향': direction})

table = pd.DataFrame(rows).sort_values('평균 영향 크기', ascending=False).head(20)
print(table.to_string(index=False))
table.to_csv('output/shap_direction.csv', index=False, encoding='utf-8-sig')
