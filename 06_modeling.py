# 06. 장기 미입양 예측 모델 4개 비교 → CatBoost 최종 선택
# 입력: output/train_joined_final.csv
# 출력: output/model_comparison.csv, output/catboost_final.cbm, output/catboost_importance.csv
#
# pip install scikit-learn xgboost lightgbm catboost pandas

import warnings
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')


# 1. 데이터 (입양 완료 글 제외)
df = pd.read_csv('output/train_joined_final.csv')
df = df[df['Adopted'] == 0]

# 타겟: 100일 안에 입양(AdoptionSpeed 0~3) = 1 / 100일 넘게 미입양(4) = 0
y = (df['AdoptionSpeed'] < 4).astype(int)


# 2. 변수 53개
#    색상은 원-핫 인코딩 / 결측은 채우지 않음 (4개 모델 모두 결측 처리 가능)
df = pd.get_dummies(df, columns=['Color1', 'Color2', 'Color3'], dtype=int)
color_cols = [c for c in df.columns if c.startswith(('Color1_', 'Color2_', 'Color3_'))]

basic = ['Type', 'Age', 'Gender', 'MaturitySize', 'FurLength', 'Health',
         'State', 'VideoAmt', 'PhotoAmt', 'Breed1', 'Breed2']                       # 정형 11개
derived = ['RescuerID_PostCount', 'MedicalCare_score', 'HasFee',
           'IsMixed', 'HasName', 'HasPhoto']                                        # 파생 6개
image = ['has_human_photo', 'eyes', 'face', 'action', 'accessory',
         'subject_focus', 'occlusion', 'near', 'Closeness']                         # 이미지 9개
text = ['WordCount', 'Urgency', 'appearance_praise', 'friendliness',
        'direct_appeal', 'contact', 'training_mention']                             # 텍스트 7개

features = basic + color_cols + derived + image + text                             # 색상 20개 포함
X = df[features]
print('데이터:', X.shape)


# 3. train 80% / test 20% 분리 (입양·미입양 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
ratio = (y_train == 0).sum() / (y_train == 1).sum()     # XGBoost 불균형 보정용


# 4. 모델 4개 + 파라미터 후보
#    과적합 방지: 트리는 얕게, 학습률은 낮게
#    불균형 보정(입양 74% : 미입양 26%)을 할지 말지도 후보에 넣어서 결정
models = {
    'XGBoost': (
        XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=1),
        {'n_estimators': [100, 200, 300], 'max_depth': [2, 3, 4],
         'learning_rate': [0.01, 0.02, 0.03], 'min_child_weight': [10, 20, 30],
         'subsample': [0.6, 0.8], 'colsample_bytree': [0.6, 0.8],
         'reg_alpha': [1, 5], 'reg_lambda': [5, 10],
         'scale_pos_weight': [1, ratio]}),
    'LightGBM': (
        LGBMClassifier(random_state=42, verbosity=-1, n_jobs=1),
        {'n_estimators': [100, 200, 300], 'max_depth': [2, 3, 4],
         'learning_rate': [0.01, 0.02, 0.03], 'num_leaves': [7, 15],
         'min_child_samples': [30, 50, 80], 'reg_alpha': [1, 5], 'reg_lambda': [5, 10],
         'class_weight': [None, 'balanced']}),
    'CatBoost': (
        CatBoostClassifier(random_state=42, verbose=0, thread_count=1),
        {'iterations': [300, 500], 'depth': [3, 4],
         'learning_rate': [0.01, 0.03, 0.05], 'l2_leaf_reg': [10, 20, 30],
         'auto_class_weights': ['Balanced', 'SqrtBalanced']}),
    'HistGB': (
        HistGradientBoostingClassifier(random_state=42),
        {'max_iter': [100, 200, 300], 'max_depth': [2, 3, 4],
         'learning_rate': [0.01, 0.02, 0.03], 'min_samples_leaf': [30, 50, 80],
         'l2_regularization': [5.0, 10.0, 20.0], 'class_weight': [None, 'balanced']}),
}


# 5. 모델마다 GridSearchCV 진행 (3-fold, Macro F1 기준)
results = []
best_models = {}

for name in models:
    model, grid = models[name]
    print('\n=====', name, '=====')
    gs = GridSearchCV(model, grid, cv=3, scoring='f1_macro', n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    best_models[name] = best
    print('최적 파라미터:', gs.best_params_)

    train_f1 = f1_score(y_train, best.predict(X_train), average='macro')
    test_f1 = f1_score(y_test, best.predict(X_test), average='macro')
    test_auc = roc_auc_score(y_test, best.predict_proba(X_test)[:, 1])
    test_acc = accuracy_score(y_test, best.predict(X_test))

    results.append({'모델': name,
                    'Test_MacroF1': round(test_f1, 4),
                    'Test_ROC_AUC': round(test_auc, 4),
                    'Test_Accuracy': round(test_acc, 4),
                    '과적합폭': round(train_f1 - test_f1, 4)})    # train F1 - test F1

result = pd.DataFrame(results).sort_values('Test_MacroF1', ascending=False)
print('\n[모델 비교]')
print(result)
result.to_csv('output/model_comparison.csv', index=False, encoding='utf-8-sig')


# 6. 최종 모델 CatBoost 저장 + 변수 중요도
final = best_models['CatBoost']
final.save_model('output/catboost_final.cbm')

importance = pd.Series(final.get_feature_importance(), index=features)
importance = importance.sort_values(ascending=False)
print('\n[CatBoost 변수 중요도 TOP 20]')
print(importance.head(20).round(2))
importance.to_csv('output/catboost_importance.csv', header=['importance'], encoding='utf-8-sig')
