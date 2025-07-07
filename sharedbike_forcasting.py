import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# 1. 데이터 로드
print("Step 1: 데이터 로딩 시작...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("sampleSubmission.csv")
print("데이터 로딩 완료!")

# 2. 데이터 전처리 및 특성 공학
print("\nStep 2: 데이터 전처리 및 특성 공학 시작...")

# datetime 컬럼을 datetime 객체로 변환 [cite: 28]
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

# 시간 관련 특성 추출 
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['dayofweek'] = train['datetime'].dt.dayofweek

test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['hour'] = test['datetime'].dt.hour
test['dayofweek'] = test['datetime'].dt.dayofweek

# 이상치 제거 (IQR 방식) [cite: 88]
# PDF에서는 windspeed에 이상치가 많아 보였지만, 0인 값을 제거하는 것이 일반적입니다.
# 여기서는 PDF의 결정에 따라 IQR 이상치 제거를 적용해 보겠습니다.
# 풍속(windspeed)의 0 값을 평균값으로 대체 (자주 사용되는 기법)
train.loc[train['windspeed'] == 0, 'windspeed'] = train['windspeed'].mean()
test.loc[test['windspeed'] == 0, 'windspeed'] = test['windspeed'].mean()

# RMSLE 평가 방식에 맞춰 타겟 변수 로그 변환
# 훈련 데이터에만 count 컬럼이 존재
train['count_log'] = np.log1p(train['count'])

# 불필요한 컬럼 제거 [cite: 93, 94, 95]
# 'datetime'은 세부 시간 특성으로 분리했으므로 제거
# 'atemp'는 'temp'와 상관관계가 높으므로 제거
# 'workingday'는 'holiday'와 유사하므로 제거
# 'casual', 'registered'는 test 데이터에 없고, 'count'를 직접 예측하므로 제거
train = train.drop(['datetime', 'atemp', 'workingday', 'casual', 'registered', 'count'], axis=1)
test = test.drop(['datetime', 'atemp', 'workingday'], axis=1)

# 범주형 변수 원-핫 인코딩
# test 데이터에는 없는 범주가 있을 수 있으므로, train/test를 합친 후 인코딩하고 다시 분리합니다.
all_data = pd.concat([train.drop('count_log', axis=1), test], axis=0)
categorical_features = ['season', 'holiday', 'weather', 'year', 'month', 'hour', 'dayofweek']
all_data_encoded = pd.get_dummies(all_data, columns=categorical_features, drop_first=False)

# 인코딩된 데이터를 다시 train/test로 분리
X_train = all_data_encoded.iloc[:len(train)]
X_test = all_data_encoded.iloc[len(train):]
y_train_log = train['count_log']

print("전처리 및 특성 공학 완료!")

# 3. 모델 훈련
print("\nStep 3: Gradient Boosting 모델 훈련 시작...")

# PDF에 명시된 하이퍼파라미터 사용 
gbr = GradientBoostingRegressor(n_estimators=2000,
                                learning_rate=0.05,
                                max_depth=5,
                                min_samples_leaf=15,
                                min_samples_split=10,
                                random_state=42)

gbr.fit(X_train, y_train_log)
print("모델 훈련 완료!")

# 4. 예측
print("\nStep 4: 테스트 데이터 예측 시작...")
predictions_log = gbr.predict(X_test)

# 로그 변환된 예측값을 원래 스케일로 되돌리기
predictions = np.expm1(predictions_log)

# 예측값이 음수일 경우 0으로 처리
predictions[predictions < 0] = 0
print("예측 완료!")

# 5. 제출 파일 생성
print("\nStep 5: 제출 파일 생성 시작...")
submission['count'] = predictions
submission.to_csv("submission.csv", index=False)
print("submission.csv 파일이 성공적으로 생성되었습니다!")