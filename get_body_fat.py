import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , Ridge , Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
%matplotlib inline

# File load
#index는 dataframe변환시 주어지고, 표준몸무게는 키로 정해지므로 삭제
file_df = pd.read_csv('C:/Users/onesoftdigm/Documents/data/body_fat_train_data_age.csv')
# file_df = train_df
target_name = 'Body_Fat_Rate'
no_need_features = ['Index', 'Standard_Weight']
category_features = ['Sex']

# feature값 정리
file_df.drop(no_need_features, axis=1, inplace=True)
y_target = file_df[target_name]
X_features = file_df.drop([target_name],axis=1,inplace=False)

# 아웃라이어 제거
outlier_name = 'Height'
cond1 = file_df[outlier_name] < 60
cond2 = file_df[target_name] < 30
outlier_index = X_features[cond1 & cond2].index
X_features.drop(outlier_index , axis=0, inplace=True)
y_target.drop(outlier_index, axis=0, inplace=True)

# 카테고리형 feature를 One Hot Encoding, 성별을 OHE
X_features_ohe = pd.get_dummies(X_features, columns=category_features)

# 타겟 컬럼값을 정규 분포 형태로 만들기 위해 log1p 로 Log 변환
y_target_log = np.log1p(y_target)

# 원-핫 인코딩이 적용된 feature 데이터 세트 기반으로 학습/예측 데이터 분할.
X_train, X_test, y_train, y_test = train_test_split(X_features_ohe, y_target_log, test_size=0.2, random_state=0)

#단일 모델
model = Lasso(alpha=0.03)
model.fit(X_train, y_train)

# #혼합 모델
# model1 = LinearRegression()
# model2 = Ridge(alpha=1)
# model1.fit(X_train, y_train)
# model2.fit(X_train, y_train)

# File load
#index는 dataframe변환시 주어지고, 표준몸무게는 키로 정해지므로 삭제
test_df = pd.read_csv('C:/Users/onesoftdigm/Documents/data/body_fat_test_data_age.csv')
print(test_df)

# feature값 정리
test_df.drop(no_need_features, axis=1, inplace=True)
y_test = test_df[target_name]
X_test = test_df.drop([target_name],axis=1,inplace=False)

#카테고리형 feature를 One Hot Encoding, 성별을 OHE
X_test_ohe = pd.get_dummies(X_test, columns=category_features)

#단일 모델
predict_value = model.predict(X_test_ohe)
predict_final = np.expm1(predict_value)
predict_final
for x in predict_final:
    print(round(x,1))

# # 혼합 모델
# pred1 = model1.predict(X_test_ohe)
# pred2 = model2.predict(X_test_ohe)
# pred = 0.8 * pred1 + 0.2 * pred2
# np.expm1(pred)
# for x in np.expm1(pred):
#     print(round(x,1))