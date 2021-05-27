import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression , Ridge , Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
%matplotlib inline

def get_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid=params, scoring='neg_mean_squared_error', cv=5)
    grid_model.fit(X_features, y_target)
    rmse = np.sqrt(-1* grid_model.best_score_)
    print('{0} 5 CV 시 최적 평균 RMSE 값: {1}, 최적 alpha:{2}'.format(model.__class__.__name__,np.round(rmse, 4), grid_model.best_params_))
    return grid_model.best_estimator_

def get_model_predict(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    #예측 결과값이 로그 변환된 타깃 기반으로 학습돼 예측됐으므로 다시 expm1로 스케일 변환
    y_test = np.expm1(y_test)
    pred = np.expm1(pred)
    print('\n###',model.__class__.__name__,'###')
    evaluate_regr(y_test, pred)

# MAE, MSE, RMSE, RMSLE 를 모두 계산
def evaluate_regr(y,pred):
    mae_val = mean_absolute_error(y,pred)
    mse_val = mean_squared_error(y,pred)
    rmse_val = rmse(y,pred)
    rmsle_val = rmsle(y,pred)
    r2_val = r2_score(y, pred)
    print('MAE: {0:.3F}, MSE: {2:.3F}, RMSE: {1:.3F}, RMSLE: {0:.3F}, R2: {3:.3F}'.format(mae_val, mse_val, rmse_val, rmsle_val, r2_val))

# log 값 변환 시 NaN등의 이슈로 log() 가 아닌 log1p() 를 이용하여 RMSLE 계산
def rmsle(y, pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred) ** 2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle

# 사이킷런의 mean_square_error() 를 이용하여 RMSE 계산
def rmse(y,pred):
    return np.sqrt(mean_squared_error(y,pred))

# 피처들의 coefficient를 시각화
def visualize_coefficient(models):
    fig, axs = plt.subplots(figsize=(24,10),nrows=1, ncols=4)
    fig.tight_layout()
    for i_num, model in enumerate(models):
        coef_high, coef_low = get_top_bottom_coef(model)
        coef_concat = pd.concat( [coef_high , coef_low] )
        axs[i_num].set_title(model.__class__.__name__+' Coeffiecents', size=25)
        axs[i_num].tick_params(axis="y",direction="in", pad=-120)
        for label in (axs[i_num].get_xticklabels() + axs[i_num].get_yticklabels()):
            label.set_fontsize(22)
        sns.barplot(x=coef_concat.values, y=coef_concat.index , ax=axs[i_num])

# 피처들 중 상위 n개, 하위 n개 coefficient 추출
def get_top_bottom_coef(model):
    # coef_ 속성을 기반으로 Series 객체를 생성. index는 컬럼명.
    coef = pd.Series(model.coef_, index=X_features_ohe.columns)
    coef_high = coef.sort_values(ascending=False).head(3)
    coef_low = coef.sort_values(ascending=False).tail(3)
    return coef_high, coef_low

def print_coefficient(models):
    for model in models:
        print('\n###',model.__class__.__name__,'###')
        coeff = pd.Series(data=np.round(model.coef_, 3), index=X_features_ohe.columns )
        print(coeff.sort_values(ascending=False))

# 피처들의 coefficient를 시각화
def visualize_ftr_importances(models):
    fig, axs = plt.subplots(figsize=(24,10),nrows=1, ncols=5)
    fig.tight_layout()
    for i_num, model in enumerate(models):
        ftr_top6 = get_top_features(model)
        axs[i_num].set_title(model.__class__.__name__+' Feature Importances', size=17)
        for label in (axs[i_num].get_xticklabels() + axs[i_num].get_yticklabels()):
            label.set_fontsize(22)
        sns.barplot(x=ftr_top6.values, y=ftr_top6.index , ax=axs[i_num])

# 모델의 중요도 상위 6개의 피처명과 그때의 중요도값을 Series로 반환.
def get_top_features(model):
    ftr_importances_values = model.feature_importances_
    ftr_importances = pd.Series(ftr_importances_values, index=X_features_ohe.columns)
    ftr_top6 = ftr_importances.sort_values(ascending=False)[:6]
    return ftr_top6

# File load
##index는 dataframe변환시 주어지고, 표준몸무게는 키로 정해지므로 삭제
file_df = pd.read_csv('./body_fat_data.csv')
target_name = 'Body_Fat_Rate'
no_need_features = ['Index', 'Standard_Weight']
category_features = ['Sex']

# feature값 정리
file_df.drop(no_need_features, axis=1, inplace=True)
y_target = file_df[target_name]
X_features = file_df.drop([target_name],axis=1,inplace=False)

# 아웃라이어 보여주기
for feature in X_features.drop(category_features, axis=1, inplace=False):
    plt.scatter(x = file_df[feature], y = y_target)
    plt.ylabel(target_name, fontsize=15)
    plt.xlabel(feature, fontsize=15)
    plt.show()

# 아웃라이어 제거
outlier_name = 'Height'
cond1 = file_df[outlier_name] < 60
cond2 = file_df[target_name] < 30
outlier_index = X_features[cond1 & cond2].index
print('아웃라이어 레코드 index :', outlier_index.values)
print('아웃라이어 삭제 전 X_feature shape:', X_features.shape)
X_features.drop(outlier_index , axis=0, inplace=True)
y_target.drop(outlier_index, axis=0, inplace=True)
print('아웃라이어 삭제 후 file_ohe shape:', X_features.shape)

# feature들의 왜곡 정도를 파악 --> 왜곡 정도가 높으면(1이상) 로그 변환
# height 로그 변환 필요
features_index = file_df.drop(category_features, axis=1, inplace=False).dtypes.index
skew_features = file_df[features_index].apply(lambda x : skew(x))
print(skew_features.sort_values(ascending=False))
skew_features_change = skew_features[skew_features < -1]
file_df[skew_features_change.index] = np.log1p(file_df[skew_features_change.index])

# 카테고리형 feature를 One Hot Encoding, 성별을 OHE
X_features_ohe = pd.get_dummies(X_features, columns=category_features)
# print(X_features_ohe)

# 타겟 컬럼값을 정규 분포 형태로 만들기 위해 log1p 로 Log 변환
y_target_log = np.log1p(y_target)
# print(y_target)
# print(y_target_log)

# 원-핫 인코딩이 적용된 feature 데이터 세트 기반으로 학습/예측 데이터 분할.
X_train, X_test, y_train, y_test = train_test_split(X_features_ohe, y_target_log, test_size=0.2, random_state=0)

# 회귀 모델 정의
lr_reg = LinearRegression()
ridge_reg = Ridge(random_state=0, alpha=0.11)
lasso_reg = Lasso(alpha=0.01)
en_reg = ElasticNet(alpha=0.07, l1_ratio=0.2)
dt_reg = DecisionTreeRegressor(max_depth=7)
rf_reg = RandomForestRegressor(random_state=0, n_estimators=700, max_depth=14, min_samples_leaf=2, min_samples_split=2, n_jobs=-1)
gbm_reg = GradientBoostingRegressor(n_estimators=500, learning_rate=0.02, subsample=0.05)
xgb_reg = XGBRegressor(n_estimators=120, eta=0.1, min_child_weight=3, max_depth=3)
lgbm_reg = LGBMRegressor(n_estimators=1000, learning_rate=0.03, max_depth=3, min_child_samples=10, num_leaves=3)

# 최적화 파라미터 찾기
# ridge_params = { 'alpha':[0.01, 0.05, 0.09, 0.1, 0.11, 0.12, 0.5, 1, 3, 5, 8, 10, 12, 15, 20, 30, 40, 50] }
# lasso_params = { 'alpha':[0.01, 0.05, 0.09, 0.1, 0.11, 0.12, 0.5, 1, 3, 5, 8, 10, 12, 15, 20, 30, 40, 50]  }
# en_params = { 'alpha':[0.07, 0.1, 0.5, 1, 3] }
# rf_params = {'n_estimators':[700], 'max_depth' : [14], 'min_samples_leaf' : [2], 'min_samples_split' : [2]}
# gbm_params = {'n_estimators':[500], 'learning_rate': [0.02], 'subsample': [0.05]}
# xgb_params = {'n_estimators':[120], 'eta': [0.1], 'min_child_weight': [3], 'max_depth': [3], 'colsample_bytree': [1]}
# lgbm_params = {'n_estimators':[1000], 'learning_rate': [0.03], 'max_depth': [3], 'min_child_samples': [10], 'num_leaves': [3], 'colsample_bytree': [0.1], 'feature_fraction': [1]}
# best_rige = get_best_params(ridge_reg, ridge_params)
# best_lasso = get_best_params(lasso_reg, lasso_params)
# best_en = get_best_params(en_reg, en_params)
# best_rf = get_best_params(rf_reg, rf_params)
# best_gbm = get_best_params(gbm_reg, gbm_params)
# best_xgb = get_best_params(xgb_reg, xgb_params)
# best_lgbm = get_best_params(lgbm_reg, lgbm_params)

# 선형 회귀 모델
models_linear = [lr_reg, ridge_reg, lasso_reg, en_reg]
for model in models_linear:
    get_model_predict(model,X_train, X_test, y_train, y_test)

# 선형 회귀 모델의 회귀 계수 출력 & 시각화.
# print_coefficient(models_linear)
visualize_coefficient(models_linear)

# 회귀 트리 모델
models_tree = [dt_reg, rf_reg, gbm_reg, xgb_reg, lgbm_reg]
for model in models_tree:
    get_model_predict(model,X_train, X_test, y_train, y_test)

# 회귀 트리 모델의 회귀 계수 시각화.
visualize_ftr_importances(models_tree)