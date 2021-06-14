import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# train data file load
# delete 'Index' because it is provided when converted to a data frame, and delete 'Standard_Weight' because it is determined by the hegith
file_df = pd.read_csv('./train_data.csv')
target_name = 'Body_Fat_Rate'
no_need_features = ['Index', 'Standard_Weight']
category_features = ['Sex']

# arrange X and y
file_df.drop(no_need_features, axis=1, inplace=True)
y_target = file_df[target_name]
X_features = file_df.drop([target_name],axis=1,inplace=False)

# visualize data to find outliers
outlier_name = 'Height'
cond1 = file_df[outlier_name] < 60
cond2 = file_df[target_name] < 30
outlier_index = X_features[cond1 & cond2].index
X_features.drop(outlier_index , axis=0, inplace=True)
y_target.drop(outlier_index, axis=0, inplace=True)

# change the category feature to One-Hot Encoding --> 'Sex'
X_features_ohe = pd.get_dummies(X_features, columns=category_features)

# the log transformation is applied on the target column to form a normal distribution
y_target_log = np.log1p(y_target)

# split train/test data based on feature dataset with One-Hot encoding
X_train, X_test, y_train, y_test = train_test_split(X_features_ohe, y_target_log, test_size=0.2, random_state=0)

# single model
model = rf_reg = RandomForestRegressor(max_depth=14, min_samples_leaf=2, min_samples_split=2, n_estimators=700, n_jobs=-1)
model.fit(X_train, y_train)

# mixed model
# model1 = LinearRegression()
# model2 = Ridge(alpha=1)
# model1.fit(X_train, y_train)
# model2.fit(X_train, y_train)

# test data file load
# delete 'Index' because it is provided when converted to a data frame, and delete 'Standard_Weight' because it is determined by the hegith
test_df = pd.read_csv('./test_data.csv')
# print(test_df)

# arrange X and y
test_df.drop(no_need_features, axis=1, inplace=True)
y_test = test_df[target_name]
X_test = test_df.drop([target_name],axis=1,inplace=False)

# change the category feature to One-Hot Encoding --> 'Sex'
X_test_ohe = pd.get_dummies(X_test, columns=category_features)

# single model
predict_value = model.predict(X_test_ohe)
predict_final = np.expm1(predict_value)
predict_final
print("**Single Model**")
for x in predict_final:
    print(round(x,1))
print("")

# mixed model
# pred1 = model1.predict(X_test_ohe)
# pred2 = model2.predict(X_test_ohe)
# pred = 0.8 * pred1 + 0.2 * pred2
# np.expm1(pred)
# print("**Mixed Model**")
# for x in np.expm1(pred):
#     print(round(x,1))