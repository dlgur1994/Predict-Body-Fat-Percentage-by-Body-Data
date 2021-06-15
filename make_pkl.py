import pandas as pd
from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor
import pickle

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

# define the model
model = RandomForestRegressor(max_depth=14, min_samples_leaf=2, min_samples_split=2, n_estimators=700, n_jobs=-1)
model.fit(X_features_ohe, y_target)

# make a pkl file
# pickle.dump(model, open('model.pkl','wb'), protocol=2)
pickle.dump(model, open('model.pkl','wb'))
print('done')
