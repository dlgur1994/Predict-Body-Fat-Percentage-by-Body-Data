## 1. About
- It is a program that predicts body fat rate with simple body data using machine learning.

## 2. Environment
- Mac OS 10.14.6
- Python 3.8.3
- Conda 4.8.3

## 3. Installation
- xgboost<br/>
    - !pip install xgboost<br/>
    - !brew install libomp
- lightgbm<br/>
    - !pip install lightgbm

## 4. Data
- format: .csv<br/>
- distribution

> |Data|Numbers|Ratio|
> |---|---|---|
> |train|684|0.8|
> |test|169|0.2|
> |total|853|1|

- example

> |Index|Age|Sex|Volt|Height|Weight|Standard_Weight|Body_Fat_Rate|
> |---|---|---|---|---|---|---|---|
> |0|23|0|1.35|167|62.8|60.3|31.9
> |1|20|1|1.15|183|75.1|74.7|12.6

## 5. Results
- 'Random Forest' shows the best performance.

|Model|MAE|MSE|RMSE|RMSLE|R2|
|---|---|---|---|---|---|
|Linear|2.902|3.731|13.920|2.902|0.158|
|Ridge|2.889|3.711|13.771|2.889|0.159|
|Lasso|3.818|4.857|23.587|3.818|0.212|
|Elastic Net|3.754|4.786|22.904|3.754|0.209|
|Decision Tree|2.350|3.346|11.195|2.350|0.141|
|**Random Forest**|**2.044**|**2.876**|**8.273**|**2.044**|**0.117**|
|Gradient Boosting|2.345|3.108|9.662|2.345|.0136|
|XGB|2.329|3.095|9.581|2.329|0.130|
|LGM|2.578|3.388|11.477|2.578|0.141|