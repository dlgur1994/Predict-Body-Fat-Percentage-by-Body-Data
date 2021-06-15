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
- distribution<br/> 
> |Data|Numbers|Ratio|
> |---|---|---|
> |train|684|0.8|
> |test|169|0.2|
> |total|853|1|<br/>
- example<br/>
> |Index|Age|Sex|Volt|Height|Weight|Standard_Weight|Body_Fat_Rate|
> |---|---|---|---|---|---|---|---|
> |0|23|0|1.35|167|62.8|60.3|31.9
> |1|20|1|1.15|183|75.1|74.7|12.6

## 5. Results
- 'Random Forest' shows the best performance.<br/>
> |Model|MAE|MSE|RMSE|RMSLE|R2|
> |---|---|---|---|---|---|
> |Linear|2.902|13.920|3.731|0.158|0.781|
> |Ridge|2.889|13.771|3.711|0.159|0.783|
> |Lasso|3.818|23.587|4.857|0.212|0.629|
> |Elastic Net|3.754|22.904|4.786|0.209|0.640|
> |Decision Tree|2.272|10.251|3.202|0.135|0.839|
> |**Random Forest**|**2.034**|**8.205**|**2.864**|**0.116**|**0.871**|
> |Gradient Boosting|2.341|9.700|3.114|0.134|0.847|
> |XGB|2.329|9.581|3.095|0.130|0.849|
> |LGM|2.578|11.477|3.388|0.141|0.820|

## 6. Run on the Web
- download files
    - css/style.css 
    - templates/index.html
    - templates/predict.html
- make the 'model.pkl' file
    - python3 make_pkl.py
- run the server
    - python3 app.py
- access to the web
    - localhost:5000