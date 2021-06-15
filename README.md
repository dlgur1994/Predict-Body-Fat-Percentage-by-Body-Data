## 1. About
It is a program that uses simple body data to predict body fat rates, and uses machine learning techniques.

## 2. Environment
- Mac OS
- Python 3.8.3
- Conda 4.8.3

## 3. Installation
- xgboost<br/>
    !pip install xgboost<br/>
    !brew install libomp
- lightgbm<br/>
    !pip install lightgbm

## 4. Data
- format<br/>
    .csv
- distribution
|Data|Numbers|Ratio|
|---|---|---|
|train|684|0.8|
|test|169|0.2|
|total|853|1|
- example
|Index|Age|Sex|Volt|Height|Weight|Standard_Weight|Body_Fat_Rate|
|---|---|---|---|---|---|---|---|
|0|23|0|1.35|167|62.8|60.3|31.9
|1|20|1|1.15|183|75.1|74.7|12.6

## 5. Results
- Random Forest
- 선정 이유