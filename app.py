import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    if sex == 1:
        sex_1 = 1
        sex_0 = 0
    else:
        sex_1 = 0
        sex_0 = 1
    volt = float(request.form['volt'])
    height = float(request.form['height'])
    weight = float(request.form['weight'])

    data_list = [(age, sex_0, sex_1, volt, height, weight)]
    X_test = pd.DataFrame(data_list, columns = ['Age', 'Sex_0', 'Sex_1', 'Volt', 'Height', 'Weight'])
    predict_value = model.predict(X_test)
    # predict_final = round(np.expm1(predict_value),1)
    predict_final = np.expm1(predict_value)
    return render_template('predict.html', prediction_text='{}'.format(predict_final))

if __name__ == "__main__":
    app.run(debug=True, port=8016)
    # 실제 사용을 위해서는 호스트와 포트를 써줘야함
    # app.run(debug=True, host='221.143.48.72', port=8017)