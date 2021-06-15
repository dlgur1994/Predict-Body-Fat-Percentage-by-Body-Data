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
    height = int(request.form['height'])
    weight = float(request.form['weight'])

    data = [(age, volt, height, weight, sex_0, sex_1)]
    X_test = pd.DataFrame(data, columns = ['Age', 'Volt', 'Height', 'Weight', 'Sex_0', 'Sex_1'])
    predict_value = np.round(model.predict(X_test),1)[0]
    return render_template('predict.html', prediction_text='Predicted value is {}%'.format(predict_value))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    # host and port must be used for actual use
    # ex) app.run(debug=True, host='221.143.48.72', port=8017)