from flask import Flask, render_template, request, url_for, flash, redirect
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=('GET', 'POST'))
def calculate():
    if request.method == 'POST':
        age = request.form.get('age')
        weight = request.form.get('weight')
        out = predictBloodPressure(float(age),float(weight))

    return render_template('index.html', bloodpressure=float(out))

def predictBloodPressure(age,weight):
    predmodel = joblib.load("regr.pkl")
    x = pd.DataFrame([[age, weight]], columns=["Age", "Weight"])
    return predmodel.predict(x)[0]





