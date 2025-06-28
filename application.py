from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

# Loading the pickled models
ridgecv_model = pickle.load(open('Models/ridgecv.pkl', 'rb'))
standard_scaler = pickle.load(open('Models/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method =="POST":
        try:
            data = [
                float(request.form.get('Temperature')),
                float(request.form.get('RH')),
                float(request.form.get('Ws')),
                float(request.form.get('Rain')),
                float(request.form.get('FFMC')),
                float(request.form.get('DMC')),
                float(request.form.get('ISI')),
                float(request.form.get('Classes')),
                float(request.form.get('Region'))
            ]

            scaled_data = standard_scaler.transform([data])
            result = ridgecv_model.predict(scaled_data)

            return render_template('home.html', result=result[0])
        except Exception as e:
            return f"Error: {e}"
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0")
