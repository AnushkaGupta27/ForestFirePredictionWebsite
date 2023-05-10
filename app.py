from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
ridge_mo=pickle.load(open("models/ridge_new.pkl",'rb'))
scaler_mo=pickle.load(open("models/scaler_new.pkl",'rb'))
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictval',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        scaled_data=scaler_mo.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_mo.predict(scaled_data)
        return render_template('home.html',result=result)
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")