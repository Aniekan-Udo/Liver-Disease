from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Age=request.form.get('Age'),
            Gender=request.form.get('Gender'),
            BMI=request.form.get('BMI'),
            AlcoholConsumption=request.form.get('AlcoholConsumption'),
            Smoking=request.form.get('Smoking'),
            GeneticRisk=float(request.form.get('GeneticRisk')),
            PhysicalActivity=float(request.form.get('PhysicalActivity')),
            Diabetes=float(request.form.get('Diabetes')),
            Hypertension=float(request.form.get('Hypertension')),
            LiverFunctionTest=float(request.form.get('LiverFunctionTest'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)     