import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd

from src.pipeline.pred_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method=='GET':
        return render_template('index.html')
    else:
        data = CustomData(
            Age = request.form.get('Age'),

            Employment_Type = request.form.get('Employment_Type'),

            GraduateOrNot = request.form.get('GraduateOrNot'),

            AnnualIncome = request.form.get('AnnualIncome'),

            FamilyMembers = request.form.get('FamilyMembers'),

            ChronicDiseases = request.form.get('ChronicDiseases'),

            FrequentFlyer = request.form.get('FrequentFlyer'),

            EverTravelledAbroad = request.form.get('EverTravelledAbroad')
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print(results)


        return render_template('index.html', prediction_text='The chances of buying Travel insurance is {}%'.format(results[0]))


if __name__ == "__main__":
    app.run(host = '0.0.0.0',debug=True)
