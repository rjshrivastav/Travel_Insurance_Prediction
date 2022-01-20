import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('finalized_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict_proba(final_features)[:,1]

    output = prediction
    pred = [i * 100 for i in output]

    return render_template('index.html', prediction_text='The chances of buying Travel insurance is {}%'.format(np.round(pred,2)))


if __name__ == "__main__":
    app.run(debug=True)
