import joblib
import numpy as np
from flask import Flask,request,render_template

app = Flask(__name__)
sentiment_vec = joblib.load('Sentiment_vector.pkl')
sentiment_model = joblib.load('Sentiment_Analysis_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    output = ' '
    int_features = [x for x in request.form.values()]
    
    input_vec = sentiment_vec.transform(int_features)
    output = sentiment_model.predict(input_vec.toarray())
    print(output)

    if output == 0:
         prediction = 'Negative'
    else:
         prediction = 'Positive'   

    return render_template('index.html',prediction_text='The Sentiment is :{}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)