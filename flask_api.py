"""
@author: Sahil
"""
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

df = pd.read_csv('Transformed_Dataset.csv')

scaler = RobustScaler()
X = df.drop('price',axis=1).values
Y = df['price'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 101)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = tf.keras.models.load_model("model.h5")


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    
    features = [float(x) for x in request.form.values()]
    features = np.array([features])
    
    single_house = scaler.transform(features.reshape(-1,19))
    
    predictions = float(model.predict(single_house))
    result = round(predictions, 2)
    
    return render_template('index.html', prediction_text='The Expected price of the house should be {}'.format(result)) 

    
 
if __name__ =='__main__':
    app.run(debug=True)


