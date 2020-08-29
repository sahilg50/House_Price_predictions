"""
@author: Sahil
"""
from flask import Flask,request
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

df = pd.read_csv('Transformed_Dataset.csv')
df.head(5)

scaler = RobustScaler()
X = df.drop('price',axis=1).values
Y = df['price'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 101)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = tf.keras.models.load_model("model.h5")

single_house = df.drop('price',axis=1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1,19))


@app.route("/")
def welcome():
    x = model.predict(single_house)
    y = str(x)
    return y

# @app.route("/predict_file", methods=["POST"])
# def predict_house_price():

      
#     df_test = pd.read_csv(request.files.get('files'))
#     predictions = model.predict(df_test)
#     return 'The predicted values is '+ str(predictions) 
 
if __name__ =='__main__':
    app.run()


