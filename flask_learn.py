"""
@author: Sahil
"""
from flask import Flask,request
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import flasgger
from flasgger import Swagger

app = Flask(__name__)
# Swagger(app)

df = pd.read_csv('Transformed_Dataset.csv')
df.head(5)

scaler = RobustScaler()
X = df.drop('price',axis=1).values
Y = df['price'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 101)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = tf.keras.models.load_model("model.h5")


@app.route("/")
def welcome():
    return "Welcome All"

@app.route("/Sample")
def predict():
    
    single_house = df.drop('price',axis=1).iloc[0]
    single_house = scaler.transform(single_house.values.reshape(-1,19))
    
    x = str(float((model.predict(single_house))))
    return "The predicted value is "+x + str(single_house)



@app.route("/Input The Values")
def predict_1():
    """Let's Predict the price of the house.
    This is using docstrings for specifications.
    ---
    parameters:
      - name: Bedrooms
        in: query
        type: float
        required: true
    
      - name: Bathrooms
        in: query
        type: float
        required: true
        
      - name: sqft_living
        in: query
        type: float
        required: true
        
      - name: sqft_lot
        in: query
        type: float
        required: true
        
      - name: floors
        in: query
        type: float
        required: true
        
      - name: waterfront
        in: query
        type: float
        required: true
        
      - name: view
        in: query
        type: float
        required: true
        
      - name: Condition
        in: query
        type: float
        required: true
        
      - name: Grade
        in: query
        type: float
        required: true
        
      - name: sqft_above
        in: query
        type: float
        required: true
        
      - name: sqft_basement
        in: query
        type: float
        required: true
        
      - name: yr_built
        in: query
        type: float
        required: true
        
      - name: yr_renovated
        in: query
        type: float
        required: true
        
      - name: latitue
        in: query
        type: float
        required: true
        
      - name: longitue
        in: query
        type: float
        required: true
        
      - name: sqft_living15
        in: query
        type: float
        required: true
        
      - name: sqft_lot15
        in: query
        type: float
        required: true
        
      - name: Year_sold
        in: query
        type: float
        required: true
        
      - name: Month_sold
        in: query
        type: float
        required: true
    responses:
        1:
            description: The expected value is: 
        
    
    """
    
    bedrooms = request.args.get('bedroom')
    bathrooms = request.args.get('bathroom')
    sqft_living = request.args.get('sqft_living')
    sqft_lot = request.args.get('sqft_lot')
    floors = request.args.get('floors')
    waterfront = request.args.get('waterfront')
    view = request.args.get('view')
    condition = request.args.get('condition')
    grade = request.args.get('grade')
    sqft_above = request.args.get('sqft_above')
    sqft_basement = request.args.get('sqft_basement')
    yr_built = request.args.get('yr_built')
    yr_renovated = request.args.get('yr_renovated')
    lat = request.args.get('lat')
    long = request.args.get('long')
    sqft_living15 = request.args.get('sqft_living15')
    sqft_lot15 = request.args.get('sqft_lot15')
    year_sold = request.args.get('year_sold')
    month_sold = request.args.get('month_sold')
    
    test_case = np.array([bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,grade,sqft_above,sqft_basement,yr_built,yr_renovated,lat,long,sqft_living15,sqft_lot15,year_sold,month_sold])
    test_case = scaler.transform(test_case.reshape(-1,19))
    
    y = str(float((model.predict(test_case))))
    return "The predicted value is "+y 
 
if __name__ =='__main__':
    app.run()


