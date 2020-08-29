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

model = tf.keras.models.load_model("model.h5")
scaler = RobustScaler()
X = df.drop('price',axis=1).values
Y = df['price'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 101)


@app.route("/")
def welcome():
    return "Welcome!"

@app.route("/predict_file", methods=["POST"])
def predict_house_price():
    # bedrooms = request.args.get('bedroom')
    # bathrooms = request.args.get('bathroom')
    # sqft_living = request.args.get('sqft_living')
    # sqft_lot = request.args.get('sqft_lot')
    # floors = request.args.get('floors')
    # waterfront = request.args.get('waterfront')
    # view = request.args.get('view')
    # condition = request.args.get('condition')
    # grade = request.args.get('grade')
    # sqft_above = request.args.get('grade')
    # sqft_basement = request.args.get('sqft_basement')
    # yr_built = request.args.get('yr_built')
    # yr_renovated = request.args.get('yr_renovated')
    # lat = request.args.get('lat')
    # long = request.args.get('long')
    # sqft_living15 = request.args.get('sqft_living15')
    # sqft_lot15 = request.args.get('sqft_lot15')
    # year_sold = request.args.get('year_sold')
    # month_sold = request.args.get('month_sold')
    
    # predictions = model.predict([[bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,grade,sqft_above,sqft_basement,yr_built,yr_renovated,lat,long,sqft_living15,sqft_lot15,year_sold,month_sold]])
      
    df_test = pd.read_csv(request.files.get('files'))
    predictions = model.predict(df_test)
    return 'The predicted values is '+ str(predictions) 
 
if __name__ =='__main__':
    app.run()


