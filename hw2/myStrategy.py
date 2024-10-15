import numpy as np
from sklearn.linear_model import LogisticRegression

import joblib
model = joblib.load("my_trained_model.pkl")

def generate_features(priceVec):
    window_size = 5
    if len(priceVec) < window_size:
        return np.zeros((1, 2)) 
    
    moving_avg = np.mean(priceVec[-window_size:])
    price_change = priceVec[-1] - priceVec[-window_size]
    
    return np.array([moving_avg, price_change]).reshape(1, -1)


def myStrategy(priceVec, currentPrice):
    features = generate_features(priceVec)
    action = model.predict(features)
    return action[0]
