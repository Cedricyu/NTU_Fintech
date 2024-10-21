import numpy as np
import pandas as pd
import joblib

def calculate_rsi(priceVec, period=14):
    if len(priceVec) < period:
        return 0  # Return 0 if not enough data to calculate RSI

    gains = np.diff(priceVec)
    ups = np.where(gains > 0, gains, 0)
    downs = np.where(gains < 0, -gains, 0)

    avg_gain = np.mean(ups[-period:])
    avg_loss = np.mean(downs[-period:])
    
    if avg_loss == 0:
        return 100  # If no losses, RSI is 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def generate_features(priceVec):
    rsi = calculate_rsi(priceVec)  # Use only RSI as the feature
    return np.array([rsi]).reshape(1, -1)

def load_model(model_path):
    return joblib.load(model_path)

model = load_model('my_trained_model.pkl')

def myStrategy(priceVec, currentPrice):
    # Generate features including RSI
    features = generate_features(priceVec)
    rsi = features[0][0]  # Extract the RSI value from the features
    
    # Prepare the feature array for prediction
    feature_array = np.array([rsi]).reshape(1, -1)

    # Make prediction using the model
    predicted_action = model.predict(feature_array)[0]

    # Adjust the action based on the current price and moving average
    window_size = 5
    moving_avg = np.mean(priceVec[-window_size:]) if len(priceVec) >= window_size else currentPrice
    
    if predicted_action == 1 and currentPrice < moving_avg:
        action = 1  # Buy
    elif predicted_action == -1 and currentPrice > moving_avg:
        action = -1  # Sell
    else:
        action = 0  # Hold

    return action