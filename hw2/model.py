import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

def generate_features(priceVec, rsi_period=14):
    rsi = calculate_rsi(priceVec, period=rsi_period)  # Use RSI as the feature
    return np.array([rsi]).reshape(1, -1)

# Load historical price data
df = pd.read_csv("price3000open.csv")  # Replace with actual path
priceVec = df["Adj Close"].values

# Create features and labels for training with different RSI periods
rsi_periods = [7, 14, 21, 28]
results = {}

for period in rsi_periods:
    features = []
    labels = []

    # Generate features and labels
    for i in range(len(priceVec) - 5):
        features.append(generate_features(priceVec[:i + 5], rsi_period=period))
        future_price = priceVec[i + 5]
        current_price = priceVec[i + 4]
        if future_price > current_price:
            labels.append(1)  # Buy
        elif future_price < current_price:
            labels.append(-1)  # Sell
        else:
            labels.append(0)  # Hold

    # Prepare training data
    X = np.array(features).reshape(-1, 1)  # Adjust shape for 1 feature (RSI)
    y = np.array(labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    results[period] = accuracy
    # Save the model for future use
    joblib.dump(model, f"trained_model_rsi_period_{period}.pkl")

# Display results
for period, accuracy in results.items():
    print(f"RSI Period: {period}, Accuracy: {accuracy:.2f}")
