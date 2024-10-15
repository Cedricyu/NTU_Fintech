from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

def generate_features(priceVec):
    # Example: Using simple moving average as a feature
    window_size = 5
    if len(priceVec) < window_size:
        return np.zeros((1,))  # Return zeros if not enough data
    # Calculate moving average
    moving_avg = np.mean(priceVec[-window_size:])
    price_change = priceVec[-1] - priceVec[-window_size]
    # Combine features
    return np.array([moving_avg, price_change]).reshape(1, -1)

# Load historical data
df = pd.read_csv("price3000open.csv")  # Replace with actual path
priceVec = df["Adj Close"].values

# Create features and labels for training
features = []
labels = []

# Example: Label based on simple strategy (e.g., if price rises in next 5 days, buy)
for i in range(len(priceVec) - 5):
    features.append(generate_features(priceVec[:i+5]))
    future_price = priceVec[i+5]
    current_price = priceVec[i+4]
    if future_price > current_price:
        labels.append(1)  # Buy
    elif future_price < current_price:
        labels.append(-1)  # Sell
    else:
        labels.append(0)  # Hold

# Train the model
X = np.array(features).reshape(-1, 2)  # Adjust shape according to generate_features output
y = np.array(labels)
model = LogisticRegression()
model.fit(X, y)

# Save the model using joblib or pickle for future use
import joblib
joblib.dump(model, "my_trained_model.pkl")
