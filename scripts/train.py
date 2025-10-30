import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import json
import os

# Load prepped data
df = pd.read_csv(os.path.join(args.data_dir, "prepped.csv"))
X = df.drop("price", axis=1)
y = df["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Save metrics
metrics = {"rmse": float(rmse), "r2": float(r2)}
os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f)

print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")
