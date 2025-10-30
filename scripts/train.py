import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib
import json

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Path to prepped data")
parser.add_argument("--train_dir", type=str, help="Output path for training artifacts")
args = parser.parse_args()

# --- Load prepped data ---
data_path = os.path.join(args.data_dir, "prepped.csv")
print(f"📂 Loading training data from: {data_path}")
df = pd.read_csv(data_path)

# --- Split features and labels ---
X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Evaluate ---
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
r2 = r2_score(y_test, preds)
metrics = {"rmse": rmse, "r2": r2}
print("📊 Metrics:", metrics)

# --- Save outputs ---
os.makedirs(args.train_dir, exist_ok=True)
model_path = os.path.join(args.train_dir, "model.joblib")
joblib.dump(model, model_path)
print(f"✅ Model saved to: {model_path}")

# Save metrics
with open(os.path.join(args.train_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f)
