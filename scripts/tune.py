import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# -----------------------------
# Parse input arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Input data directory")
parser.add_argument("--output_dir", type=str, help="Output directory")
args = parser.parse_args()

# -----------------------------
# Load preprocessed data
# -----------------------------
data_path = os.path.join(args.data_dir, "prepped.csv")
df = pd.read_csv(data_path)
X = df.drop("price", axis=1)
y = df["price"]

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Set MLflow tracking for AzureML
# -----------------------------
mlflow.set_tracking_uri("azureml://")
mlflow.start_run(run_name="auto-pricing-experiment")

# -----------------------------
# Define and tune model
# -----------------------------
ridge = Ridge()
params = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
grid = GridSearchCV(ridge, param_grid=params, scoring="r2", cv=5)
grid.fit(X_train, y_train)

# -----------------------------
# Evaluate best model
# -----------------------------
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Best Alpha: {grid.best_params_['alpha']}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# -----------------------------
# Log metrics and model to MLflow
# -----------------------------
mlflow.log_param("best_alpha", grid.best_params_["alpha"])
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("r2", r2)
mlflow.sklearn.log_model(best_model, "model")

# -----------------------------
# Save best model locally for pipeline
# -----------------------------
os.makedirs(args.output_dir, exist_ok=True)
model_path = os.path.join(args.output_dir, "best_model.pkl")
import joblib
joblib.dump(best_model, model_path)

print(f"Saved best model to {model_path}")

mlflow.end_run()
