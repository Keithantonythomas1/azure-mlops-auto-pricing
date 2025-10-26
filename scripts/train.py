import os
import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def load_splits(indir="outputs"):
    X_train = pd.read_csv(os.path.join(indir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(indir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(indir, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(indir, "y_test.csv")).squeeze()
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    mlflow.set_experiment("auto-pricing-experiment")
    with mlflow.start_run():
        X_train, X_test, y_train, y_test = load_splits(os.environ.get("INPUT_DIR", "outputs"))
        params = {
            "n_estimators": int(os.environ.get("N_ESTIMATORS", 200)),
            "max_depth": int(os.environ.get("MAX_DEPTH", 12)),
            "random_state": 42,
            "n_jobs": -1
        }
        mlflow.log_params(params)
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2  = r2_score(y_test, preds)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        os.makedirs("outputs", exist_ok=True)
        model_path = os.path.join("outputs", "model.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")
        # Log model in MLflow format (optional, if running inside AML tracking server)
        try:
            mlflow.sklearn.log_model(model, "rf_model")
        except Exception as e:
            print("MLflow model log skipped:", e)
        print(f"MAE: {mae:.4f}  R2: {r2:.4f}")