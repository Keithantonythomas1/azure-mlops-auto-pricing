import os
import mlflow
import pandas as pd
from sklearn.model_selection import GridSearchCV
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
        grid = {
            "n_estimators": [150, 200, 300],
            "max_depth": [8, 12, 16]
        }
        base = RandomForestRegressor(random_state=42, n_jobs=-1)
        gs = GridSearchCV(base, grid, scoring="neg_mean_absolute_error", cv=3, n_jobs=-1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        preds = best.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2  = r2_score(y_test, preds)
        mlflow.log_params(gs.best_params_)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        # Save best model
        os.makedirs("outputs", exist_ok=True)
        from joblib import dump
        dump(best, os.path.join("outputs", "best_model.pkl"))
        try:
            mlflow.sklearn.log_model(best, "best_rf_model")
        except Exception as e:
            print("MLflow model log skipped:", e)
        print("Best params:", gs.best_params_)
        print(f"TUNE -> MAE: {mae:.4f}  R2: {r2:.4f}")