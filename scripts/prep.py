import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    return df

def clean_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Basic cleaning
    df = df.copy()
    # Drop missing target rows
    df = df.dropna(subset=["Price"])
    # Handle missing numerics
    numeric_cols = ["Kilometers_Driven", "Mileage", "Engine", "Power", "Seats"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    # Categorical handling
    if "Segment" in df.columns:
        df["Segment"] = df["Segment"].astype(str).str.lower().str.strip()
        df["Segment"] = df["Segment"].map({"luxury":1, "non-luxury":0}).fillna(0).astype(int)
    return df

def split_save(df: pd.DataFrame, outdir: str = "outputs"):
    os.makedirs(outdir, exist_ok=True)
    X = df.drop(columns=["Price"])
    y = df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train.to_csv(os.path.join(outdir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(outdir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(outdir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(outdir, "y_test.csv"), index=False)

if __name__ == "__main__":
    data_path = os.environ.get("DATA_PATH", "data/vehicles.csv")
    df = load_data(data_path)
    df = clean_engineering(df)
    split_save(df, outdir=os.environ.get("OUTPUT_DIR", "outputs"))