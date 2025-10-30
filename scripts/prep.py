#!/usr/bin/env python3
"""
Data Prep step for Auto Pricing pipeline.

- Works in Azure ML and locally
- Input CSV: from --data arg OR env AZUREML_DATAREFERENCE_data_file OR 'data/vehicles.csv'
- Output dir: --out (required in AML; defaults to './prep_out' locally)
- Produces:
    - <out>/prepped.csv
    - <out>/prep_metrics.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def log(msg: str):
    print(f"[prep] {msg}", flush=True)


def resolve_input_path(cli_path: str | None) -> str:
    """
    Resolution order:
      1) --data argument (if provided)
      2) AZUREML_DATAREFERENCE_data_file (set by Azure ML for input datasets)
      3) local fallback: 'data/vehicles.csv'
    """
    if cli_path and str(cli_path).strip():
        return cli_path
    env_path = os.environ.get("AZUREML_DATAREFERENCE_data_file")
    if env_path and Path(env_path).exists():
        return env_path
    # Fallback for local dev
    return "data/vehicles.csv"


def safe_read_csv(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    # robust CSV read (handles BOM and common encodings)
    try:
        return pd.read_csv(path, engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", engine="python")


def light_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, schema-agnostic cleanup.
    - Drop exact duplicates
    - Drop rows fully empty
    - If numeric target-ish columns exist (e.g., price), drop rows where they’re missing or nonpositive
    """
    before = len(df)
    df = df.drop_duplicates().dropna(how="all").copy()

    # Optional heuristics if columns are present
    for col in ["price", "Price", "sale_price", "SalePrice"]:
        if col in df.columns:
            # coerce to numeric
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[(df[col].notna()) & (df[col] > 0)]
            break

    after = len(df)
    log(f"Rows before: {before}, after cleanup: {after} (removed {before - after})")
    return df


def write_outputs(df: pd.DataFrame, out_dir: Path, src_path: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    prepped_csv = out_dir / "prepped.csv"
    metrics_json = out_dir / "prep_metrics.json"

    df.to_csv(prepped_csv, index=False)

    metrics = {
        "source_path": src_path,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns": list(map(str, df.columns[:50])),  # cap to avoid huge logs
    }
    metrics_json.write_text(json.dumps(metrics, indent=2))
    log(f"Wrote: {prepped_csv}")
    log(f"Wrote: {metrics_json}")


def parse_args():
    p = argparse.ArgumentParser(description="Auto Pricing — Data Prep")
    p.add_argument("--data", type=str, default=None, help="Path to input CSV (optional in AML)")
    p.add_argument("--out", type=str, default=None, help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()

    data_path = resolve_input_path(args.data)
    out_dir = Path(args.out or "./prep_out")

    log(f"Resolved input CSV: {data_path}")
    log(f"Output directory:   {out_dir}")

    try:
        df = safe_read_csv(data_path)
    except Exception as e:
        log(f"ERROR reading CSV: {e}")
        # In AML this makes the step fail clearly
        sys.exit(1)

    log(f"Loaded shape: {df.shape}")
    df = light_cleanup(df)
    write_outputs(df, out_dir, data_path)

    log("Prep completed successfully ✅")


if __name__ == "__main__":
    main()
