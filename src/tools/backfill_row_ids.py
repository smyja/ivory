import os
import json
import argparse
import hashlib
from typing import List

import pandas as pd


def compute_row_id(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    def _row_hash(row: pd.Series) -> str:
        parts = []
        for col in columns:
            val = row.get(col, None)
            try:
                parts.append(json.dumps(val, sort_keys=True, ensure_ascii=False, default=str))
            except Exception:
                parts.append(str(val))
        base = "\u241F".join(parts)
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    return df.apply(_row_hash, axis=1)


def backfill_dataset(dataset_dir: str) -> int:
    parquet_path = os.path.join(dataset_dir, "data.parquet")
    if not os.path.exists(parquet_path):
        return 0

    df = pd.read_parquet(parquet_path)
    if "__row_id" in df.columns:
        return 0

    cols = sorted([c for c in df.columns if c != "__row_id"])
    df["__row_id"] = compute_row_id(df, cols)
    df.to_parquet(parquet_path, index=False)

    # Update schema.json if present
    schema_path = os.path.join(dataset_dir, "schema.json")
    if os.path.exists(schema_path):
        try:
            with open(schema_path) as f:
                schema = json.load(f)
        except Exception:
            schema = {"columns": list(df.columns), "column_types": {}, "num_rows": len(df)}
        schema["columns"] = list(df.columns)
        types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        schema["column_types"] = types
        schema["num_rows"] = len(df)
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)

    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill __row_id into dataset Parquet files")
    parser.add_argument("--root", default="datasets", help="Root datasets directory")
    args = parser.parse_args()

    changed = 0
    for root, dirs, files in os.walk(args.root):
        if "data.parquet" in files:
            n = backfill_dataset(root)
            if n:
                print(f"Updated {root} with __row_id for {n} rows")
                changed += 1
    if changed == 0:
        print("No updates needed. All Parquet files already have __row_id.")


if __name__ == "__main__":
    main()

