#!/usr/bin/env python3
"""Inspect model completions from evaluation parquet files."""

import argparse
from pathlib import Path

import pandas as pd


def find_parquet(base_dir: Path, model_name: str) -> Path:
    """Find the most recent parquet file for a given model."""
    patterns = [
        base_dir / "details" / "**" / model_name / "**" / "*.parquet",
        base_dir / "details" / model_name / "**" / "*.parquet",
    ]

    for pattern in patterns:
        files = list(base_dir.glob(str(pattern.relative_to(base_dir))))
        if files:
            return max(files, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(
        f"No parquet file found for model '{model_name}' in {base_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description="Inspect evaluation results")
    parser.add_argument("model", nargs="?", help="Model name (e.g., Llama-3.1-8B)")
    parser.add_argument(
        "--base-dir", default="./test_output", help="Base output directory"
    )
    parser.add_argument(
        "-n", "--num-samples", type=int, default=5, help="Number of samples to show"
    )
    parser.add_argument(
        "--parquet", help="Direct path to parquet file (skips auto-discovery)"
    )
    args = parser.parse_args()

    if args.parquet:
        parquet_path = Path(args.parquet)
    elif args.model:
        parquet_path = find_parquet(Path(args.base_dir), args.model)
    else:
        parser.error("Either 'model' or '--parquet' must be provided")

    print(f"Reading: {parquet_path}\n")
    df = pd.read_parquet(parquet_path)

    for i in range(min(args.num_samples, len(df))):
        row = df.iloc[i]
        print(f"{'=' * 80}")
        print(f"EXAMPLE {i + 1}")
        print(f"{'=' * 80}")

        print(f"\n--- PROMPT ---\n{row['full_prompt']}")

        predictions = row["predictions"]
        pred_text = predictions[0] if len(predictions) > 0 else "(empty)"
        if not pred_text:
            pred_text = "(empty string)"
        print(f"\n--- MODEL OUTPUT ---\n{pred_text}")

        gold = row["gold"]
        gold_text = gold[0] if len(gold) > 0 else "(none)"
        if len(gold_text) > 500:
            gold_text = gold_text[:500] + "..."
        print(f"\n--- GOLD ANSWER ---\n{gold_text}")

        metrics = row.get("metrics", {})
        print(f"\n--- METRICS ---\n{metrics}")
        print()


if __name__ == "__main__":
    main()
