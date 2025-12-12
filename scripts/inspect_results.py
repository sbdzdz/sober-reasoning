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


def get_score(row) -> float:
    """Extract score from metrics."""
    metrics = row.get("metrics", {})
    if isinstance(metrics, dict):
        return metrics.get("extractive_match", 0.0)
    return 0.0


def get_question(row) -> str:
    """Extract the question from the example field."""
    example = row.get("example", "")
    if isinstance(example, str):
        return example
    return str(example)


def print_example(row, idx: int, label: str):
    """Print a single example."""
    print(f"{'=' * 80}")
    print(f"{label} EXAMPLE {idx}")
    print(f"{'=' * 80}")

    question = get_question(row)
    if len(question) > 1000:
        question = question[:1000] + "..."
    print(f"\n--- QUESTION ---\n{question}")

    predictions = row["predictions"]
    pred_text = predictions[0] if len(predictions) > 0 else "(empty)"
    if not pred_text:
        pred_text = "(empty string)"
    if len(pred_text) > 2000:
        pred_text = pred_text[:2000] + "..."
    print(f"\n--- MODEL ANSWER ---\n{pred_text}")

    score = get_score(row)
    status = "✓ CORRECT" if score > 0.5 else "✗ INCORRECT"
    print(f"\n--- SCORE: {status} ({score}) ---")
    print()


def main():
    parser = argparse.ArgumentParser(description="Inspect evaluation results")
    parser.add_argument("model", nargs="?", help="Model name (e.g., Llama-3.1-8B)")
    parser.add_argument(
        "--base-dir", default="./test_output", help="Base output directory"
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=3,
        help="Number of correct/incorrect samples to show each",
    )
    parser.add_argument(
        "--parquet", help="Direct path to parquet file (skips auto-discovery)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all samples (not split by correct/incorrect)",
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

    df["_score"] = df.apply(get_score, axis=1)
    correct_df = df[df["_score"] > 0.5]
    incorrect_df = df[df["_score"] <= 0.5]

    total = len(df)
    n_correct = len(correct_df)
    n_incorrect = len(incorrect_df)
    accuracy = n_correct / total * 100 if total > 0 else 0

    print(
        f"Total: {total} | Correct: {n_correct} | Incorrect: {n_incorrect} | Accuracy: {accuracy:.1f}%\n"
    )

    if args.all:
        for i in range(min(args.num_samples, len(df))):
            row = df.iloc[i]
            score = get_score(row)
            label = "✓" if score > 0.5 else "✗"
            print_example(row, i + 1, label)
    else:
        print(f"\n{'#' * 80}")
        print(
            f"# CORRECT EXAMPLES (showing {min(args.num_samples, len(correct_df))} of {n_correct})"
        )
        print(f"{'#' * 80}\n")
        for i in range(min(args.num_samples, len(correct_df))):
            print_example(correct_df.iloc[i], i + 1, "✓")

        print(f"\n{'#' * 80}")
        print(
            f"# INCORRECT EXAMPLES (showing {min(args.num_samples, len(incorrect_df))} of {n_incorrect})"
        )
        print(f"{'#' * 80}\n")
        for i in range(min(args.num_samples, len(incorrect_df))):
            print_example(incorrect_df.iloc[i], i + 1, "✗")


if __name__ == "__main__":
    main()
