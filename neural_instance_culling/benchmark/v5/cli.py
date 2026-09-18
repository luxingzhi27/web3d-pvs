"""Command-line entry point for the V5 evaluation and result matrix."""
from __future__ import annotations

import argparse
from pathlib import Path

from .contracts import V5Run
from .evaluation import evaluate_loso_matrix, evaluate_shared_matrix
from .summary import summarize_loso, summarize_shared, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("shared", "loso"), required=True)
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--evaluation-bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [V5Run.from_json(path) for path in args.bundle]
    common = {
        "evaluation_split": args.split,
        "calibration_bootstrap_replicates": args.calibration_bootstrap_replicates,
        "evaluation_bootstrap_replicates": args.evaluation_bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
    }
    if args.protocol == "shared":
        rows = evaluate_shared_matrix(runs, **common)
        payload = summarize_shared(rows)
    else:
        rows = evaluate_loso_matrix(runs, **common)
        payload = summarize_loso(rows)
    write_summary(args.output, payload)
    print(
        {
            "output": str(args.output.resolve()),
            "protocol": args.protocol,
            "rows": len(rows),
            "split": args.split,
            "selectionSplit": "calibration",
            "testReadForSelection": False,
        }
    )


if __name__ == "__main__":
    main()
