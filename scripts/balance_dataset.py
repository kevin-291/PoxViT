import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset: split if needed, then balance")
    parser.add_argument(
        "--data-root",
        default="sorted_dataset",
        help="Target root expected to contain train/val/test splits",
    )
    parser.add_argument(
        "--source-dataset-dir",
        default=None,
        help="Source unsplit dataset directory (class folders). Defaults to --data-root",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _print_report(report: dict) -> None:
    print(f"data_root: {report['data_root']}")
    print(f"performed_split: {report['performed_split']}")

    for split, (is_bal, counts) in report["before"].items():
        status = "balanced" if is_bal else "not balanced"
        print(f"[{split}] before balance: {status} counts={counts}")

    for split, result in report["balance"].items():
        status_before = "balanced" if result["was_balanced"] else "not balanced"
        status_after = "balanced" if result["is_balanced"] else "not balanced"
        print(f"[{split}] before: {status_before}; after: {status_after}; changed={result['changed']}")
        print(f"[{split}] class counts before: {result['before_counts']}")
        print(f"[{split}] class counts after:  {result['after_counts']}")


def main():
    args = parse_args()
    from data.prepare import ensure_split_and_balanced

    report = ensure_split_and_balanced(
        data_root=args.data_root,
        source_dataset_dir=args.source_dataset_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    _print_report(report)


if __name__ == "__main__":
    main()