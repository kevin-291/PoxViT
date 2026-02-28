from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COMMANDS = {"train", "infer", "visualize", "balance"}


def _print_usage() -> None:
    print("Usage: python app/main.py {train|infer|visualize|balance} [args...]")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        _print_usage()
        return

    command = sys.argv[1]
    remaining = sys.argv[2:]
    if command not in COMMANDS:
        _print_usage()
        raise SystemExit(f"Unknown command: {command}")

    if command == "train":
        from scripts import train

        sys.argv = ["train.py", *remaining]
        train.main()
    elif command == "infer":
        from scripts import infer

        sys.argv = ["infer.py", *remaining]
        infer.main()
    elif command == "visualize":
        from scripts import visualize

        sys.argv = ["visualize.py", *remaining]
        visualize.main()
    else:
        from scripts import balance_dataset

        sys.argv = ["balance_dataset.py", *remaining]
        balance_dataset.main()


if __name__ == "__main__":
    main()