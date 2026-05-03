import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    print(f"Generate self-answers from {args.model_path} using {args.input_file}.")
    print(f"Write structured outputs to {args.output_file}.")


if __name__ == "__main__":
    main()
