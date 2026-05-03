import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-input", required=True)
    parser.add_argument("--s3ft-output", required=True)
    parser.add_argument("--self-answer-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    with open(args.scored_input, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    Path(args.s3ft_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.self_answer_output).parent.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(records)} scored examples from {args.scored_input}.")
    print("Implement selective rewrite logic here.")


if __name__ == "__main__":
    main()
