import argparse
import re


NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_answer(text: str) -> str:
    text = text.replace(",", "").strip()
    match = NUMBER_PATTERN.findall(text)
    return match[-1] if match else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=False)
    parser.add_argument("--references", required=False)
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    print("Implement unified math evaluation here.")


if __name__ == "__main__":
    main()
