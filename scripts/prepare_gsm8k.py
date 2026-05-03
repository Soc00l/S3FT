from pathlib import Path


def main() -> None:
    out_dir = Path("data/gsm8k")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Prepare GSM8K here and export train.jsonl / test.jsonl into data/gsm8k/.")


if __name__ == "__main__":
    main()
