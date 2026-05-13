import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


ANSWER_PATTERN = re.compile(r"####\s*([^\n]+)")
SYSTEM_PROMPT = (
    "You are a careful math reasoning assistant. Solve the problem step by step, "
    "and make sure the final answer is clearly stated."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/gsm8k")
    parser.add_argument("--dataset-name", default="gsm8k")
    parser.add_argument("--dataset-config", default="main")
    return parser.parse_args()


def extract_final_answer(answer_text: str) -> str:
    match = ANSWER_PATTERN.search(answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return answer_text.replace(",", "").strip()


def build_record(example: dict) -> dict:
    question = example["question"].strip()
    answer = example["answer"].strip()
    return {
        "prompt": question,
        "response": answer,
        "answer": extract_final_answer(answer),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


def write_jsonl(records: list[dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset_name, args.dataset_config)

    train_records = [build_record(example) for example in dataset["train"]]
    test_records = [build_record(example) for example in dataset["test"]]

    out_dir = Path(args.output_dir)
    write_jsonl(train_records, out_dir / "train.jsonl")
    write_jsonl(test_records, out_dir / "test.jsonl")

    print(f"Wrote {len(train_records)} train examples to {out_dir / 'train.jsonl'}.")
    print(f"Wrote {len(test_records)} test examples to {out_dir / 'test.jsonl'}.")


if __name__ == "__main__":
    main()
