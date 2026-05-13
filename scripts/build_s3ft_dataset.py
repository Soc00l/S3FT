import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-input", required=True)
    parser.add_argument("--s3ft-output", required=True)
    parser.add_argument("--self-answer-output", required=True)
    parser.add_argument("--stats-output", required=False)
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_messages(system_text: str, prompt: str, assistant_text: str) -> list[dict]:
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_text},
    ]


def rewrite_record(record: dict, use_self_answer: bool) -> dict:
    prompt = record.get("prompt")
    response = record.get("response", "")
    messages = record.get("messages", [])
    if not prompt and messages:
        user_messages = [m for m in messages if m.get("role") == "user"]
        prompt = user_messages[-1]["content"] if user_messages else ""
    system_text = ""
    if messages and messages[0].get("role") == "system":
        system_text = messages[0].get("content", "")

    chosen_response = record.get("model_response", response) if use_self_answer else response
    rewritten = dict(record)
    rewritten["response"] = chosen_response
    rewritten["messages"] = build_messages(system_text, prompt or "", chosen_response)
    rewritten["data_source"] = "self_answer" if use_self_answer else "gold"
    return rewritten


def write_jsonl(records: list[dict], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.scored_input)

    s3ft_records = []
    self_answer_records = []
    replaced = 0

    for record in records:
        use_self_answer = bool(record.get("is_correct", False)) and bool(record.get("model_response", "").strip())
        s3ft_records.append(rewrite_record(record, use_self_answer=use_self_answer))
        if use_self_answer:
            self_answer_records.append(rewrite_record(record, use_self_answer=True))
            replaced += 1

    write_jsonl(s3ft_records, args.s3ft_output)
    write_jsonl(self_answer_records, args.self_answer_output)

    stats = {
        "total": len(records),
        "replaced_with_self_answer": replaced,
        "replacement_rate": (replaced / len(records)) if records else 0.0,
        "self_answer_only_size": len(self_answer_records),
    }

    if args.stats_output:
        stats_path = Path(args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
