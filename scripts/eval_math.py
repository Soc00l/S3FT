import argparse
import json
import re
from pathlib import Path


NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    cleaned = str(text).replace(",", "").replace("$", "").strip()
    matches = NUMBER_PATTERN.findall(cleaned)
    if not matches:
        return cleaned
    value = matches[-1]
    try:
        numeric = float(value)
    except ValueError:
        return value
    if numeric.is_integer():
        return str(int(numeric))
    return format(numeric, "g")


def is_correct(prediction: str, reference: str) -> bool:
    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)
    return bool(pred_norm) and pred_norm == ref_norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--references", required=False)
    parser.add_argument("--output-file", required=False)
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def merge_references(predictions: list[dict], references: list[dict]) -> list[dict]:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    merged = []
    for pred, ref in zip(predictions, references):
        item = dict(pred)
        item["answer"] = ref.get("answer", "")
        if "response" not in item and "response" in ref:
            item["response"] = ref["response"]
        if "messages" not in item and "messages" in ref:
            item["messages"] = ref["messages"]
        merged.append(item)
    return merged


def score_records(records: list[dict]) -> tuple[list[dict], dict]:
    scored = []
    correct = 0
    for record in records:
        prediction = record.get("predicted_answer", record.get("prediction", record.get("model_answer", "")))
        reference = record.get("answer", "")
        pred_norm = normalize_answer(prediction)
        ref_norm = normalize_answer(reference)
        ok = bool(pred_norm) and pred_norm == ref_norm
        if ok:
            correct += 1

        enriched = dict(record)
        enriched["predicted_answer_normalized"] = pred_norm
        enriched["answer_normalized"] = ref_norm
        enriched["is_correct"] = ok
        enriched["judge_reason"] = "exact_numeric_match" if ok else "numeric_mismatch"
        scored.append(enriched)

    total = len(scored)
    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
    }
    return scored, metrics


def write_jsonl(records: list[dict], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    predictions = load_jsonl(args.predictions)
    records = predictions

    if args.references:
        references = load_jsonl(args.references)
        records = merge_references(predictions, references)

    scored, metrics = score_records(records)

    if args.output_file:
        write_jsonl(scored, args.output_file)
        print(f"Wrote scored outputs to {args.output_file}.")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
