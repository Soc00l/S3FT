import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


ANSWER_PATTERN = re.compile(r"####\s*([^\n]+)")
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser.parse_args()


def load_jsonl(path, max_samples=None):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    match = ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).replace(",", "").strip()
    number_matches = NUMBER_PATTERN.findall(text.replace(",", ""))
    return number_matches[-1].strip() if number_matches else text.strip()


def build_prompts(records, tokenizer):
    prompts = []
    for record in records:
        prompt_messages = [
            msg for msg in record["messages"] if msg.get("role") != "assistant"
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)
    return prompts


def to_output_record(record, model_response):
    return {
        "prompt": record.get("prompt", ""),
        "response": record.get("response", ""),
        "answer": record.get("answer", ""),
        "messages": record.get("messages", []),
        "model_response": model_response,
        "predicted_answer": extract_final_answer(model_response),
    }


def write_jsonl(records, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    records = load_jsonl(args.input_file, max_samples=args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    prompts = build_prompts(records, tokenizer)

    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    outputs = llm.generate(prompts, sampling_params)

    result_records = []
    for record, output in zip(records, outputs):
        text = output.outputs[0].text.strip() if output.outputs else ""
        result_records.append(to_output_record(record, text))

    write_jsonl(result_records, args.output_file)
    print(f"Wrote {len(result_records)} generated records to {args.output_file}.")


if __name__ == "__main__":
    main()
