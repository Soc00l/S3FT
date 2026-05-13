import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ANSWER_PATTERN = re.compile(r"####\s*([^\n]+)")
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--use-4bit", action="store_true")
    return parser.parse_args()


def load_jsonl(path: str, max_samples: int | None = None) -> list[dict]:
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


def load_model_and_tokenizer(model_path: str, use_4bit: bool):
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def build_prompt_text(messages: list[dict], tokenizer: AutoTokenizer) -> str:
    prompt_messages = [message for message in messages if message.get("role") != "assistant"]
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_response(
    prompt_text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def to_output_record(record: dict, model_response: str) -> dict:
    return {
        "prompt": record.get("prompt", ""),
        "response": record.get("response", ""),
        "answer": record.get("answer", ""),
        "messages": record.get("messages", []),
        "model_response": model_response,
        "predicted_answer": extract_final_answer(model_response),
    }


def write_jsonl(records: list[dict], output_file: str) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input_file, max_samples=args.max_samples)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.use_4bit)

    outputs = []
    for idx, record in enumerate(records, start=1):
        prompt_text = build_prompt_text(record["messages"], tokenizer)
        model_response = generate_response(
            prompt_text=prompt_text,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        outputs.append(to_output_record(record, model_response))

        if idx % 50 == 0:
            print(f"Generated {idx} / {len(records)} samples.")

    write_jsonl(outputs, args.output_file)
    print(f"Wrote {len(outputs)} generated records to {args.output_file}.")


if __name__ == "__main__":
    main()
