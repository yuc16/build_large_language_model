from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = ROOT / "data" / "qwen_zh_short_sft" / "raw" / "short_sft_mix_6000.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="下载中文短答案 SFT 数据，并转成 messages 格式。"
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-output-chars", type=int, default=24)
    parser.add_argument("--target-samples", type=int, default=6000)
    return parser.parse_args()


def build_user_content(instruction: str, input_text: str) -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    if not input_text:
        return instruction
    return f"{instruction}\n\n补充信息：\n{input_text}"


def keep_example(instruction: str, input_text: str, output_text: str, max_output_chars: int) -> bool:
    if not instruction or not output_text:
        return False
    if len(output_text) > max_output_chars:
        return False
    if len(instruction) > 160:
        return False
    if len(input_text) > 160:
        return False
    if output_text.count("\n") > 1:
        return False
    return True


def iter_rows(dataset_name: str):
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    for row in dataset:
        yield row


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    sources = [
        ("BelleGroup/train_0.5M_CN", args.target_samples * 2 // 3),
        ("shibing624/alpaca-zh", args.target_samples // 3),
    ]

    saved = 0
    with args.output_path.open("w", encoding="utf-8") as f:
        for dataset_name, target_count in sources:
            local_saved = 0
            for row in iter_rows(dataset_name):
                instruction = str(row.get("instruction", "")).strip()
                input_text = str(row.get("input", "")).strip()
                output_text = str(row.get("output", "")).strip()
                if not keep_example(instruction, input_text, output_text, args.max_output_chars):
                    continue

                record = {
                    "messages": [
                        {"role": "system", "content": "你是一个有帮助的中文助手。除非用户要求详细解释，否则优先简短回答。"},
                        {"role": "user", "content": build_user_content(instruction, input_text)},
                        {"role": "assistant", "content": output_text},
                    ],
                    "source_dataset": dataset_name,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved += 1
                local_saved += 1
                if local_saved >= target_count or saved >= args.target_samples:
                    break
            if saved >= args.target_samples:
                break

    print(f"saved_samples={saved}")
    print(f"output_path={args.output_path}")


if __name__ == "__main__":
    main()
