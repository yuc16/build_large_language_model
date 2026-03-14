from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = ROOT / "data" / "qwen_zh_sft" / "raw" / "alpaca_zh_5000.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="下载中文单轮 SFT 数据，并规范成 messages 格式。"
    )
    parser.add_argument("--dataset-name", default="shibing624/alpaca-zh")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def build_user_content(instruction: str, input_text: str) -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    if not input_text:
        return instruction
    return f"{instruction}\n\n补充信息：\n{input_text}"


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset_name, split=args.split, streaming=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = 0
    with args.output_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            instruction = str(row.get("instruction", "")).strip()
            input_text = str(row.get("input", "")).strip()
            output_text = str(row.get("output", "")).strip()
            if not instruction or not output_text:
                continue

            record = {
                "messages": [
                    {"role": "system", "content": "你是一个有帮助的中文助手。"},
                    {"role": "user", "content": build_user_content(instruction, input_text)},
                    {"role": "assistant", "content": output_text},
                ],
                "source_dataset": args.dataset_name,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1
            if saved >= args.max_samples:
                break

    print(f"dataset_name={args.dataset_name}")
    print(f"split={args.split}")
    print(f"saved_samples={saved}")
    print(f"output_path={args.output_path}")


if __name__ == "__main__":
    main()
