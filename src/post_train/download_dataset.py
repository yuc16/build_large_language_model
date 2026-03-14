from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = ROOT / "data" / "post_train" / "alpaca_zh_500.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Hugging Face 下载一个简单的中文指令微调数据集子集。"
    )
    parser.add_argument(
        "--dataset-name",
        default="shibing624/alpaca-zh",
        help="Hugging Face 数据集名称。",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset_name, split=args.split)
    subset = dataset.select(range(min(args.max_samples, len(dataset))))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        for row in subset:
            record = {
                "instruction": row.get("instruction", "").strip(),
                "input": row.get("input", "").strip(),
                "output": row.get("output", "").strip(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"dataset_name={args.dataset_name}")
    print(f"split={args.split}")
    print(f"saved_samples={len(subset)}")
    print(f"output_path={args.output_path}")


if __name__ == "__main__":
    main()
