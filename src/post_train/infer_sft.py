from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "data" / "pre_train" / "Qwen2.5-0.5B"
DEFAULT_ADAPTER_DIR = ROOT / "data" / "post_train" / "qwen2.5-0.5b-alpaca-zh-lora"


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="加载 base model + LoRA adapter 做推理。"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--prompt", default="法国的首都在哪里")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    dtype = pick_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=dtype).to(device)  # type: ignore
    model = PeftModel.from_pretrained(base_model, args.adapter_dir).to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    do_sample = args.temperature > 0

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
        )

    print("mode=sft_lora_inference")
    print(f"model_dir={args.model_dir}")
    print(f"adapter_dir={args.adapter_dir}")
    print(f"device={device}")
    print(f"temperature={args.temperature}")
    print("generated_text:")
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
