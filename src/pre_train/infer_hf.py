from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from modeling import ROOT, pick_device


DEFAULT_HF_MODEL_DIR = ROOT / "data" / "pre_train" / "Qwen2.5-0.5B"


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="加载 Hugging Face 预训练权重并做文本生成。"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_HF_MODEL_DIR)
    parser.add_argument(
        "--prompt",
        default="你是一个有帮助的助手。\n\n### 指令：介绍一下比亚迪\n\n### 回答：",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    torch_dtype = pick_dtype(device)

    # 这里加载的是别人已经预训练好的模型权重，不会再做训练。
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch_dtype,
    ).to(
        device  # type: ignore
    )  # type: ignore
    model.eval()

    # Qwen 类模型通常可以直接用普通 prompt；如果没有 pad token，就回退到 eos token。
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    generation_config = model.generation_config
    generation_config.do_sample = args.temperature > 0
    generation_config.temperature = args.temperature if args.temperature > 0 else None
    generation_config.top_p = None
    generation_config.top_k = None
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.pad_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            generation_config=generation_config,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("mode=hf_pretrained_inference")
    print(f"device={device}")
    print(f"model_dir={args.model_dir}")
    print(f"torch_dtype={torch_dtype}")
    print(f"temperature={args.temperature}")
    print("generated_text:")
    print(generated_text)


if __name__ == "__main__":
    main()
