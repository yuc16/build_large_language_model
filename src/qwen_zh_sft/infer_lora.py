from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = str(ROOT / "data" / "pre_train" / "Qwen2.5-0.5B")
DEFAULT_ADAPTER_DIR = (
    ROOT / "data" / "qwen_zh_sft" / "adapters" / "qwen2.5-0.5b-alpaca-zh-lora"
)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="加载 Qwen2.5-0.5B + 中文 SFT LoRA adapter 做推理。"
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--prompt", default="法国的首都在哪里？请只回答城市名。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    dtype = pick_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype).to(device)  # type: ignore
    model = PeftModel.from_pretrained(base_model, args.adapter_dir).to(device)
    model.eval()

    messages = [
        {"role": "system", "content": "你是一个有帮助的中文助手。"},
        {"role": "user", "content": args.prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(  # type: ignore
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)  # type: ignore
    do_sample = args.temperature > 0
    terminators = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")  # type: ignore
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in terminators:
        terminators.append(im_end_id)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            eos_token_id=terminators,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.05,
        )

    new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    response_text = tokenizer.decode(new_ids, skip_special_tokens=True)  # type: ignore

    print("mode=qwen_zh_sft_lora_inference")
    print(f"model_id={args.model_id}")
    print(f"adapter_dir={args.adapter_dir}")
    print(f"device={device}")
    print(f"temperature={args.temperature}")
    print("assistant_response:")
    print(response_text)


if __name__ == "__main__":
    main()
