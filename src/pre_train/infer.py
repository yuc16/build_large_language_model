from __future__ import annotations

import argparse
from pathlib import Path

from modeling import DEFAULT_CHECKPOINT_PATH, generate, load_checkpoint, pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="加载已训练 checkpoint 并做文本生成。")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--prompt", default="预训练的目标是")
    parser.add_argument("--generate-tokens", type=int, default=120)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    model, tokenizer, model_config, train_config = load_checkpoint(
        args.checkpoint_path, device
    )

    sample = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.generate_tokens,
        device=device,
    )

    print(f"device={device}")
    print(f"checkpoint_path={args.checkpoint_path}")
    print(f"seq_len={model_config.seq_len} emb_dim={model_config.emb_dim}")
    if train_config:
        print(f"trained_from={train_config.get('data_path')}")
    print("generated_text:")
    print(sample)


if __name__ == "__main__":
    main()
