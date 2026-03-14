from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from modeling import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DATA_PATH,
    CharTokenizer,
    ModelConfig,
    TinyGPT,
    build_loaders,
    cycle,
    estimate_loss,
    generate,
    pick_device,
    save_checkpoint,
    set_seed,
)


@dataclass
class TrainConfig:
    data_path: Path
    checkpoint_path: Path
    batch_size: int
    train_split: float
    learning_rate: float
    weight_decay: float
    max_steps: int
    eval_every: int
    eval_batches: int
    seed: int
    device: str
    sample_prompt: str
    sample_tokens: int


def train(train_config: TrainConfig, model_config: ModelConfig) -> None:
    set_seed(train_config.seed)
    device = pick_device(train_config.device)

    text = train_config.data_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text=text)
    train_loader, val_loader = build_loaders(
        text=text,
        tokenizer=tokenizer,
        seq_len=model_config.seq_len,
        batch_size=train_config.batch_size,
        train_split=train_config.train_split,
    )

    model = TinyGPT(vocab_size=tokenizer.vocab_size, config=model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    train_iter = cycle(train_loader)
    total_params = sum(param.numel() for param in model.parameters())
    best_val_loss = float("inf")

    print(f"device={device}")
    print(f"data_path={train_config.data_path}")
    print(f"checkpoint_path={train_config.checkpoint_path}")
    print(f"chars={len(text)} vocab_size={tokenizer.vocab_size}")
    print(f"train_batches={len(train_loader)} val_batches={len(val_loader)}")
    print(f"model_params={total_params:,}")

    for step in range(1, train_config.max_steps + 1):
        x, y = next(train_iter)  # type: ignore
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (
            step == 1
            or step % train_config.eval_every == 0
            or step == train_config.max_steps
        ):
            train_loss = loss.item()
            val_loss = estimate_loss(
                model, val_loader, train_config.eval_batches, device
            )
            sample = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=train_config.sample_prompt,
                max_new_tokens=train_config.sample_tokens,
                device=device,
            )
            print(
                f"\nstep={step} train_loss={train_loss:.4f} val_loss={val_loss:.4f}\n"
                f"sample:\n{sample}\n"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    checkpoint_path=train_config.checkpoint_path,
                    model=model,
                    tokenizer=tokenizer,
                    model_config=model_config,
                    train_config=asdict(train_config),
                )
                print(
                    f"saved checkpoint to {train_config.checkpoint_path} "
                    f"(best_val_loss={best_val_loss:.4f})"
                )


def parse_args() -> tuple[TrainConfig, ModelConfig]:
    parser = argparse.ArgumentParser(
        description="训练一个最小可运行的 GPT 预训练 MVP。"
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-prompt", default="预训练的目标是")
    parser.add_argument("--sample-tokens", type=int, default=120)
    args = parser.parse_args()

    train_config = TrainConfig(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        train_split=args.train_split,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        seed=args.seed,
        device=args.device,
        sample_prompt=args.sample_prompt,
        sample_tokens=args.sample_tokens,
    )
    model_config = ModelConfig(
        seq_len=args.seq_len,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    return train_config, model_config


if __name__ == "__main__":
    train(*parse_args())
