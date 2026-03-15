from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "data" / "pre_train" / "Qwen2.5-0.5B"
DEFAULT_DATA_PATH = ROOT / "data" / "post_train" / "sft_mix_3000.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "post_train" / "qwen2.5-0.5b-alpaca-zh-lora"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def format_prompt(instruction: str, input_text: str) -> str:
    parts = [
        "你是一个有帮助的助手。",
        "",
        "### 指令：",
        instruction.strip(),
    ]
    if input_text.strip():
        parts.extend(["", "### 输入：", input_text.strip()])
    parts.extend(["", "### 回答：", ""])
    return "\n".join(parts)


class SFTDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, data_path: Path, tokenizer: AutoTokenizer, max_length: int):
        self.samples: list[dict[str, torch.Tensor]] = []
        rows = [
            json.loads(line)
            for line in data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        for row in rows:
            prompt = format_prompt(row["instruction"], row["input"])
            answer = row["output"].strip() + tokenizer.eos_token  # type: ignore # type: ignore

            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids  # type: ignore
            answer_ids = tokenizer(answer, add_special_tokens=False).input_ids  # type: ignore

            if len(prompt_ids) >= max_length:
                prompt_ids = prompt_ids[: max_length // 2]
            remaining = max_length - len(prompt_ids)
            answer_ids = answer_ids[:remaining]

            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids
            attention_mask = [1] * len(input_ids)

            self.samples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


class SFTCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].size(0) for item in batch)
        input_ids = []
        labels = []
        attention_mask = []

        for item in batch:
            pad_len = max_len - item["input_ids"].size(0)
            input_ids.append(
                torch.cat(
                    [
                        item["input_ids"],
                        torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                    ]
                )
            )
            labels.append(
                torch.cat(
                    [item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]
                )
            )
            attention_mask.append(
                torch.cat(
                    [item["attention_mask"], torch.zeros((pad_len,), dtype=torch.long)]
                )
            )

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }


@dataclass
class TrainConfig:
    model_dir: Path
    data_path: Path
    output_dir: Path
    max_length: int
    num_epochs: int
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    max_updates: int | None
    log_every: int
    save_every: int
    seed: int
    device: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    sample_prompt: str
    sample_tokens: int


@torch.no_grad()
def generate_sample(
    model,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    was_training = model.training
    model.eval()
    prompt_text = format_prompt(prompt, "")
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)  # type: ignore
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,  # type: ignore
        pad_token_id=tokenizer.pad_token_id,  # type: ignore
    )
    new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)  # type: ignore
    if was_training:
        model.train()
    return text


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = pick_device(config.device)
    dtype = pick_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = SFTDataset(config.data_path, tokenizer, config.max_length)
    collator = SFTCollator(tokenizer.pad_token_id)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collator
    )
    num_batches = len(loader)
    if num_batches == 0:
        raise ValueError(f"dataset is empty after preprocessing: {config.data_path}")
    updates_per_epoch = math.ceil(num_batches / config.grad_accum_steps)
    planned_updates = updates_per_epoch * config.num_epochs
    total_updates = (
        min(planned_updates, config.max_updates)
        if config.max_updates is not None
        else planned_updates
    )
    warmup_steps = int(total_updates * config.warmup_ratio)

    model = AutoModelForCausalLM.from_pretrained(config.model_dir, dtype=dtype).to(
        device  # type: ignore
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    def lr_lambda(current_step: int) -> float:
        if total_updates <= 0:
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        if total_updates == warmup_steps:
            return 1.0
        decay_steps = total_updates - warmup_steps
        progress = (current_step - warmup_steps) / max(decay_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    optimizer.zero_grad(set_to_none=True)

    micro_step = 0
    update_step = 0
    running_loss = 0.0
    micro_since_log = 0

    print(f"device={device}")
    print(f"model_dir={config.model_dir}")
    print(f"data_path={config.data_path}")
    print(f"output_dir={config.output_dir}")
    print(f"samples={len(dataset)}")
    print(f"num_epochs={config.num_epochs}")
    print(f"num_batches_per_epoch={num_batches}")
    print(f"updates_per_epoch={updates_per_epoch}")
    print(f"planned_updates={planned_updates}")
    print(f"total_updates={total_updates}")
    print(f"warmup_steps={warmup_steps}")

    for epoch in range(1, config.num_epochs + 1):
        final_accum_size = num_batches % config.grad_accum_steps
        if final_accum_size == 0:
            final_accum_size = config.grad_accum_steps

        for batch_index, batch in enumerate(loader, start=1):
            micro_step += 1
            is_final_group = batch_index > num_batches - final_accum_size
            accum_target = (
                final_accum_size if is_final_group else config.grad_accum_steps
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            raw_loss = outputs.loss
            loss = raw_loss / accum_target
            loss.backward()
            running_loss += raw_loss.item()
            micro_since_log += 1

            if (
                batch_index % config.grad_accum_steps != 0
                and batch_index != num_batches
            ):
                continue

            torch.nn.utils.clip_grad_norm_(
                (param for param in model.parameters() if param.requires_grad),
                config.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1

            if (
                update_step == 1
                or update_step % config.log_every == 0
                or update_step == total_updates
            ):
                avg_loss = running_loss / micro_since_log
                running_loss = 0.0
                micro_since_log = 0
                sample = generate_sample(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=config.sample_prompt,
                    max_new_tokens=config.sample_tokens,
                    device=device,
                )
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"\nepoch={epoch} update={update_step}/{total_updates} "
                    f"lr={current_lr:.7f} loss={avg_loss:.4f}\nsample:\n{sample}\n"
                )

            if update_step % config.save_every == 0 or update_step == total_updates:
                config.output_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(config.output_dir)  # type: ignore
                tokenizer.save_pretrained(config.output_dir)
                (config.output_dir / "train_config.json").write_text(
                    json.dumps(
                        asdict(config), ensure_ascii=False, indent=2, default=str
                    ),
                    encoding="utf-8",
                )
                print(f"saved adapter to {config.output_dir}")

            if update_step >= total_updates:
                break

        if update_step >= total_updates:
            break


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="用 LoRA 对 Qwen2.5-0.5B 做按 epoch 的监督微调。"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Only for debug; when unset, all samples are used for every epoch.",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--sample-prompt", default="美国的首都在哪里。")
    parser.add_argument("--sample-tokens", type=int, default=80)
    args = parser.parse_args()

    return TrainConfig(
        model_dir=args.model_dir,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        max_updates=args.max_updates,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed,
        device=args.device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        sample_prompt=args.sample_prompt,
        sample_tokens=args.sample_tokens,
    )


if __name__ == "__main__":
    train(parse_args())
