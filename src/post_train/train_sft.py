from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "data" / "pre_train" / "Qwen2.5-0.5B"
DEFAULT_DATA_PATH = ROOT / "data" / "post_train" / "alpaca_zh_64.jsonl"
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
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    max_steps: int
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
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # type: ignore
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,  # type: ignore
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)  # type: ignore
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

    model = AutoModelForCausalLM.from_pretrained(config.model_dir, dtype=dtype).to(
        device  # type: ignore
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    step = 0
    running_loss = 0.0
    since_last_log = 0

    print(f"device={device}")
    print(f"model_dir={config.model_dir}")
    print(f"data_path={config.data_path}")
    print(f"output_dir={config.output_dir}")
    print(f"samples={len(dataset)}")

    while step < config.max_steps:
        for batch in loader:
            step += 1
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            raw_loss = outputs.loss
            loss = raw_loss / config.grad_accum_steps
            loss.backward()
            running_loss += raw_loss.item()
            since_last_log += 1

            if step % config.grad_accum_steps == 0 or step == config.max_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step == 1 or step % config.log_every == 0 or step == config.max_steps:
                avg_loss = running_loss / since_last_log
                running_loss = 0.0
                since_last_log = 0
                sample = generate_sample(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=config.sample_prompt,
                    max_new_tokens=config.sample_tokens,
                    device=device,
                )
                print(f"\nstep={step} loss={avg_loss:.4f}\nsample:\n{sample}\n")

            if step % config.save_every == 0 or step == config.max_steps:
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

            if step >= config.max_steps:
                break


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="用 LoRA 对 Qwen2.5-0.5B 做一个最小监督微调示例。"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--sample-prompt", default="请介绍一下什么是监督微调。")
    parser.add_argument("--sample-tokens", type=int, default=80)
    args = parser.parse_args()

    return TrainConfig(
        model_dir=args.model_dir,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
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
