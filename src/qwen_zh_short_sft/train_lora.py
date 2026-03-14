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
DEFAULT_MODEL_ID = str(ROOT / "data" / "pre_train" / "Qwen2.5-0.5B")
DEFAULT_DATA_PATH = ROOT / "data" / "qwen_zh_short_sft" / "raw" / "short_sft_mix_6000.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "qwen_zh_short_sft" / "adapters" / "qwen2.5-0.5b-short-sft-lora"


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
    if device.type == "cuda":
        return torch.float16
    return torch.float32


class ChatDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, data_path: Path, tokenizer: AutoTokenizer, max_length: int):
        self.samples: list[dict[str, torch.Tensor]] = []
        self.skipped = 0

        rows = [
            json.loads(line)
            for line in data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        for row in rows:
            messages = row["messages"]
            prompt_messages = messages[:-1]
            full_text = tokenizer.apply_chat_template(  # type: ignore
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = tokenizer.apply_chat_template(  # type: ignore
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_ids = tokenizer(full_text, add_special_tokens=False).input_ids  # type: ignore
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids  # type: ignore
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]
            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = full_ids.copy()
            for idx in range(prompt_len):
                labels[idx] = -100

            if all(token == -100 for token in labels):
                self.skipped += 1
                continue

            self.samples.append(
                {
                    "input_ids": torch.tensor(full_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


class Collator:
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
                    [item["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]
                )
            )
            labels.append(
                torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            )
            attention_mask.append(
                torch.cat([item["attention_mask"], torch.zeros((pad_len,), dtype=torch.long)])
            )

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }


@dataclass
class TrainConfig:
    model_id: str
    data_path: Path
    output_dir: Path
    max_length: int
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    max_updates: int
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
    messages = [
        {"role": "system", "content": "你是一个有帮助的中文助手。除非用户要求详细解释，否则优先简短回答。"},
        {"role": "user", "content": prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(  # type: ignore
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)  # type: ignore
    terminators = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")  # type: ignore
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in terminators:
        terminators.append(im_end_id)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=terminators,
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

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = ChatDataset(config.data_path, tokenizer, config.max_length)
    collator = Collator(tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collator)

    model = AutoModelForCausalLM.from_pretrained(config.model_id, dtype=dtype).to(device)  # type: ignore
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loader_iter = iter(loader)
    micro_step = 0
    update_step = 0
    running_loss = 0.0
    micro_since_log = 0

    print(f"device={device}")
    print(f"model_id={config.model_id}")
    print(f"data_path={config.data_path}")
    print(f"output_dir={config.output_dir}")
    print(f"samples={len(dataset)}")
    print(f"skipped_samples={dataset.skipped}")

    while update_step < config.max_updates:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        micro_step += 1
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        raw_loss = outputs.loss
        loss = raw_loss / config.grad_accum_steps
        loss.backward()
        running_loss += raw_loss.item()
        micro_since_log += 1

        if micro_step % config.grad_accum_steps != 0:
            continue

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1

        if update_step == 1 or update_step % config.log_every == 0 or update_step == config.max_updates:
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
            print(f"\nupdate={update_step} loss={avg_loss:.4f}\nsample:\n{sample}\n")

        if update_step % config.save_every == 0 or update_step == config.max_updates:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(config.output_dir)  # type: ignore
            tokenizer.save_pretrained(config.output_dir)
            (config.output_dir / "train_config.json").write_text(
                json.dumps(asdict(config), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"saved adapter to {config.output_dir}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="在本地 Qwen2.5-0.5B 上用中文短答案数据做 LoRA 微调。")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-updates", type=int, default=160)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--sample-prompt", default="法国的首都在哪里？请只回答城市名。")
    parser.add_argument("--sample-tokens", type=int, default=8)
    args = parser.parse_args()

    return TrainConfig(
        model_id=args.model_id,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
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
