from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT / "data" / "pre_train" / "tiny_corpus.txt"
DEFAULT_CHECKPOINT_PATH = ROOT / "data" / "pre_train" / "tiny_gpt_mvp.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _make_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_serializable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_make_serializable(item) for item in value)
    return value


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class ModelConfig:
    seq_len: int
    emb_dim: int
    num_heads: int
    num_layers: int
    dropout: float


class CharTokenizer:
    def __init__(self, text: str | None = None, itos: list[str] | None = None):
        if itos is not None:
            self.itos = itos
        elif text is not None:
            chars = sorted(set(text))
            self.itos = ["<unk>", *chars]
        else:
            raise ValueError("Either text or itos must be provided.")

        self.stoi = {token: index for index, token in enumerate(self.itos)}
        self.unk_token = "<unk>"
        self.unk_id = self.stoi[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        pieces = []
        for token_id in token_ids:
            token = self.itos[token_id]
            pieces.append("?" if token == self.unk_token else token)
        return "".join(pieces)


class NextTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, token_ids: list[int], seq_len: int):
        if len(token_ids) <= seq_len:
            raise ValueError(
                f"Dataset is too small for seq_len={seq_len}. "
                f"Need more than {seq_len} tokens, got {len(token_ids)}."
            )
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.tokens) - self.seq_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[index : index + self.seq_len]
        y = self.tokens[index + 1 : index + self.seq_len + 1]
        return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float, seq_len: int):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, emb_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:num_tokens, :num_tokens], float("-inf"))  # type: ignore
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, num_tokens, emb_dim)
        out = self.resid_dropout(self.proj(out))
        return out


class MLP(nn.Module):
    def __init__(self, emb_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float, seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = CausalSelfAttention(emb_dim, num_heads, dropout, seq_len)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(emb_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig):
        super().__init__()
        self.seq_len = config.seq_len
        self.token_embedding = nn.Embedding(vocab_size, config.emb_dim)
        self.position_embedding = nn.Embedding(config.seq_len, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.emb_dim,
                    config.num_heads,
                    config.dropout,
                    config.seq_len,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, num_tokens = x.shape
        positions = torch.arange(num_tokens, device=x.device)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def build_loaders(
    text: str,
    tokenizer: CharTokenizer,
    seq_len: int,
    batch_size: int,
    train_split: float,
) -> tuple[DataLoader, DataLoader]:
    token_ids = tokenizer.encode(text)
    split_index = int(len(token_ids) * train_split)
    train_ids = token_ids[:split_index]
    val_ids = token_ids[split_index - seq_len :]

    train_dataset = NextTokenDataset(train_ids, seq_len)
    val_dataset = NextTokenDataset(val_ids, seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader


def cycle(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore
    while True:
        for batch in loader:
            yield batch  # type: ignore


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    loader: DataLoader,
    eval_batches: int,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for step, (x, y) in enumerate(loader):
        if step >= eval_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    if was_training:
        model.train()
    if not losses:
        raise ValueError("Validation loader is empty. Reduce batch size or seq_len.")
    return sum(losses) / len(losses)


@torch.no_grad()
def generate(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    was_training = model.training
    model.eval()
    token_ids = tokenizer.encode(prompt)
    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        context = x[:, -model.seq_len :]
        logits = model(context)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat([x, next_token], dim=1)

    if was_training:
        model.train()
    return tokenizer.decode(x[0].tolist())


def save_checkpoint(
    checkpoint_path: Path,
    model: TinyGPT,
    tokenizer: CharTokenizer,
    model_config: ModelConfig,
    train_config: dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "tokenizer_itos": tokenizer.itos,
        "train_config": _make_serializable(train_config),
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[TinyGPT, CharTokenizer, ModelConfig, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ModelConfig(**payload["model_config"])
    tokenizer = CharTokenizer(itos=payload["tokenizer_itos"])
    model = TinyGPT(vocab_size=tokenizer.vocab_size, config=model_config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    train_config = payload.get("train_config", {})
    return model, tokenizer, model_config, train_config
