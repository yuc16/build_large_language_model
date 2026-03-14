# 监督微调 MVP

这个目录放的是一个尽量简单的监督微调示例，目标是帮助你理解：

1. 从 Hugging Face 下载一个指令数据集子集。
2. 把数据整理成本地 `jsonl`。
3. 在 `Qwen2.5-0.5B` 底座模型上做一个最小的 LoRA 微调。
4. 推理时加载 `base model + adapter` 看看效果。

## 文件说明

- `download_dataset.py`：下载 Hugging Face 数据集子集并保存到本地。
- `train_sft.py`：用 LoRA 做一个最小监督微调。
- `infer_sft.py`：加载 `Qwen2.5-0.5B + LoRA adapter` 做推理。

## 使用的数据集

默认下载：

- `shibing624/alpaca-zh`

这是一个中文指令数据集，字段比较简单，适合拿来做 SFT 入门。

## 第一步：下载训练数据

```bash
uv run python src/post_train/download_dataset.py
```

默认会保存到：

- `data/post_train/alpaca_zh_500.jsonl`

如果你只想先下载更小的数据子集：

```bash
uv run python src/post_train/download_dataset.py --max-samples 100 --output-path data/post_train/alpaca_zh_100.jsonl
```

## 第二步：做一个最小 LoRA 微调

```bash
uv run python src/post_train/train_sft.py
```

默认配置会：

- 使用 `data/pre_train/Qwen2.5-0.5B`
- 使用 `data/post_train/alpaca_zh_500.jsonl`
- 把 LoRA adapter 保存到 `data/post_train/qwen2.5-0.5b-alpaca-zh-lora`

如果你想先做一次更轻量的尝试：

```bash
uv run python src/post_train/train_sft.py --data-path data/post_train/alpaca_zh_100.jsonl --max-steps 10 --max-length 192
```

## 第三步：加载 adapter 做推理

```bash
uv run python src/post_train/infer_sft.py --prompt "请介绍一下什么是监督微调。"
```

## 训练脚本核心参数

- `--model-dir`：底座模型目录，默认是 `data/pre_train/Qwen2.5-0.5B`
- `--data-path`：训练数据 `jsonl`
- `--output-dir`：LoRA adapter 输出目录
- `--max-length`：单条样本最大长度
- `--batch-size`：每步 batch 大小
- `--grad-accum-steps`：梯度累积步数
- `--learning-rate`：学习率
- `--max-steps`：训练总步数
- `--lora-r`、`--lora-alpha`、`--lora-dropout`：LoRA 配置

## 说明

- 这里用的是 LoRA，不是全量微调。这样更省资源，也更适合入门。
- 训练时只会更新 adapter 权重，底座模型权重不会改。
- 推理时需要同时加载底座模型和 LoRA adapter。
