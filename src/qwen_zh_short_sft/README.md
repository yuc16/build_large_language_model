# Qwen 中文短答 SFT

这是一套和 `src/qwen_zh_sft` 分开的短答案专项 SFT 实验。

目录约定：

- `data/qwen_zh_short_sft/raw/`
  存放短答案训练数据
- `data/qwen_zh_short_sft/adapters/`
  存放训练得到的 LoRA adapter

文件说明：

- `download_dataset.py`
  从 Hugging Face 下载中文数据，并过滤出短答案样本
- `train_lora.py`
  在本地 `Qwen2.5-0.5B` 上做短答专项 LoRA SFT
- `infer_lora.py`
  加载 `Qwen2.5-0.5B + LoRA adapter` 做推理

## 下载数据

```bash
uv run python src/qwen_zh_short_sft/download_dataset.py
```

## 训练

```bash
uv run python src/qwen_zh_short_sft/train_lora.py
```

## 推理

```bash
uv run python src/qwen_zh_short_sft/infer_lora.py --prompt "法国的首都在哪里？请只回答城市名。"
```
