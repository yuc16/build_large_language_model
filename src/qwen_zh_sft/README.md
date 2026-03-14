# Qwen 中文单轮 SFT

这是一套和现有 `src/post_train` / `data/post_train` 分开的中文监督微调实验。

目录约定：

- `data/qwen_zh_sft/raw/`
  存放下载并规范化后的中文 SFT 数据
- `data/qwen_zh_sft/adapters/`
  存放训练得到的 LoRA adapter

文件说明：

- `download_dataset.py`
  下载 `shibing624/alpaca-zh`，并转成 `messages` 格式
- `train_lora.py`
  在本地 `Qwen2.5-0.5B` 上做中文单轮 LoRA SFT
- `infer_lora.py`
  加载 `Qwen2.5-0.5B + LoRA adapter` 做推理

## 第一步：下载数据

```bash
uv run python src/qwen_zh_sft/download_dataset.py
```

默认会生成：

- `data/qwen_zh_sft/raw/alpaca_zh_5000.jsonl`

## 第二步：训练 LoRA

```bash
uv run python src/qwen_zh_sft/train_lora.py
```

默认：

- 基座模型：`data/pre_train/Qwen2.5-0.5B`
- 数据：`data/qwen_zh_sft/raw/alpaca_zh_5000.jsonl`
- 输出：`data/qwen_zh_sft/adapters/qwen2.5-0.5b-alpaca-zh-lora`

## 第三步：推理

```bash
uv run python src/qwen_zh_sft/infer_lora.py --prompt "法国的首都在哪里？请只回答城市名。"
```

如果你想测试“只回答一个词/一句话”的场景，建议显式限制生成长度，例如：

```bash
uv run python src/qwen_zh_sft/infer_lora.py --prompt "法国的首都在哪里？请只回答城市名。" --max-new-tokens 2
```

说明：

- 这套实验只使用中文、单轮问答为主的数据。
- 推理时使用 Qwen 的 chat template，而不是直接把裸文本塞给模型。
- 训练脚本里的 `max-updates` 表示真实的参数更新次数，不是 micro-step。
