# 预训练 MVP

这个目录里现在有两条清晰分开的路径：

1. 自己训练一个最小的 GPT MVP，再加载自己保存的 checkpoint 做推理。
2. 直接加载 Hugging Face 上已经预训练好的轻量模型权重做推理。

第一条路径故意保持得很小，目的是让你能直接看清完整链路：

1. 读取原始文本。
2. 把字符转换成 token id。
3. 把连续 token 流切成 `(input, target)` 训练样本。
4. 用 next-token cross-entropy 训练一个很小的因果 Transformer。
5. 把训练好的模型保存成 checkpoint。
6. 推理时直接加载 checkpoint，用 prompt 做生成。

文件说明：

- `modeling.py`：共享的 tokenizer、dataset、模型结构、生成函数、checkpoint 保存/加载逻辑。
- `train.py`：训练脚本，负责训练并保存你自己的 MVP checkpoint。
- `infer.py`：推理脚本，负责加载你自己训练出来的 checkpoint 并生成文本。
- `infer_hf.py`：推理脚本，负责加载 Hugging Face 上下载的现成权重并生成文本。
- `../../data/pre_train/tiny_corpus.txt`：用于快速实验的中文小语料。
- `../../data/pre_train/tiny_gpt_mvp.pt`：自己训练出来的默认 checkpoint。
- `../../data/pre_train/Qwen2.5-0.5B/`：从 Hugging Face 下载的纯预训练底座模型权重目录。

## 路径一：自己训练一个最小 GPT MVP

训练：

```bash
.venv/bin/python src/pre_train/train.py
```

如果你想先做一次更短的 CPU 训练：

```bash
.venv/bin/python src/pre_train/train.py --max-steps 80 --eval-every 20 --batch-size 8 --seq-len 48 --sample-prompt "预训练的目标是"
```

训练完成后做推理：

```bash
.venv/bin/python src/pre_train/infer.py --prompt "预训练的目标是" --generate-tokens 80
```

如果你想指定自己的 checkpoint：

```bash
.venv/bin/python src/pre_train/infer.py --checkpoint-path data/pre_train/tiny_gpt_mvp.pt --prompt "模型通过读取"
```

训练脚本常用参数：

- `--data-path`：替换成你自己的文本文件。
- `--checkpoint-path`：训练后 checkpoint 的保存位置。
- `--seq-len`：上下文窗口长度。
- `--emb-dim`、`--num-heads`、`--num-layers`：模型规模。
- `--max-steps`：训练步数。
- `--sample-prompt`：训练过程中打印示例生成时的起始文本。
- `--sample-tokens`：训练过程中每次示例生成多少个 token。

训练参数逐项解释：

- `--data-path`：训练语料文件路径。脚本会读取这个文本文件，并做字符级 tokenizer。
- `--checkpoint-path`：模型 checkpoint 的保存路径。验证损失变得更好时，会把当前模型保存到这里。
- `--batch-size`：每一步训练同时使用多少条样本。更大通常更稳定，但更占显存。
- `--train-split`：训练集占总语料的比例。`0.9` 表示 90% 用于训练，10% 用于验证。
- `--seq-len`：上下文窗口长度，也就是每个样本包含多少个 token。它决定模型一次最多能看到多长的上下文。
- `--emb-dim`：token embedding 和隐藏状态的维度。越大表示能力越强，但参数量和计算量也会增加。
- `--num-heads`：多头注意力里的头数。它必须能整除 `emb-dim`。
- `--num-layers`：Transformer block 的层数。层数越多，模型越深。
- `--dropout`：dropout 比例，用来减轻过拟合。
- `--learning-rate`：学习率，控制每次参数更新的步幅。太大容易不稳定，太小会学得很慢。
- `--weight-decay`：权重衰减，一种常见的正则化手段，用来抑制权重过大。
- `--max-steps`：总训练步数。一轮步数就是取一个 batch、前向传播、反向传播、更新参数。
- `--eval-every`：每隔多少步做一次验证，并打印一次示例生成。
- `--eval-batches`：每次验证时使用多少个 batch 来估算验证损失。越大越稳定，但也越慢。
- `--seed`：随机种子，用于尽量保证结果可复现。
- `--device`：训练设备。`auto` 会自动选择 `cuda`、`mps` 或 `cpu`。
- `--sample-prompt`：训练过程中用于观察生成效果的起始文本。它不参与训练，只用于展示当前模型学到了什么。
- `--sample-tokens`：训练过程中每次示例生成要续写多少个 token。

推理脚本常用参数：

- `--checkpoint-path`：要加载的 checkpoint 路径。
- `--prompt`：生成时的起始文本。
- `--generate-tokens`：往后续写多少个 token。

说明：

- 这是教学用 MVP，不是生产级训练器。
- tokenizer 采用字符级实现，是为了把过程尽量展开、看得更清楚。
- 当前示例语料是中文，因此默认 prompt 也用中文，直接运行时更容易观察结果。
- 训练和推理已经分开，训练阶段只负责更新参数和保存模型，推理阶段只负责加载模型和生成文本。
- 在这么小的语料上，模型大多会记住局部模式，这属于预期现象。

## 路径二：直接使用 Hugging Face 现成权重

当前已经下载了这个模型：

- `Qwen/Qwen2.5-0.5B`
- 本地目录：`data/pre_train/Qwen2.5-0.5B`

它不是你自己训练出来的模型，而是别人已经预训练好的纯底座模型。

直接推理：

```bash
uv run python src/pre_train/infer_hf.py --model-dir data/pre_train/Qwen2.5-0.5B --prompt "什么是 next token prediction？"
```

如果你想指定本地权重目录：

```bash
uv run python src/pre_train/infer_hf.py --model-dir data/pre_train/Qwen2.5-0.5B --prompt "模型为什么需要大量语料？"
```

如果你想控制生成随机性，可以加 `--temperature`：

```bash
uv run python src/pre_train/infer_hf.py --model-dir data/pre_train/Qwen2.5-0.5B --prompt "什么是大模型预训练？" --temperature 0.7
```

这个脚本的定位要分清：

- `infer.py`：加载你自己训练出来的 `tiny_gpt_mvp.pt`
- `infer_hf.py`：加载 Hugging Face 下载下来的 `Qwen2.5-0.5B`

`infer_hf.py` 常用参数：

- `--model-dir`：Hugging Face 模型目录。
- `--prompt`：生成时的输入文本。
- `--max-new-tokens`：最多续写多少个 token。
- `--temperature`：生成温度。大于 0 时启用采样，越大越发散；小于等于 0 时退回贪心解码。
- `--device`：运行设备，`auto` 会自动选 `cuda`、`mps` 或 `cpu`。

补充说明：

- Hugging Face 这套权重体积大约 953MB，已经放进 `.gitignore`，不会被误提交。
- 相关依赖已经通过 `uv` 加入项目，包括 `huggingface-hub`、`transformers` 和 `safetensors`。
