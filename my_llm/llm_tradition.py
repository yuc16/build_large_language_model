import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


# ============================================================
# 第一步：模型配置
# ============================================================
@dataclass
class LLMConfig:
    """模型超参数配置，参考 LLaMA 的设计"""

    vocab_size: int = 32000  # 词表大小
    hidden_size: int = 512  # 隐藏层维度 (d_model)
    num_layers: int = 8  # Transformer 层数
    num_heads: int = 8  # Query 注意力头数
    num_kv_heads: int = 4  # Key/Value 头数 (GQA), 若等于num_heads则退化为MHA
    intermediate_size: int = 1408  # FFN 中间层维度 (通常约 hidden_size * 2.75)
    max_seq_len: int = 2048  # 最大序列长度
    rms_norm_eps: float = 1e-6  # RMSNorm 的 epsilon
    rope_theta: float = 10000.0  # RoPE 的 base frequency


# ============================================================
# 第二步：RMSNorm — 替代 LayerNorm
# ============================================================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    相比 LayerNorm，去掉了减均值的操作，只做缩放，计算更快。
    公式: output = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可学习的缩放参数 γ
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, hidden_size)
        # 1. 计算输入的均方根
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # 2. 归一化并缩放
        return (x / rms) * self.weight


# ============================================================
# 第三步：RoPE 旋转位置编码
# ============================================================
def precompute_rope_freqs(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """
    预计算 RoPE 所需的复数频率向量。
    每两个维度共享一个旋转频率: theta_i = theta^(-2i/d)
    返回 shape: (max_seq_len, head_dim // 2) 的复数张量
    """
    # 频率: theta_i = 1 / (theta^(2i/d)), i = 0, 1, ..., d/2 - 1
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # 位置索引: m = 0, 1, ..., max_seq_len - 1
    positions = torch.arange(max_seq_len).float()
    # 外积得到每个位置、每个维度对的角度: (seq_len, head_dim//2)
    angles = torch.outer(positions, freqs)
    # 转为复数形式 e^(i*angle) = cos(angle) + i*sin(angle)，方便后续旋转
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    将 RoPE 应用到输入张量 x 上。
    x shape: (batch, seq_len, num_heads, head_dim)
    freqs_cis shape: (seq_len, head_dim // 2)
    """
    # 把相邻两个维度配对，视为复数: (batch, seq_len, num_heads, head_dim//2, 2) -> 复数
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # 调整 freqs_cis 的形状以便广播: (1, seq_len, 1, head_dim//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    # 复数乘法实现旋转！这就是 RoPE 的精髓
    x_rotated = x_complex * freqs_cis
    # 复数转回实数: (batch, seq_len, num_heads, head_dim//2, 2) -> (batch, seq_len, num_heads, head_dim)
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    return x_out.type_as(x)


# ============================================================
# 第四步：Grouped Query Attention (GQA)
# ============================================================
class Attention(nn.Module):
    """
    分组查询注意力 (Grouped Query Attention)
    - 多个 Q 头共享一组 KV 头，节省 KV Cache 显存
    - 支持因果掩码 (causal mask)，确保 token 只能看到之前的 token
    - 对 Q 和 K 应用 RoPE 位置编码
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.num_heads = config.num_heads  # Q 头数
        self.num_kv_heads = config.num_kv_heads  # KV 头数
        self.head_dim = config.hidden_size // config.num_heads  # 每个头的维度
        # 每组 KV 头对应多少个 Q 头
        self.num_groups = self.num_heads // self.num_kv_heads

        # Q/K/V 投影矩阵 (注意 KV 的参数量比 Q 少)
        self.wq = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        # 输出投影
        self.wo = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # 1. 线性投影得到 Q, K, V
        q = self.wq(x)  # (batch, seq_len, num_heads * head_dim)
        k = self.wk(x)  # (batch, seq_len, num_kv_heads * head_dim)
        v = self.wv(x)  # (batch, seq_len, num_kv_heads * head_dim)

        # 2. 拆分多头: (batch, seq_len, num_heads, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # 3. 对 Q 和 K 应用 RoPE (注意: V 不需要位置编码!)
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # 4. GQA: 将 KV 头扩展以匹配 Q 头数量
        #    例如 num_heads=8, num_kv_heads=4, 则每个 KV 头复制 2 次
        if self.num_groups > 1:
            k = k.unsqueeze(3).expand(
                -1, -1, -1, self.num_groups, -1
            )  # 在头维度插入并扩展
            k = k.reshape(batch, seq_len, self.num_heads, self.head_dim)  # 合并
            v = v.unsqueeze(3).expand(-1, -1, -1, self.num_groups, -1)
            v = v.reshape(batch, seq_len, self.num_heads, self.head_dim)

        # 5. 转置为 (batch, num_heads, seq_len, head_dim) 以便做注意力
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 6. 计算注意力分数: softmax(QK^T / sqrt(d_k)) * V
        scale = math.sqrt(self.head_dim)
        scores = (
            torch.matmul(q, k.transpose(-2, -1)) / scale
        )  # (batch, heads, seq, seq)

        # 7. 应用因果掩码: 上三角设为 -inf，确保只能看到之前的 token
        if mask is not None:
            scores = scores + mask  # mask 中未来位置为 -inf

        attn_weights = F.softmax(scores, dim=-1)

        # 8. 加权求和
        output = torch.matmul(attn_weights, v)  # (batch, heads, seq, head_dim)

        # 9. 合并多头: (batch, seq_len, hidden_size)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        # 10. 输出投影
        return self.wo(output)


# ============================================================
# 第五步：SwiGLU Feed-Forward Network
# ============================================================
class FeedForward(nn.Module):
    """
    SwiGLU 前馈网络
    相比传统 FFN 多了一个门控分支，效果更好。
    公式: output = (Swish(x @ W1) * (x @ W3)) @ W2
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        hidden = config.intermediate_size
        # W1: 门控分支的投影
        self.w1 = nn.Linear(config.hidden_size, hidden, bias=False)
        # W2: 降维投影 (输出)
        self.w2 = nn.Linear(hidden, config.hidden_size, bias=False)
        # W3: 值分支的投影
        self.w3 = nn.Linear(config.hidden_size, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: Swish 激活的门控 * 值分支，再降维
        # Swish(x) = x * sigmoid(x)，即 F.silu
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ============================================================
# 第六步：Transformer Block (一层 Decoder)
# ============================================================
class TransformerBlock(nn.Module):
    """
    一个 Transformer Decoder 层，采用 Pre-Norm 结构:
    x -> RMSNorm -> Attention -> 残差连接
    x -> RMSNorm -> FFN       -> 残差连接

    Pre-Norm (先归一化再计算) 比 Post-Norm 训练更稳定。
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 注意力子层 + 残差连接
        x = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        # FFN 子层 + 残差连接
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


# ============================================================
# 第七步：完整的 LLM 模型
# ============================================================
class LLM(nn.Module):
    """
    完整的 LLM 模型 (传统 LLaMA 架构)，包含:
    1. Token Embedding (词嵌入)
    2. N 层 Transformer Block
    3. 最终 RMSNorm
    4. 输出线性层 (映射回词表大小)

    注意: 没有 Position Embedding 层，因为位置信息通过 RoPE 在注意力中注入。
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config

        # 词嵌入层: token_id -> 向量
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # N 层 Transformer Block
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        # 最终归一化层
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # 输出头: 隐藏向量 -> 词表 logits
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 预计算 RoPE 频率 (注册为 buffer，不参与梯度更新，但会随模型移动到 GPU)
        head_dim = config.hidden_size // config.num_heads
        freqs_cis = precompute_rope_freqs(
            head_dim, config.max_seq_len, config.rope_theta
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Xavier/He 风格的权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        tokens: (batch, seq_len) 输入 token ids
        返回:  (batch, seq_len, vocab_size) 每个位置的 logits
        """
        batch, seq_len = tokens.shape

        # 1. Token Embedding
        h = self.tok_embeddings(tokens)  # (batch, seq_len, hidden_size)

        # 2. 获取当前序列长度对应的 RoPE 频率
        freqs_cis = self.freqs_cis[:seq_len]  # type: ignore

        # 3. 构建因果掩码 (上三角为 -inf)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1)  # 上三角(不含对角线)设为 -inf
        # 扩展为 (1, 1, seq_len, seq_len) 以便广播到 (batch, heads, seq, seq)
        mask = mask.unsqueeze(0).unsqueeze(0)

        # 4. 逐层通过 Transformer Block
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        # 5. 最终归一化
        h = self.norm(h)

        # 6. 输出 logits
        logits = self.output(h)  # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        自回归生成
        tokens: (batch, seq_len) 初始 prompt 的 token ids
        """
        for _ in range(max_new_tokens):
            # 截断到最大长度
            input_tokens = tokens[:, -self.config.max_seq_len :]
            # 前向得到 logits
            logits = self.forward(input_tokens)
            # 只取最后一个位置的 logits
            logits = logits[:, -1, :]  # (batch, vocab_size)

            if temperature > 0:
                # 温度缩放
                logits = logits / temperature
                # Top-K 采样: 只保留概率最高的 k 个 token
                if top_k > 0:
                    top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < top_k_values[:, [-1]]] = float("-inf")
                # 按概率采样
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪心解码: 取概率最大的 token
                next_token = logits.argmax(dim=-1, keepdim=True)

            # 拼接新 token
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    # 创建一个小型模型用于测试
    config = LLMConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=4,
        intermediate_size=704,
        max_seq_len=512,
    )

    model = LLM(config)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")
    print(f"模型结构:\n{model}")

    # 模拟前向传播
    dummy_tokens = torch.randint(0, config.vocab_size, (2, 64))  # batch=2, seq_len=64
    logits = model(dummy_tokens)
    print(f"\n输入 shape: {dummy_tokens.shape}")
    print(f"输出 shape: {logits.shape}")

    # 模拟生成
    prompt = torch.randint(0, config.vocab_size, (1, 10))  # 1 条 prompt，长度 10
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"\nPrompt 长度: {prompt.shape[1]}")
    print(f"生成后长度: {generated.shape[1]}")
