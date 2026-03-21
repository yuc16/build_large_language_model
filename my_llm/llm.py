import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


# ============================================================
# 第一步：模型配置 — Qwen3.5 架构
# ============================================================
@dataclass
class LLMConfig:
    """
    Qwen3.5 模型超参数配置
    核心创新: 混合 Gated DeltaNet (线性注意力) + GQA (全注意力) 架构
    """

    vocab_size: int = 248320  # 词表大小 (Qwen3.5 统一词表)
    hidden_size: int = 512  # 隐藏层维度
    num_layers: int = 8  # 总层数
    # --- 全注意力层 (GQA) 参数 ---
    num_heads: int = 8  # Q 注意力头数
    num_kv_heads: int = 2  # KV 头数 (GQA, 远少于 Q 头)
    head_dim: int = 256  # 每个注意力头的维度
    # --- DeltaNet 线性注意力层参数 ---
    num_deltanet_v_heads: int = 16  # DeltaNet V 头数
    num_deltanet_qk_heads: int = 16  # DeltaNet QK 头数
    deltanet_head_dim: int = 128  # DeltaNet 每个头的维度
    conv_kernel_size: int = 4  # 因果 Conv1D 核大小
    # --- FFN ---
    intermediate_size: int = 1408  # FFN 中间层维度 (dense 模型或 MoE 共享专家用)
    # --- MoE (Mixture of Experts) ---
    use_moe: bool = False  # 是否启用 MoE (False=dense, True=MoE)
    num_experts: int = 256  # 路由专家总数 (Qwen3.5: 256 或 512)
    num_experts_per_tok: int = 8  # 每个 token 激活的路由专家数
    num_shared_experts: int = 1  # 共享专家数 (每个 token 都会经过)
    expert_intermediate_size: int = 512  # 每个专家的 FFN 中间层维度 (细粒度: 很小!)
    router_aux_loss_coef: float = 0.01  # 负载均衡辅助损失系数
    # --- 层排布 ---
    full_attention_interval: int = 4  # 每隔 N 层一层全注意力 (其余为 DeltaNet)
    # --- 归一化 & 位置编码 ---
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0  # 10M, 比传统 10K 大 1000 倍
    partial_rotary_factor: float = 0.25  # 只对 25% 的维度应用 RoPE
    max_seq_len: int = 4096


# ============================================================
# 第二步：GatedRMSNorm — Qwen3.5 的门控归一化
# ============================================================
class GatedRMSNorm(nn.Module):
    """
    门控 RMSNorm: RMSNorm + SiLU 门控
    相比普通 RMSNorm 多了一个门控分支，增加了表达能力。
    公式: output = RMSNorm(x) * SiLU(x @ W_gate)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class ZeroCenteredRMSNorm(nn.Module):
    """
    零中心 RMSNorm: weight 初始化为 0 而非 1
    用于注意力层的 QK 归一化，训练初期效果接近恒等映射，
    避免了 Qwen3 中 QK-Norm 权重异常增长的问题。
    公式: output = x / sqrt(mean(x^2) + eps) * (1 + weight)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))  # 零初始化！
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # (1 + weight) 使得初始时接近恒等映射
        return (x / rms) * (1.0 + self.weight)


# ============================================================
# 第三步：RoPE — 部分旋转位置编码
# ============================================================
def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000000.0,
    partial_rotary_factor: float = 0.25,
) -> torch.Tensor:
    """
    预计算 RoPE 频率向量。
    Qwen3.5 的创新: 只对 head_dim 的前 25% 维度应用 RoPE (partial_rotary_factor=0.25)
    其余 75% 维度不做旋转，保持原始信息。
    """
    # 只对部分维度计算旋转频率
    rotary_dim = int(head_dim * partial_rotary_factor)
    # 确保 rotary_dim 是偶数
    rotary_dim = rotary_dim - (rotary_dim % 2)

    freqs = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)  # (max_seq_len, rotary_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis


def apply_partial_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    部分 RoPE: 只旋转前 rotary_dim 个维度，其余维度不变。
    x shape: (batch, seq_len, num_heads, head_dim)
    """
    rotary_dim = freqs_cis.shape[-1] * 2  # 每个复数对应 2 个实数维度

    # 分割: 前 rotary_dim 维做旋转，后面的保持不变
    x_rot = x[..., :rotary_dim]  # 需要旋转的部分
    x_pass = x[..., rotary_dim:]  # 直通的部分

    # 旋转部分转为复数
    x_complex = torch.view_as_complex(x_rot.float().reshape(*x_rot.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, rotary_dim//2)
    x_rotated = x_complex * freqs_cis
    x_rot_out = torch.view_as_real(x_rotated).reshape(*x_rot.shape)

    # 拼接旋转后的和直通的
    return torch.cat([x_rot_out.type_as(x), x_pass], dim=-1)


# ============================================================
# 第四步：GQA 全注意力层 (每 4 层用 1 次)
# ============================================================
class FullAttention(nn.Module):
    """
    标准 GQA 全注意力 + Zero-Centered RMSNorm QK归一化 + 部分 RoPE
    Qwen3.5 中每隔 full_attention_interval 层使用一次。
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_groups = self.num_heads // self.num_kv_heads

        # Q/K/V 投影
        self.wq = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        # Zero-Centered RMSNorm 用于稳定 QK 的注意力分数
        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # 1. 投影
        q = self.wq(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # 2. QK 归一化 (Zero-Centered RMSNorm)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 3. 部分 RoPE (只旋转 25% 的维度)
        q = apply_partial_rope(q, freqs_cis)
        k = apply_partial_rope(k, freqs_cis)

        # 4. GQA 扩展 KV 头
        if self.num_groups > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.num_groups, -1)
            k = k.reshape(batch, seq_len, self.num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.num_groups, -1)
            v = v.reshape(batch, seq_len, self.num_heads, self.head_dim)

        # 5. 转置并计算注意力
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        # 6. 合并头并输出投影
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(output)


# ============================================================
# 第五步：Gated DeltaNet — 线性注意力 (核心创新！)
# ============================================================
class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet: Qwen3.5 的核心创新组件
    用线性注意力替代 softmax 注意力，75% 的层使用此模块。

    核心思想:
    1. 维护一个固定大小的记忆矩阵 S: (d_k, d_v)
    2. 每个 token 通过 Delta Rule 更新记忆: 先擦除旧信息，再写入新信息
    3. 用指数门控 (beta) 控制记忆的衰减与更新强度
    4. 用因果 Conv1D 捕获局部上下文

    相比全注意力:
    - 生成时内存 O(1) vs O(seq_len) — 不需要 KV Cache！
    - 训练时计算 O(n*d^2) vs O(n^2*d)
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.num_v_heads = config.num_deltanet_v_heads
        self.num_qk_heads = config.num_deltanet_qk_heads
        self.head_dim = config.deltanet_head_dim
        self.hidden_size = config.hidden_size

        # Q, K, V 投影
        self.wq = nn.Linear(
            config.hidden_size, self.num_qk_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(
            config.hidden_size, self.num_qk_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            config.hidden_size, self.num_v_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(
            self.num_v_heads * self.head_dim, config.hidden_size, bias=False
        )

        # 门控参数: 学习每个 token 的记忆衰减率
        # beta = sigmoid(x @ w_beta) 控制"遗忘多少旧记忆，写入多少新信息"
        self.w_beta = nn.Linear(config.hidden_size, self.num_qk_heads, bias=False)

        # 因果 Conv1D: 捕获局部上下文 (kernel_size=4)
        # DeltaNet 没有显式位置编码，靠 Conv1D 感知局部位置
        self.conv_q = nn.Conv1d(
            self.num_qk_heads * self.head_dim,
            self.num_qk_heads * self.head_dim,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size - 1,  # 因果填充
            groups=self.num_qk_heads * self.head_dim,  # depthwise
            bias=False,
        )
        self.conv_k = nn.Conv1d(
            self.num_qk_heads * self.head_dim,
            self.num_qk_heads * self.head_dim,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size - 1,
            groups=self.num_qk_heads * self.head_dim,
            bias=False,
        )
        self.conv_v = nn.Conv1d(
            self.num_v_heads * self.head_dim,
            self.num_v_heads * self.head_dim,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size - 1,
            groups=self.num_v_heads * self.head_dim,
            bias=False,
        )

        # 输出门控: 控制最终输出
        self.w_og = nn.Linear(
            config.hidden_size, self.num_v_heads * self.head_dim, bias=False
        )

        # QK 头和 V 头数量可能不同，需要分组匹配
        self.qk_per_v = self.num_qk_heads // self.num_v_heads

    def _causal_conv1d(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        """因果卷积: 只看当前和之前的 token，不看未来"""
        # x: (batch, channels, seq_len)
        out = conv(x)
        # 截断右侧填充，确保因果性
        return out[..., : x.shape[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        DeltaNet 前向传播
        x: (batch, seq_len, hidden_size)

        Delta Rule 更新公式:
          S_t = (I - beta_t * k_t * k_t^T) * S_{t-1} + beta_t * v_t * k_t^T
          o_t = S_t * q_t

        直觉理解:
          - k_t * k_t^T * S: 用当前 key 检索旧记忆中与之相关的部分并擦除
          - v_t * k_t^T: 写入新的 key-value 关联
          - beta_t: 门控，控制更新强度 (0=完全保留旧记忆, 1=激进更新)
        """
        batch, seq_len, _ = x.shape

        # 1. 投影
        q = self.wq(x)  # (batch, seq_len, num_qk_heads * head_dim)
        k = self.wk(x)
        v = self.wv(x)

        # 2. 因果 Conv1D 注入局部上下文
        q = self._causal_conv1d(self.conv_q, q.transpose(1, 2)).transpose(1, 2)
        k = self._causal_conv1d(self.conv_k, k.transpose(1, 2)).transpose(1, 2)
        v = self._causal_conv1d(self.conv_v, v.transpose(1, 2)).transpose(1, 2)

        # 激活: Q 用 SiLU, K 用 SiLU
        q = F.silu(q)
        k = F.silu(k)

        # 3. 拆分多头
        q = q.view(batch, seq_len, self.num_qk_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_qk_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_v_heads, self.head_dim)

        # 4. L2 归一化 Q 和 K (稳定训练)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # 5. 计算门控 beta
        beta = torch.sigmoid(self.w_beta(x))  # (batch, seq_len, num_qk_heads)
        beta = beta.unsqueeze(-1)  # (batch, seq_len, num_qk_heads, 1)

        # 6. Delta Rule 循环 (训练时按 chunk 并行，这里用简单循环演示原理)
        # 记忆矩阵 S: (batch, num_qk_heads, head_dim, head_dim)
        S = torch.zeros(
            batch,
            self.num_qk_heads,
            self.head_dim,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )
        outputs = []

        for t in range(seq_len):
            q_t = q[:, t]  # (batch, num_qk_heads, head_dim)
            k_t = k[:, t]  # (batch, num_qk_heads, head_dim)
            beta_t = beta[:, t]  # (batch, num_qk_heads, 1)

            # Delta Rule 更新:
            # 擦除: 从记忆中移除与当前 key 相关的旧信息
            # 写入: 写入新的 key-value 关联
            # S = (1 - beta * k * k^T) * S + beta * v_grouped * k^T

            # 检索旧记忆中与 k_t 相关的部分
            k_t_expanded = k_t.unsqueeze(-1)  # (batch, heads, dim, 1)
            k_t_row = k_t.unsqueeze(-2)  # (batch, heads, 1, dim)

            # 擦除 + 写入
            erase = beta_t.unsqueeze(-1) * (
                k_t_expanded @ k_t_row
            )  # (batch, heads, dim, dim)
            S = S * (1 - erase)

            # 将 V 头映射到 QK 头 (如果数量不同)
            v_t = v[:, t]  # (batch, num_v_heads, head_dim)
            if self.qk_per_v > 1:
                v_t = v_t.unsqueeze(2).expand(-1, -1, self.qk_per_v, -1)
                v_t = v_t.reshape(batch, self.num_qk_heads, self.head_dim)

            v_t_expanded = v_t.unsqueeze(-1)  # (batch, heads, dim, 1)
            write = beta_t.unsqueeze(-1) * (v_t_expanded @ k_t_row)
            S = S + write

            # 读取: 用 q 从记忆中检索
            o_t = (S @ q_t.unsqueeze(-1)).squeeze(-1)  # (batch, num_qk_heads, head_dim)

            # 将 QK 头聚合回 V 头
            if self.qk_per_v > 1:
                o_t = o_t.view(batch, self.num_v_heads, self.qk_per_v, self.head_dim)
                o_t = o_t.sum(dim=2)  # (batch, num_v_heads, head_dim)

            outputs.append(o_t)

        # 7. 拼接所有时间步
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, num_v_heads, head_dim)
        output = output.reshape(
            batch, seq_len, -1
        )  # (batch, seq_len, num_v_heads * head_dim)

        # 8. 输出门控
        gate = F.silu(self.w_og(x))
        output = output * gate

        # 9. 输出投影
        return self.wo(output)


# ============================================================
# 第六步：SwiGLU FFN (与之前类似)
# ============================================================
class FeedForward(nn.Module):
    """SwiGLU 前馈网络: output = (SiLU(x @ W1) * (x @ W3)) @ W2"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ============================================================
# 第 6.5 步：MoE — 细粒度混合专家 (Qwen3.5 大模型核心)
# ============================================================
class MoEGate(nn.Module):
    """
    MoE 路由门控: 决定每个 token 分配给哪些专家
    使用 Top-K 路由策略 + 负载均衡辅助损失

    Qwen3.5 的特点:
    - 细粒度 MoE: 大量小专家 (256~512个)，每个专家的 FFN 很小
    - 好处: 更灵活的专家组合，更好的知识分配
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # 门控投影: hidden_size -> num_experts
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x: (batch * seq_len, hidden_size)
        返回: (router_weights, selected_experts)
            router_weights: (batch*seq_len, num_experts_per_tok) 归一化权重
            selected_experts: (batch*seq_len, num_experts_per_tok) 选中的专家索引
        """
        # 1. 计算每个 token 对每个专家的分数
        logits = self.gate(x)  # (batch*seq_len, num_experts)

        # 2. Top-K 选择: 每个 token 选出分数最高的 K 个专家
        topk_weights, topk_indices = torch.topk(
            logits, self.num_experts_per_tok, dim=-1
        )

        # 3. Softmax 归一化选中专家的权重 (只在 top-k 之间竞争)
        topk_weights = F.softmax(topk_weights, dim=-1)

        return topk_weights, topk_indices


class MoEFeedForward(nn.Module):
    """
    MoE 前馈网络: 路由专家 + 共享专家

    结构:
    input ──┬── Router Gate ──> Top-K 路由专家 (加权求和) ──┐
            │                                                ├── 相加 → output
            └── 共享专家 (所有 token 都经过) ────────────────┘

    Qwen3.5 MoE 特点:
    - 路由专家: 256~512 个小专家，每个 FFN 中间维度只有 512~1024
    - 共享专家: 1 个，使用标准 intermediate_size
    - 每个 token 激活 8~10 个路由专家 + 1 个共享专家
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        # 路由门控
        self.gate = MoEGate(config)

        # 路由专家: 大量小型 SwiGLU FFN
        self.experts = nn.ModuleList(
            [
                FeedForward(config.hidden_size, config.expert_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )

        # 共享专家: 每个 token 都会经过，使用较大的 intermediate_size
        self.shared_experts = nn.ModuleList(
            [
                FeedForward(config.hidden_size, config.intermediate_size)
                for _ in range(config.num_shared_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_size)
        """
        batch, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)  # (batch*seq_len, hidden_size)

        # 1. 路由: 为每个 token 选择 top-k 个专家
        router_weights, selected_experts = self.gate(x_flat)
        # router_weights: (num_tokens, k), selected_experts: (num_tokens, k)

        # 2. 路由专家计算 (逐专家处理，合并同一专家的 token 以提高效率)
        routed_output = torch.zeros_like(x_flat)

        for expert_idx in range(len(self.experts)):
            # 找到所有选了这个专家的 (token, slot) 对
            token_indices, slot_indices = torch.where(selected_experts == expert_idx)

            if token_indices.numel() == 0:
                continue  # 没有 token 选这个专家

            # 提取这些 token 的输入
            expert_input = x_flat[token_indices]  # (num_selected, hidden_size)

            # 过专家 FFN
            expert_output = self.experts[expert_idx](expert_input)

            # 乘以路由权重并累加
            weights = router_weights[token_indices, slot_indices].unsqueeze(-1)
            routed_output.index_add_(0, token_indices, expert_output * weights)

        # 3. 共享专家: 所有 token 都经过
        shared_output = sum(expert(x_flat) for expert in self.shared_experts)

        # 4. 路由输出 + 共享输出
        output = routed_output + shared_output
        return output.view(batch, seq_len, hidden_size)


# ============================================================
# 第七步：Transformer Block — 混合层
# ============================================================
class TransformerBlock(nn.Module):
    """
    Qwen3.5 的 Transformer 层，根据层索引选择不同的注意力机制:
    - 每 full_attention_interval 层 (如每4层): 使用 FullAttention (GQA + RoPE)
    - 其余层: 使用 GatedDeltaNet (线性注意力)

    排布模式: [DeltaNet, DeltaNet, DeltaNet, FullAttn, DeltaNet, DeltaNet, DeltaNet, FullAttn, ...]
    """

    def __init__(self, config: LLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        # 判断当前层是全注意力还是 DeltaNet
        self.is_full_attention = (layer_idx + 1) % config.full_attention_interval == 0

        if self.is_full_attention:
            self.token_mixer = FullAttention(config)
        else:
            self.token_mixer = GatedDeltaNet(config)

        # FFN: 根据配置选择 Dense FFN 或 MoE FFN
        if config.use_moe:
            self.feed_forward = MoEFeedForward(config)
        else:
            self.feed_forward = FeedForward(
                config.hidden_size, config.intermediate_size
            )
        self.input_norm = GatedRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = GatedRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 注意力子层 + 残差
        if self.is_full_attention:
            x = x + self.token_mixer(self.input_norm(x), freqs_cis, mask)
        else:
            x = x + self.token_mixer(self.input_norm(x))
        # FFN 子层 + 残差
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


# ============================================================
# 第八步：完整的 Qwen3.5 LLM
# ============================================================
class LLM(nn.Module):
    """
    Qwen3.5 完整模型

    架构流程:
    Token Embedding
        ↓
    [DeltaNet Block] × 3  ←  线性注意力，O(1) 生成内存
    [FullAttn Block] × 1  ←  全注意力 + RoPE，捕获全局依赖
        ↓ (重复 N/4 组)
    GatedRMSNorm
        ↓
    Output Linear → logits
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config

        # 词嵌入
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # 混合层: 根据 layer_idx 自动选择 DeltaNet 或 FullAttention
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.num_layers)]
        )

        # 最终归一化
        self.norm = GatedRMSNorm(config.hidden_size, config.rms_norm_eps)

        # 输出头
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 预计算部分 RoPE 频率 (只用于 FullAttention 层)
        freqs_cis = precompute_rope_freqs(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
            config.partial_rotary_factor,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        tokens: (batch, seq_len) → logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len = tokens.shape

        # 1. Token Embedding
        h = self.tok_embeddings(tokens)

        # 2. RoPE 频率 (只给 FullAttention 层用)
        freqs_cis = self.freqs_cis[:seq_len]  # type: ignore

        # 3. 因果掩码 (只给 FullAttention 层用)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)

        # 4. 逐层前向
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        # 5. 归一化 + 输出
        h = self.norm(h)
        return self.output(h)

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """自回归生成"""
        for _ in range(max_new_tokens):
            input_tokens = tokens[:, -self.config.max_seq_len :]
            logits = self.forward(input_tokens)
            logits = logits[:, -1, :]

            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < top_k_values[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    # 小型 Qwen3.5 用于测试
    config = LLMConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=8,  # 8 层: 6 层 DeltaNet + 2 层 FullAttn
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        num_deltanet_v_heads=8,
        num_deltanet_qk_heads=8,
        deltanet_head_dim=32,
        intermediate_size=704,
        full_attention_interval=4,
        max_seq_len=512,
    )

    model = LLM(config)

    # 打印层排布
    print("=== Qwen3.5 层排布 ===")
    for i, layer in enumerate(model.layers):
        layer_type = (
            "FullAttention (GQA + RoPE)"
            if layer.is_full_attention
            else "GatedDeltaNet (线性注意力)"
        )
        print(f"  Layer {i}: {layer_type}")

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params / 1e6:.2f}M")

    # 前向传播测试
    dummy_tokens = torch.randint(0, config.vocab_size, (2, 64))
    logits = model(dummy_tokens)
    print(f"\n输入 shape: {dummy_tokens.shape}")
    print(f"输出 shape: {logits.shape}")

    # 生成测试
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"\nPrompt 长度: {prompt.shape[1]}")
    print(f"生成后长度: {generated.shape[1]}")

    # ==================== MoE 模式测试 ====================
    print("\n" + "=" * 60)
    print("=== Qwen3.5-MoE 模式测试 ===")
    moe_config = LLMConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=8,
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        num_deltanet_v_heads=8,
        num_deltanet_qk_heads=8,
        deltanet_head_dim=32,
        intermediate_size=704,  # 共享专家用
        full_attention_interval=4,
        max_seq_len=512,
        # --- MoE 参数 ---
        use_moe=True,
        num_experts=16,  # 测试用 16 个路由专家
        num_experts_per_tok=4,  # 每个 token 激活 4 个
        num_shared_experts=1,  # 1 个共享专家
        expert_intermediate_size=128,  # 每个小专家的 FFN 维度
    )

    moe_model = LLM(moe_config)

    total_params = sum(p.numel() for p in moe_model.parameters())
    active_note = (
        f"每个 token 激活 {moe_config.num_experts_per_tok}/{moe_config.num_experts} "
        f"路由专家 + {moe_config.num_shared_experts} 共享专家"
    )
    print(f"总参数量: {total_params / 1e6:.2f}M ({active_note})")

    # 前向测试
    dummy_tokens = torch.randint(0, moe_config.vocab_size, (2, 64))
    logits = moe_model(dummy_tokens)
    print(f"输入 shape: {dummy_tokens.shape}")
    print(f"输出 shape: {logits.shape}")
