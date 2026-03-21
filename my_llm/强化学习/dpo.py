"""
DPO (Direct Preference Optimization) 直接偏好优化

DPO 的核心思想:
  PPO-RLHF 太复杂了 (4 个模型、奖励模型训练、PPO 超参调优)，
  DPO 证明可以跳过奖励模型，直接从偏好数据优化策略！

数学推导 (从 RLHF 目标到 DPO):
  1. RLHF 目标: max_π E[R(x,y)] - β * KL(π || π_ref)
  2. 最优策略的闭式解: π*(y|x) = π_ref(y|x) * exp(R(y|x)/β) / Z(x)
  3. 反解奖励: R(y|x) = β * log(π*(y|x) / π_ref(y|x)) + β * log Z(x)
  4. 代入 Bradley-Terry 偏好模型，Z(x) 被消掉！
  5. 最终 DPO 损失:
     L = -log σ(β * [log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x))])

  直觉理解:
  - 让好回答 y_w 的概率相对于参考模型增大
  - 让差回答 y_l 的概率相对于参考模型减小
  - β 控制偏离参考模型的程度

优点:
  - 不需要训练奖励模型
  - 不需要 PPO 的复杂训练循环
  - 只需要 2 个模型 (策略模型 + 冻结的参考模型)
  - 训练稳定性好，超参数少
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ============================================================
# 第一步：DPO 配置
# ============================================================
@dataclass
class DPOConfig:
    """DPO 训练超参数"""

    beta: float = 0.1  # 温度参数 β, 控制偏离参考模型的程度
    #   β 大 → 更保守, 更接近参考模型
    #   β 小 → 更激进, 更多地从偏好数据中学习

    label_smoothing: float = 0.0  # 标签平滑 (防止过拟合, 通常 0~0.1)

    loss_type: str = "sigmoid"  # 损失类型: "sigmoid" (标准DPO), "hinge", "ipo"

    lr: float = 5e-7  # 学习率 (DPO 通常比 SFT 小)
    max_grad_norm: float = 1.0  # 梯度裁剪


# ============================================================
# 第二步：DPO 偏好数据
# ============================================================
@dataclass
class DPOBatch:
    """
    一批 DPO 训练数据

    每条数据包含一个 prompt 和一对 (chosen, rejected) 回答:
    - chosen: 人类标注的更好的回答 y_w
    - rejected: 人类标注的较差的回答 y_l
    """

    # Prompt
    prompt_ids: torch.Tensor  # (batch, prompt_len)
    prompt_mask: torch.Tensor  # (batch, prompt_len)

    # Chosen response (好回答)
    chosen_ids: torch.Tensor  # (batch, chosen_len) prompt + chosen_response
    chosen_mask: torch.Tensor  # (batch, chosen_len)
    chosen_labels: torch.Tensor  # (batch, chosen_len) -100 for prompt tokens

    # Rejected response (差回答)
    rejected_ids: torch.Tensor  # (batch, rejected_len) prompt + rejected_response
    rejected_mask: torch.Tensor  # (batch, rejected_len)
    rejected_labels: torch.Tensor  # (batch, rejected_len) -100 for prompt tokens


# ============================================================
# 第三步：核心算法 — 计算序列级 log 概率
# ============================================================
def compute_sequence_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    计算序列级 log 概率: log π(y|x) = Σ_t log π(y_t | x, y_{<t})

    这是 DPO 的基础计算单元。

    参数:
        model: 语言模型
        input_ids: (batch, seq_len) 完整序列 (prompt + response)
        labels: (batch, seq_len) 标签, prompt 部分为 -100 (不计入)
        attention_mask: (batch, seq_len) 注意力掩码

    返回:
        (batch,) 每个序列的总 log 概率
    """
    # 1. 前向传播得到 logits
    logits = model(input_ids)  # (batch, seq_len, vocab_size)

    # 2. Shift: logits[t] 预测的是 labels[t+1]
    shift_logits = logits[:, :-1, :]  # (batch, seq_len-1, vocab_size)
    shift_labels = labels[:, 1:]  # (batch, seq_len-1)
    shift_mask = attention_mask[:, 1:]  # (batch, seq_len-1)

    # 3. 逐 token 计算 log prob
    log_probs = F.log_softmax(shift_logits, dim=-1)  # (batch, seq_len-1, vocab_size)

    # 4. 取对应 label 的 log prob (忽略 -100 位置)
    valid_mask = shift_labels != -100
    # 将 -100 替换为 0 (避免 gather 越界)，后面会用 mask 过滤
    safe_labels = shift_labels.clamp(min=0)
    per_token_logprobs = log_probs.gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)  # (batch, seq_len-1)

    # 5. 只对 response 部分求和 (prompt 部分 labels=-100, 被 mask 掉)
    per_token_logprobs = per_token_logprobs * valid_mask * shift_mask

    # 6. 求和得到序列级 log prob: Σ_t log π(y_t | y_{<t}, x)
    sequence_logprobs = per_token_logprobs.sum(dim=-1)  # (batch,)

    return sequence_logprobs


# ============================================================
# 第四步：DPO 损失函数
# ============================================================
class DPOLoss(nn.Module):
    """
    DPO 损失函数

    标准 DPO 损失:
      L = -log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))

    变形理解:
      令 h = β * (Δ_w - Δ_l)
      其中 Δ_w = log π_θ(y_w|x) - log π_ref(y_w|x)  # 好回答的 log ratio
           Δ_l = log π_θ(y_l|x) - log π_ref(y_l|x)  # 差回答的 log ratio

      L = -log σ(h)

    梯度分析:
      ∂L/∂θ = -β * σ(-h) * [∇logπ(y_w|x) - ∇logπ(y_l|x)]

      当 h 很负 (模型犯错): σ(-h) ≈ 1, 梯度大, 强力纠正
      当 h 很正 (模型正确): σ(-h) ≈ 0, 梯度小, 已经学好了

    支持三种变体:
      - sigmoid: 标准 DPO
      - hinge: 铰链损失变体 (来自 RSO 论文)
      - ipo: IPO (Identity Preference Optimization)
    """

    def __init__(self, config: DPOConfig):
        super().__init__()
        self.beta = config.beta
        self.label_smoothing = config.label_smoothing
        self.loss_type = config.loss_type

    def forward(
        self,
        policy_chosen_logprobs: torch.Tensor,
        policy_rejected_logprobs: torch.Tensor,
        ref_chosen_logprobs: torch.Tensor,
        ref_rejected_logprobs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        参数 (all shapes: (batch,)):
            policy_chosen_logprobs: log π_θ(y_w | x)
            policy_rejected_logprobs: log π_θ(y_l | x)
            ref_chosen_logprobs: log π_ref(y_w | x)
            ref_rejected_logprobs: log π_ref(y_l | x)
        """
        # 1. 计算 log ratio (相对于参考模型的偏移)
        #    Δ_w = log π_θ(y_w) - log π_ref(y_w) : 好回答的概率变化
        #    Δ_l = log π_θ(y_l) - log π_ref(y_l) : 差回答的概率变化
        chosen_logratios = policy_chosen_logprobs - ref_chosen_logprobs
        rejected_logratios = policy_rejected_logprobs - ref_rejected_logprobs

        # 2. 计算隐式奖励差: h = β * (Δ_w - Δ_l)
        #    正数 → 模型正确地偏好 chosen
        #    负数 → 模型错误地偏好 rejected
        logits = self.beta * (chosen_logratios - rejected_logratios)

        # 3. 计算损失
        if self.loss_type == "sigmoid":
            # 标准 DPO: L = -log σ(h)
            if self.label_smoothing > 0:
                # 标签平滑: 防止过度自信
                # L = -(1-α) * log σ(h) - α * log σ(-h)
                losses = (
                    -(1 - self.label_smoothing) * F.logsigmoid(logits)
                    - self.label_smoothing * F.logsigmoid(-logits)
                )
            else:
                losses = -F.logsigmoid(logits)

        elif self.loss_type == "hinge":
            # 铰链损失: L = max(0, 1 - h)
            # 只要 h > 1 就不再优化, 避免过拟合
            losses = torch.relu(1.0 - logits)

        elif self.loss_type == "ipo":
            # IPO: L = (h - 1/(2β))^2
            # 正则化效果更强, 避免 DPO 的过拟合问题
            losses = (logits - 1.0 / (2 * self.beta)).pow(2)

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        loss = losses.mean()

        # 4. 计算指标
        # 准确率: 模型是否正确地给 chosen 更高的概率
        accuracy = (logits > 0).float().mean()

        # 隐式奖励 (DPO 论文中的 "implicit reward")
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        reward_margin = (chosen_rewards - rejected_rewards).mean()

        return {
            "loss": loss,
            "accuracy": accuracy.detach(),
            "chosen_rewards": chosen_rewards.mean().detach(),
            "rejected_rewards": rejected_rewards.mean().detach(),
            "reward_margin": reward_margin.detach(),
            "logits_mean": logits.mean().detach(),
        }


# ============================================================
# 第五步：DPO 训练器
# ============================================================
class DPOTrainer:
    """
    DPO 训练器

    相比 PPO 的优势:
      - 只需 2 个模型 (策略 + 参考), 不需要奖励模型和 Critic
      - 标准的监督学习训练循环, 无需采样和 rollout
      - 训练稳定, 超参数少 (主要就是 β)

    训练循环:
    for each batch of (prompt, chosen, rejected):
        1. 策略模型和参考模型分别计算 chosen/rejected 的 log prob
        2. 计算 DPO 损失
        3. 梯度更新策略模型
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        config: DPOConfig,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.config = config

        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            policy_model.parameters(), lr=config.lr
        )
        self.loss_fn = DPOLoss(config)

    def train_step(self, batch: DPOBatch) -> dict[str, float]:
        """
        一步 DPO 训练

        核心就这几行:
        1. 算 4 个 log prob
        2. 算 loss
        3. 反向传播
        """
        self.policy_model.train()

        # 1. 策略模型: 计算 chosen 和 rejected 的 log prob
        policy_chosen_logprobs = compute_sequence_logprobs(
            self.policy_model,
            batch.chosen_ids,
            batch.chosen_labels,
            batch.chosen_mask,
        )
        policy_rejected_logprobs = compute_sequence_logprobs(
            self.policy_model,
            batch.rejected_ids,
            batch.rejected_labels,
            batch.rejected_mask,
        )

        # 2. 参考模型: 计算 chosen 和 rejected 的 log prob (不需要梯度)
        with torch.no_grad():
            ref_chosen_logprobs = compute_sequence_logprobs(
                self.ref_model,
                batch.chosen_ids,
                batch.chosen_labels,
                batch.chosen_mask,
            )
            ref_rejected_logprobs = compute_sequence_logprobs(
                self.ref_model,
                batch.rejected_ids,
                batch.rejected_labels,
                batch.rejected_mask,
            )

        # 3. 计算 DPO 损失
        metrics = self.loss_fn(
            policy_chosen_logprobs,
            policy_rejected_logprobs,
            ref_chosen_logprobs,
            ref_rejected_logprobs,
        )

        # 4. 梯度更新
        self.optimizer.zero_grad()
        metrics["loss"].backward()
        nn.utils.clip_grad_norm_(
            self.policy_model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=== DPO (Direct Preference Optimization) 测试 ===\n")

    # 模拟一个简单的语言模型
    class SimpleLM(nn.Module):
        def __init__(self, vocab_size=100, hidden_size=64, num_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            self.lm_head = nn.Linear(hidden_size, vocab_size)

        def forward(self, x):
            h = self.embedding(x)
            h = self.layers(h)
            return self.lm_head(h)

    # 1. 测试序列 log prob 计算
    print("--- 序列 log prob 计算 ---")
    model = SimpleLM()
    input_ids = torch.randint(0, 100, (4, 20))
    labels = input_ids.clone()
    labels[:, :5] = -100  # prompt 部分不计入
    mask = torch.ones_like(input_ids)

    seq_logprobs = compute_sequence_logprobs(model, input_ids, labels, mask)
    print(f"  序列 log prob shape: {seq_logprobs.shape}")
    print(f"  序列 log prob: {seq_logprobs.tolist()}")

    # 2. 测试三种 DPO 损失变体
    print("\n--- DPO 损失变体 ---")
    for loss_type in ["sigmoid", "hinge", "ipo"]:
        config = DPOConfig(beta=0.1, loss_type=loss_type)
        loss_fn = DPOLoss(config)

        # 模拟 log probs
        policy_chosen = torch.tensor([-10.0, -12.0, -8.0, -15.0])
        policy_rejected = torch.tensor([-15.0, -10.0, -20.0, -11.0])
        ref_chosen = torch.tensor([-11.0, -13.0, -9.0, -16.0])
        ref_rejected = torch.tensor([-16.0, -11.0, -21.0, -12.0])

        metrics = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        print(f"\n  [{loss_type}] DPO 损失:")
        for k, v in metrics.items():
            print(f"    {k}: {v.item():.4f}")

    # 3. 测试 DPO 训练器
    print("\n--- DPO 训练器 ---")
    policy = SimpleLM()
    ref = SimpleLM()
    ref.load_state_dict(policy.state_dict())  # 参考模型初始化为策略模型的副本

    config = DPOConfig(beta=0.1, lr=1e-4)
    trainer = DPOTrainer(policy, ref, config)

    # 构造假数据
    batch = DPOBatch(
        prompt_ids=torch.randint(0, 100, (4, 5)),
        prompt_mask=torch.ones(4, 5),
        chosen_ids=torch.randint(0, 100, (4, 20)),
        chosen_mask=torch.ones(4, 20),
        chosen_labels=torch.cat([
            torch.full((4, 5), -100, dtype=torch.long),
            torch.randint(0, 100, (4, 15)),
        ], dim=1),
        rejected_ids=torch.randint(0, 100, (4, 20)),
        rejected_mask=torch.ones(4, 20),
        rejected_labels=torch.cat([
            torch.full((4, 5), -100, dtype=torch.long),
            torch.randint(0, 100, (4, 15)),
        ], dim=1),
    )

    # 训练几步
    for step in range(3):
        metrics = trainer.train_step(batch)
        print(f"  Step {step}: loss={metrics['loss']:.4f}, "
              f"accuracy={metrics['accuracy']:.2f}, "
              f"margin={metrics['reward_margin']:.4f}")

    print("\n✓ DPO 所有模块测试通过!")
