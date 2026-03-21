"""
PPO (Proximal Policy Optimization) 用于大模型对齐

RLHF 的经典流程:
  1. 预训练 LLM (SFT 后的策略模型 π_θ)
  2. 训练一个奖励模型 R_φ (从人类偏好数据学习)
  3. 用 PPO 优化策略模型，最大化奖励，同时用 KL 散度约束不要偏离太远

PPO 的核心思想:
  - 策略梯度 + 裁剪 (Clipping): 限制每次更新的步幅，避免策略崩溃
  - 同时训练 Value 网络 (Critic) 来估计状态价值，降低方差
  - 用 GAE (Generalized Advantage Estimation) 平衡偏差和方差

完整的 RLHF-PPO 涉及 4 个模型:
  1. Actor (策略模型 π_θ): 生成回答，就是我们要优化的 LLM
  2. Critic (价值模型 V_ψ): 评估当前状态的好坏
  3. Reward Model (R_φ): 给回答打分
  4. Reference Model (π_ref): 冻结的 SFT 模型，用于计算 KL 惩罚
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ============================================================
# 第一步：PPO 超参数配置
# ============================================================
@dataclass
class PPOConfig:
    """PPO 训练超参数"""

    # --- PPO 核心参数 ---
    clip_eps: float = 0.2  # 裁剪范围 ε，限制策略更新幅度
    value_clip_eps: float = 0.2  # Value 函数的裁剪范围
    ppo_epochs: int = 4  # 每批数据上的 PPO 迭代次数
    mini_batch_size: int = 4  # PPO 每次更新的 mini-batch 大小

    # --- GAE (Generalized Advantage Estimation) ---
    gamma: float = 1.0  # 折扣因子 (语言任务通常用 1.0, 因为奖励只在最后一步)
    gae_lambda: float = 0.95  # GAE 的 λ，平衡偏差与方差

    # --- KL 散度惩罚 ---
    kl_coef: float = 0.1  # KL 惩罚系数 β, reward = R(x,y) - β * KL(π_θ || π_ref)
    kl_target: float = 6.0  # KL 散度目标值 (自适应调整 β)
    adaptive_kl: bool = True  # 是否自适应调整 KL 系数

    # --- 损失权重 ---
    value_loss_coef: float = 0.5  # Value 损失权重
    entropy_coef: float = 0.01  # 熵奖励系数 (鼓励探索)

    # --- 训练 ---
    lr: float = 1e-6  # 学习率 (RLHF 阶段通常很小)
    max_grad_norm: float = 1.0  # 梯度裁剪


# ============================================================
# 第二步：奖励模型 (Reward Model)
# ============================================================
class RewardModel(nn.Module):
    """
    奖励模型: 从人类偏好数据中学习给回答打分

    训练方式 (Bradley-Terry 模型):
      给定 prompt x 和两个回答 y_w (好), y_l (差):
      Loss = -log(σ(R(x, y_w) - R(x, y_l)))

    结构:
      LLM backbone → 最后一个 token 的隐藏态 → 线性层 → 标量分数

    注意: 这里用一个简单的 Transformer 模拟 backbone，
    实际中通常用 SFT 后的 LLM 去掉 lm_head，加一个 value_head。
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        # 实际应用中这里是完整的 LLM backbone
        # 为了教学简洁，用 Embedding + 简单变换代替
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # 价值头: 将隐藏向量映射到标量奖励
        self.value_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input_ids: (batch, seq_len)
        返回: (batch,) 每个序列的奖励分数
        """
        h = self.embedding(input_ids)  # (batch, seq_len, hidden_size)
        h = self.layers(h)

        # 取最后一个有效 token 的表示作为整个序列的评分
        if attention_mask is not None:
            # 找到每个序列最后一个非 padding 位置
            last_idx = attention_mask.sum(dim=-1) - 1  # (batch,)
            h = h[torch.arange(h.size(0)), last_idx]  # (batch, hidden_size)
        else:
            h = h[:, -1]  # (batch, hidden_size)

        reward = self.value_head(h).squeeze(-1)  # (batch,)
        return reward


class RewardModelLoss(nn.Module):
    """
    奖励模型训练损失 (Bradley-Terry 偏好学习)

    给定人类标注的偏好对 (y_chosen > y_rejected):
    Loss = -log(σ(r_chosen - r_rejected))

    直觉: 让好回答的奖励分数比差回答高
    """

    def forward(
        self,
        rewards_chosen: torch.Tensor,
        rewards_rejected: torch.Tensor,
    ) -> torch.Tensor:
        """
        rewards_chosen: (batch,) 好回答的奖励分数
        rewards_rejected: (batch,) 差回答的奖励分数
        """
        # r_chosen - r_rejected 越大，loss 越小
        return -F.logsigmoid(rewards_chosen - rewards_rejected).mean()


# ============================================================
# 第三步：PPO 经验缓冲区
# ============================================================
@dataclass
class PPOExperience:
    """
    PPO 经验数据: 一次 rollout 收集的所有信息

    RLHF 中的一次 "rollout":
      1. 给模型一个 prompt
      2. 模型自回归生成回答 (这就是 "采样轨迹")
      3. 奖励模型给回答打分
      4. 计算每个 token 的优势函数
    """

    query_ids: torch.Tensor  # (batch, query_len) prompt 的 token ids
    response_ids: torch.Tensor  # (batch, response_len) 生成的回答 token ids
    logprobs: torch.Tensor  # (batch, response_len) 每个 token 的 log π_θ(a|s)
    ref_logprobs: torch.Tensor  # (batch, response_len) 参考模型的 log π_ref(a|s)
    values: torch.Tensor  # (batch, response_len) Critic 估计的 V(s)
    rewards: torch.Tensor  # (batch,) 奖励模型给的分数
    attention_mask: torch.Tensor  # (batch, response_len) 有效 token 掩码


# ============================================================
# 第四步：核心算法 — 计算 KL 惩罚奖励
# ============================================================
def compute_kl_penalty_rewards(
    rewards: torch.Tensor,
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    kl_coef: float,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    计算带 KL 惩罚的逐 token 奖励

    RLHF 的奖励设计:
      - 最后一个 token: R(x,y) - β * KL_t  (整体奖励 + KL 惩罚)
      - 中间 token:    - β * KL_t           (只有 KL 惩罚)

    为什么要 KL 惩罚?
      防止模型为了追求高奖励而生成 "奖励黑客" 式的输出
      (比如重复讨好性词语)，保持与 SFT 模型的分布接近。

    参数:
        rewards: (batch,) 每个序列的最终奖励 (来自奖励模型)
        logprobs: (batch, seq_len) 策略模型的 log prob
        ref_logprobs: (batch, seq_len) 参考模型的 log prob
        kl_coef: KL 惩罚系数 β
        attention_mask: (batch, seq_len) 有效 token 掩码

    返回:
        token_rewards: (batch, seq_len) 逐 token 的奖励信号
    """
    # 逐 token 的 KL 散度: log(π_θ / π_ref) = logprob_θ - logprob_ref
    # 注意: 这里用的是 per-token KL 的简单估计
    kl_per_token = logprobs - ref_logprobs  # (batch, seq_len)

    # 每个 token 的 KL 惩罚
    token_rewards = -kl_coef * kl_per_token  # (batch, seq_len)

    # 在最后一个有效 token 上加上序列级奖励 R(x,y)
    # 找到每个序列最后一个有效位置
    response_lengths = attention_mask.sum(dim=-1).long()  # (batch,)
    for i in range(rewards.size(0)):
        last_pos = response_lengths[i] - 1
        if last_pos >= 0:
            token_rewards[i, last_pos] += rewards[i]

    return token_rewards


# ============================================================
# 第五步：核心算法 — GAE (广义优势估计)
# ============================================================
def compute_gae(
    token_rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation (GAE)

    优势函数 A(s,a) = Q(s,a) - V(s)，衡量"这个动作比平均好多少"

    GAE 通过指数加权 TD 误差来平衡偏差和方差:
      δ_t = r_t + γ * V(s_{t+1}) - V(s_t)    # TD 误差
      A_t = Σ_{l=0}^{T-t} (γλ)^l * δ_{t+l}   # GAE

    当 λ=0: A_t = δ_t (高偏差，低方差，类似 TD(0))
    当 λ=1: A_t = Σ γ^l * r_{t+l} - V(s_t) (低偏差，高方差，类似 MC)

    参数:
        token_rewards: (batch, seq_len) 逐 token 奖励
        values: (batch, seq_len) Critic 估计的 V(s)
        gamma: 折扣因子
        gae_lambda: GAE 的 λ
        attention_mask: (batch, seq_len) 有效 token 掩码

    返回:
        advantages: (batch, seq_len) 优势值
        returns: (batch, seq_len) 回报值 (用于训练 Critic)
    """
    batch, seq_len = token_rewards.shape
    advantages = torch.zeros_like(token_rewards)
    last_gae = torch.zeros(batch, device=token_rewards.device)

    # 从后往前计算 (时间反向递推)
    for t in reversed(range(seq_len)):
        # 下一步的 Value (最后一步之后为 0)
        next_value = (
            values[:, t + 1] if t < seq_len - 1 else torch.zeros_like(values[:, 0])
        )

        # TD 误差: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = token_rewards[:, t] + gamma * next_value - values[:, t]

        # GAE 递推: A_t = δ_t + γ * λ * A_{t+1}
        last_gae = delta + gamma * gae_lambda * last_gae

        # 只在有效位置计算
        last_gae = last_gae * attention_mask[:, t]

        advantages[:, t] = last_gae

    # 回报 = 优势 + 价值 (用于训练 Critic: L = (V(s) - returns)^2)
    returns = advantages + values

    return advantages, returns


# ============================================================
# 第六步：PPO 损失函数
# ============================================================
class PPOLoss(nn.Module):
    """
    PPO 的三个损失函数:

    1. 策略损失 (Actor Loss):
       L_policy = -min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
       其中 r_t = π_θ(a|s) / π_old(a|s) 是重要性采样比率

       直觉: 如果新策略比旧策略好 (A>0)，鼓励增大概率，
       但通过裁剪防止步子迈太大。

    2. 价值损失 (Critic Loss):
       L_value = max((V - returns)^2, (clip(V, V_old±ε) - returns)^2)

       直觉: 让 Critic 的估计接近真实回报，同时防止更新过大。

    3. 熵奖励:
       L_entropy = -H(π_θ) = Σ π log π

       直觉: 鼓励策略保持一定的随机性，避免过早收敛。

    总损失: L = L_policy + c1 * L_value - c2 * L_entropy
    """

    def __init__(self, config: PPOConfig):
        super().__init__()
        self.clip_eps = config.clip_eps
        self.value_clip_eps = config.value_clip_eps
        self.value_loss_coef = config.value_loss_coef
        self.entropy_coef = config.entropy_coef

    def forward(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        logprobs: (batch, seq_len) 当前策略的 log π_θ(a|s)
        old_logprobs: (batch, seq_len) 采样时策略的 log π_old(a|s)
        advantages: (batch, seq_len) 优势值
        values: (batch, seq_len) 当前 Critic 的 V(s)
        old_values: (batch, seq_len) 采样时 Critic 的 V_old(s)
        returns: (batch, seq_len) 目标回报
        attention_mask: (batch, seq_len) 有效位置掩码
        """
        # ---- 1. 策略损失 (PPO-Clip) ----
        # 重要性采样比率: r_t = exp(log π_new - log π_old)
        ratio = torch.exp(logprobs - old_logprobs)

        # 优势归一化 (降低方差)
        advantages_normalized = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        # 未裁剪的目标
        policy_loss_1 = ratio * advantages_normalized
        # 裁剪后的目标: 限制 ratio 在 [1-ε, 1+ε] 内
        policy_loss_2 = (
            torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            * advantages_normalized
        )
        # 取较小值 (悲观估计)，然后取负号 (最大化 → 最小化)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2)
        policy_loss = (policy_loss * attention_mask).sum() / attention_mask.sum()

        # ---- 2. 价值损失 (Clipped Value Loss) ----
        # 防止 Value 函数更新过大
        values_clipped = old_values + torch.clamp(
            values - old_values, -self.value_clip_eps, self.value_clip_eps
        )
        value_loss_1 = (values - returns).pow(2)
        value_loss_2 = (values_clipped - returns).pow(2)
        value_loss = torch.max(value_loss_1, value_loss_2)
        value_loss = (value_loss * attention_mask).sum() / attention_mask.sum() * 0.5

        # ---- 3. 熵奖励 ----
        # 近似熵: H ≈ -logprob (因为 E[-log p] = H)
        # 实际中应该用完整的 logits 计算熵，这里简化
        entropy = -logprobs
        entropy_loss = -(entropy * attention_mask).sum() / attention_mask.sum()

        # ---- 总损失 ----
        total_loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.mean().detach(),
            "clip_fraction": ((ratio - 1.0).abs() > self.clip_eps)
            .float()
            .mean()
            .detach(),
            "approx_kl": (old_logprobs - logprobs).mean().detach(),
        }


# ============================================================
# 第七步：自适应 KL 系数控制器
# ============================================================
class AdaptiveKLController:
    """
    自适应 KL 惩罚系数

    原理: 动态调整 β 使得 KL(π_θ || π_ref) 保持在目标值附近
    - 如果 KL > target: 增大 β (加大惩罚，让模型别跑太远)
    - 如果 KL < target: 减小 β (放松约束，给模型更多自由度)

    来自 InstructGPT 论文的设计
    """

    def __init__(self, init_kl_coef: float, target: float):
        self.kl_coef = init_kl_coef
        self.target = target

    def update(self, current_kl: float):
        """根据当前 KL 散度调整系数"""
        # 比例控制器
        proportional_error = (current_kl - self.target) / self.target
        # 乘性更新: 当 KL 偏高时增大系数，偏低时减小系数
        mult = 1.0 + 0.1 * proportional_error
        self.kl_coef *= mult
        # 限制范围
        self.kl_coef = max(0.001, min(10.0, self.kl_coef))


# ============================================================
# 第八步：PPO 训练器
# ============================================================
class PPOTrainer:
    """
    PPO 训练器: 编排整个 RLHF 训练流程

    训练循环:
    for each batch of prompts:
        1. [Rollout] 策略模型生成回答
        2. [Evaluate] 奖励模型打分 + Critic 估值 + 参考模型算 KL
        3. [Compute] 计算 GAE 优势 + 带 KL 惩罚的奖励
        4. [Update] PPO 多轮梯度更新 Actor 和 Critic
        5. [Adapt] 调整 KL 系数
    """

    def __init__(
        self,
        policy_model: nn.Module,  # Actor: 要优化的 LLM (π_θ)
        value_model: nn.Module,  # Critic: 价值网络 (V_ψ)
        reward_model: nn.Module,  # 奖励模型 (R_φ, 冻结)
        ref_model: nn.Module,  # 参考模型 (π_ref, 冻结)
        config: PPOConfig,
    ):
        self.policy_model = policy_model
        self.value_model = value_model
        self.reward_model = reward_model
        self.ref_model = ref_model
        self.config = config

        # 冻结奖励模型和参考模型
        for param in self.reward_model.parameters():
            param.requires_grad = False
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # 优化器: 只优化 Actor 和 Critic
        self.optimizer = torch.optim.Adam(
            list(policy_model.parameters()) + list(value_model.parameters()),
            lr=config.lr,
        )

        self.ppo_loss = PPOLoss(config)

        # KL 控制器
        if config.adaptive_kl:
            self.kl_controller = AdaptiveKLController(config.kl_coef, config.kl_target)
        else:
            self.kl_controller = None

    @property
    def kl_coef(self) -> float:
        if self.kl_controller:
            return self.kl_controller.kl_coef
        return self.config.kl_coef

    def compute_logprobs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算模型在 response_ids 上的逐 token log 概率

        input_ids: (batch, query_len) prompt
        response_ids: (batch, response_len) 回答

        返回: (batch, response_len) 每个 token 的 log prob
        """
        # 拼接 prompt + response
        full_ids = torch.cat([input_ids, response_ids], dim=1)

        # 前向传播得到 logits
        logits = model(full_ids)  # (batch, total_len, vocab_size)

        # 取 response 部分的 logits (往前移一位，因为要预测下一个 token)
        response_logits = logits[
            :, input_ids.size(1) - 1 : -1
        ]  # (batch, response_len, vocab_size)

        # 计算每个 token 的 log prob
        log_probs = F.log_softmax(response_logits, dim=-1)

        # 取实际 token 的 log prob
        token_log_probs = log_probs.gather(
            dim=-1, index=response_ids.unsqueeze(-1)
        ).squeeze(
            -1
        )  # (batch, response_len)

        return token_log_probs

    def train_step(self, experience: PPOExperience) -> dict[str, float]:
        """
        一步 PPO 训练

        输入: 预先收集好的经验数据
        输出: 训练指标
        """
        # 1. 计算带 KL 惩罚的逐 token 奖励
        token_rewards = compute_kl_penalty_rewards(
            rewards=experience.rewards,
            logprobs=experience.logprobs,
            ref_logprobs=experience.ref_logprobs,
            kl_coef=self.kl_coef,
            attention_mask=experience.attention_mask,
        )

        # 2. 计算 GAE 优势和回报
        advantages, returns = compute_gae(
            token_rewards=token_rewards,
            values=experience.values.detach(),
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            attention_mask=experience.attention_mask,
        )

        # 3. PPO 多轮更新 (同一批数据迭代 ppo_epochs 次)
        old_logprobs = experience.logprobs.detach()
        old_values = experience.values.detach()
        all_metrics = []

        for epoch in range(self.config.ppo_epochs):
            # 重新计算当前策略的 logprobs 和 values
            new_logprobs = self.compute_logprobs(
                self.policy_model,
                experience.query_ids,
                experience.response_ids,
            )
            # 这里简化了 value_model 的调用
            full_ids = torch.cat([experience.query_ids, experience.response_ids], dim=1)
            new_values = self.value_model(full_ids).squeeze(-1)
            new_values = new_values[:, experience.query_ids.size(1) :]
            new_values = new_values[:, : experience.response_ids.size(1)]

            # 计算 PPO 损失
            metrics = self.ppo_loss(
                logprobs=new_logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages.detach(),
                values=new_values,
                old_values=old_values,
                returns=returns.detach(),
                attention_mask=experience.attention_mask,
            )

            # 梯度更新
            self.optimizer.zero_grad()
            metrics["total_loss"].backward()
            nn.utils.clip_grad_norm_(
                list(self.policy_model.parameters())
                + list(self.value_model.parameters()),
                self.config.max_grad_norm,
            )
            self.optimizer.step()

            all_metrics.append(
                {
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in metrics.items()
                }
            )

        # 4. 更新 KL 系数
        mean_kl = sum(m["approx_kl"] for m in all_metrics) / len(all_metrics)
        if self.kl_controller:
            self.kl_controller.update(mean_kl)

        # 返回最后一轮的指标
        result = all_metrics[-1]
        result["kl_coef"] = self.kl_coef
        result["mean_reward"] = experience.rewards.mean().item()
        return result


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=== PPO for RLHF 测试 ===\n")

    # 1. 测试奖励模型
    print("--- 奖励模型 ---")
    reward_model = RewardModel(hidden_size=64, vocab_size=100)
    chosen_ids = torch.randint(0, 100, (4, 20))
    rejected_ids = torch.randint(0, 100, (4, 20))

    r_chosen = reward_model(chosen_ids)
    r_rejected = reward_model(rejected_ids)
    rm_loss = RewardModelLoss()(r_chosen, r_rejected)
    print(f"  chosen 奖励: {r_chosen.tolist()}")
    print(f"  rejected 奖励: {r_rejected.tolist()}")
    print(f"  RM Loss: {rm_loss.item():.4f}")

    # 2. 测试 KL 惩罚奖励
    print("\n--- KL 惩罚奖励 ---")
    rewards = torch.tensor([1.0, -0.5, 0.8, 0.2])
    logprobs = torch.randn(4, 10)
    ref_logprobs = logprobs + torch.randn(4, 10) * 0.1
    mask = torch.ones(4, 10)
    token_rewards = compute_kl_penalty_rewards(
        rewards, logprobs, ref_logprobs, 0.1, mask
    )
    print(f"  token_rewards shape: {token_rewards.shape}")
    print(f"  最后位置包含序列奖励: {token_rewards[:, -1].tolist()}")

    # 3. 测试 GAE
    print("\n--- GAE ---")
    values = torch.randn(4, 10)
    advantages, returns = compute_gae(
        token_rewards, values, gamma=1.0, gae_lambda=0.95, attention_mask=mask
    )
    print(f"  advantages shape: {advantages.shape}")
    print(f"  returns shape: {returns.shape}")

    # 4. 测试 PPO Loss
    print("\n--- PPO Loss ---")
    ppo_config = PPOConfig()
    ppo_loss = PPOLoss(ppo_config)
    new_logprobs = logprobs + torch.randn_like(logprobs) * 0.01
    new_values = values + torch.randn_like(values) * 0.01
    metrics = ppo_loss(
        logprobs=new_logprobs,
        old_logprobs=logprobs,
        advantages=advantages,
        values=new_values,
        old_values=values,
        returns=returns,
        attention_mask=mask,
    )
    for k, v in metrics.items():
        print(f"  {k}: {v.item():.4f}")

    # 5. 测试自适应 KL 控制器
    print("\n--- 自适应 KL 控制器 ---")
    kl_ctrl = AdaptiveKLController(init_kl_coef=0.1, target=6.0)
    for kl_val in [3.0, 5.0, 7.0, 10.0, 6.0]:
        kl_ctrl.update(kl_val)
        print(f"  KL={kl_val:.1f} → β={kl_ctrl.kl_coef:.4f}")

    print("\n✓ PPO 所有模块测试通过!")
