"""
GRPO (Group Relative Policy Optimization) 组相对策略优化

来自 DeepSeek-R1 论文，是 DeepSeek 用来训练推理模型的核心算法。

GRPO 的核心创新:
  PPO 需要单独训练一个 Critic (价值模型) 来估计优势函数，
  这增加了计算成本和训练不稳定性。
  GRPO 的做法: 不要 Critic！用同一个 prompt 的多个采样回答互相比较。

算法流程:
  1. 对每个 prompt x，采样 G 个回答 {y_1, y_2, ..., y_G}
  2. 对每个回答用规则/奖励模型打分 {r_1, r_2, ..., r_G}
  3. 在组内做 Z-Score 归一化作为优势: A_i = (r_i - mean(r)) / std(r)
  4. 用 PPO-Clip 风格的损失更新策略

GRPO 损失:
  L = -1/G * Σ_{i=1}^{G} [
      min(r_i * A_i, clip(r_i, 1-ε, 1+ε) * A_i) - β * KL(π_θ || π_ref)
  ]

  其中:
  - r_i = π_θ(y_i|x) / π_old(y_i|x) 是重要性采样比率
  - A_i = (R_i - mean(R)) / std(R) 是组内归一化优势
  - β * KL 是 KL 散度惩罚

为什么 GRPO 好:
  1. 不需要 Critic 模型: 减少 50% 的 GPU 显存
  2. 优势估计更直接: 同 prompt 多采样对比，无需拟合价值函数
  3. 天然适合推理任务: 可以用规则奖励 (数学答案对/错) 代替奖励模型
  4. 比 DPO 更灵活: 不限于偏好对，支持任意奖励信号
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable, Optional


# ============================================================
# 第一步：GRPO 配置
# ============================================================
@dataclass
class GRPOConfig:
    """GRPO 训练超参数"""

    # --- 组采样 ---
    group_size: int = 8  # 每个 prompt 采样 G 个回答
    temperature: float = 0.7  # 采样温度
    max_new_tokens: int = 256  # 最大生成长度

    # --- PPO-Clip ---
    clip_eps: float = 0.2  # 裁剪范围 ε
    ppo_epochs: int = 1  # 每批数据的 PPO 迭代次数 (GRPO 通常只需 1 次)

    # --- KL 散度 ---
    kl_coef: float = 0.04  # KL 惩罚系数 β
    kl_type: str = "per_token"  # "per_token" 逐 token KL, "sequence" 序列级 KL

    # --- 训练 ---
    lr: float = 1e-6
    max_grad_norm: float = 1.0

    # --- 奖励 ---
    reward_clip: float = 5.0  # 奖励裁剪范围 (防止极端值)


# ============================================================
# 第二步：GRPO 经验数据
# ============================================================
@dataclass
class GRPOExperience:
    """
    GRPO 经验数据: 一组采样及其奖励

    对每个 prompt，采样 G 个回答，形成一个 "组"
    """

    prompt_ids: torch.Tensor  # (batch, prompt_len)
    response_ids: torch.Tensor  # (batch * G, response_len) G 个采样展开
    response_mask: torch.Tensor  # (batch * G, response_len)
    old_logprobs: torch.Tensor  # (batch * G, response_len) 采样时的 log prob
    ref_logprobs: torch.Tensor  # (batch * G, response_len) 参考模型的 log prob
    rewards: torch.Tensor  # (batch * G,) 每个回答的奖励
    group_indices: torch.Tensor  # (batch * G,) 每个回答属于哪个 prompt


# ============================================================
# 第三步：核心算法 — 组内优势计算
# ============================================================
def compute_group_advantages(
    rewards: torch.Tensor,
    group_indices: torch.Tensor,
    clip_range: float = 5.0,
) -> torch.Tensor:
    """
    组内 Z-Score 归一化计算优势

    对每个 prompt 的 G 个采样:
      A_i = (R_i - mean(R_group)) / (std(R_group) + eps)

    这就是 GRPO 的精髓: 用组内相对排名代替 Critic!

    直觉:
      - 同一个 prompt 的 G 个回答互相比较
      - 奖励高于组均值的回答获得正优势 (鼓励)
      - 奖励低于组均值的回答获得负优势 (抑制)

    参数:
        rewards: (N,) 所有回答的奖励 (N = batch * group_size)
        group_indices: (N,) 每个回答属于哪个 prompt
        clip_range: 优势值裁剪范围

    返回:
        advantages: (N,) 组内归一化后的优势值
    """
    advantages = torch.zeros_like(rewards)

    # 对每个 prompt 的组分别归一化
    unique_groups = group_indices.unique()

    for group_id in unique_groups:
        # 找到属于同一 prompt 的所有回答
        mask = group_indices == group_id
        group_rewards = rewards[mask]

        # Z-Score 归一化
        mean = group_rewards.mean()
        std = group_rewards.std()

        if std > 1e-8:
            # 标准归一化
            group_advantages = (group_rewards - mean) / std
        else:
            # 如果所有回答奖励相同，优势为 0
            group_advantages = torch.zeros_like(group_rewards)

        advantages[mask] = group_advantages

    # 裁剪极端值
    advantages = advantages.clamp(-clip_range, clip_range)

    return advantages


# ============================================================
# 第四步：GRPO 损失函数
# ============================================================
class GRPOLoss(nn.Module):
    """
    GRPO 损失函数

    L = -1/(G*T) * Σ_i Σ_t [
        min(r_t * A_i, clip(r_t, 1-ε, 1+ε) * A_i)
        - β * KL_t(π_θ || π_ref)
    ]

    与 PPO 损失的区别:
    1. 优势 A_i 是组内归一化的序列级奖励 (不是 GAE)
    2. KL 项直接加在损失里 (不是加在奖励里)
    3. 没有 Value Loss (因为没有 Critic)

    KL 散度估计 (逐 token):
      KL_t = π_ref(y_t|...) / π_θ(y_t|...) - log(π_ref(y_t|...) / π_θ(y_t|...)) - 1
      这是 KL 散度的一个无偏估计器 (Schulman 2020)
    """

    def __init__(self, config: GRPOConfig):
        super().__init__()
        self.clip_eps = config.clip_eps
        self.kl_coef = config.kl_coef
        self.kl_type = config.kl_type

    def forward(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        logprobs: (N, seq_len) 当前策略的 log prob
        old_logprobs: (N, seq_len) 采样时的 log prob
        ref_logprobs: (N, seq_len) 参考模型的 log prob
        advantages: (N,) 组内归一化的优势 (序列级)
        attention_mask: (N, seq_len) 有效 token 掩码
        """
        # ---- 1. PPO-Clip 策略损失 ----
        # 重要性采样比率
        ratio = torch.exp(logprobs - old_logprobs)  # (N, seq_len)

        # 优势扩展到 token 级别 (同一序列所有 token 共享同一个优势)
        adv_expanded = advantages.unsqueeze(-1)  # (N, 1)

        # 裁剪
        policy_loss_1 = ratio * adv_expanded
        policy_loss_2 = (
            torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_expanded
        )
        policy_loss = -torch.min(policy_loss_1, policy_loss_2)

        # 只在有效位置计算
        policy_loss = (policy_loss * attention_mask).sum() / attention_mask.sum()

        # ---- 2. KL 散度惩罚 ----
        # 使用 Schulman 的无偏 KL 估计器:
        # KL ≈ exp(log π_ref - log π_θ) - (log π_ref - log π_θ) - 1
        log_ratio = ref_logprobs - logprobs  # log(π_ref / π_θ)
        kl = torch.exp(log_ratio) - log_ratio - 1.0  # 无偏 KL 估计

        kl_loss = (kl * attention_mask).sum() / attention_mask.sum()

        # ---- 3. 总损失 ----
        total_loss = policy_loss + self.kl_coef * kl_loss

        # ---- 4. 指标 ----
        clip_fraction = (
            ((ratio - 1.0).abs() > self.clip_eps).float() * attention_mask
        ).sum() / attention_mask.sum()

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "kl": kl.mean().detach(),
            "clip_fraction": clip_fraction.detach(),
        }


# ============================================================
# 第五步：奖励函数 (规则奖励示例)
# ============================================================
class RuleBasedReward:
    """
    基于规则的奖励函数 (DeepSeek-R1 风格)

    GRPO 的一大优势是可以用规则奖励 (不需要奖励模型):
    - 数学: 答案是否正确 (精确匹配)
    - 代码: 测试是否通过
    - 格式: 是否包含 <think>...</think> 标签

    组合多种奖励:
      R_total = w1 * R_accuracy + w2 * R_format + w3 * R_length
    """

    def __init__(
        self,
        accuracy_reward: float = 1.0,  # 答对 +1
        format_reward: float = 0.5,  # 格式正确 +0.5
        length_penalty: float = -0.001,  # 过长惩罚
    ):
        self.accuracy_reward = accuracy_reward
        self.format_reward = format_reward
        self.length_penalty = length_penalty

    def __call__(
        self,
        responses: list[str],
        ground_truths: list[str],
    ) -> torch.Tensor:
        """
        计算每个回答的奖励

        responses: 模型生成的回答列表
        ground_truths: 标准答案列表

        返回: (N,) 奖励分数
        """
        rewards = []

        for response, gt in zip(responses, ground_truths):
            reward = 0.0

            # 1. 准确性奖励: 回答是否包含正确答案
            if gt.strip() in response:
                reward += self.accuracy_reward

            # 2. 格式奖励: 是否有思考过程标签
            if "<think>" in response and "</think>" in response:
                reward += self.format_reward

            # 3. 长度惩罚: 防止生成过长
            reward += self.length_penalty * len(response)

            rewards.append(reward)

        return torch.tensor(rewards)


# ============================================================
# 第六步：GRPO 训练器
# ============================================================
class GRPOTrainer:
    """
    GRPO 训练器

    训练循环:
    for each batch of prompts:
        1. [采样] 对每个 prompt 采样 G 个回答
        2. [评分] 用奖励函数给每个回答打分
        3. [归一化] 组内 Z-Score 归一化得到优势
        4. [更新] PPO-Clip 损失更新策略

    与 PPO 的对比:
        PPO:  需要 Actor, Critic, Reward Model, Reference Model (4 个模型)
        GRPO: 只需要 Actor, Reference Model (2 个模型)

    与 DPO 的对比:
        DPO:  只能用偏好对 (pairwise), 离线数据
        GRPO: 可以用任意奖励信号, 在线采样
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_fn: Callable,  # 奖励函数 (可以是规则函数或奖励模型)
        config: GRPOConfig,
        tokenizer=None,  # 用于 decode (这里简化省略)
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.config = config

        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(policy_model.parameters(), lr=config.lr)
        self.loss_fn = GRPOLoss(config)

    def compute_token_logprobs(
        self,
        model: nn.Module,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 response 中每个 token 的 log prob

        prompt_ids: (N, prompt_len)
        response_ids: (N, response_len)
        返回: (N, response_len)
        """
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        logits = model(full_ids)  # (N, total_len, vocab_size)

        # 取 response 对应位置的 logits
        response_logits = logits[:, prompt_ids.size(1) - 1 : -1]
        log_probs = F.log_softmax(response_logits, dim=-1)

        token_logprobs = log_probs.gather(
            dim=-1, index=response_ids.unsqueeze(-1)
        ).squeeze(-1)

        return token_logprobs

    def train_step(self, experience: GRPOExperience) -> dict[str, float]:
        """
        一步 GRPO 训练

        输入: 预先收集好的组采样经验
        """
        self.policy_model.train()

        # 1. 计算组内优势 (GRPO 的核心!)
        advantages = compute_group_advantages(
            rewards=experience.rewards,
            group_indices=experience.group_indices,
            clip_range=self.config.reward_clip,
        )

        # 2. PPO 更新
        all_metrics = []

        for epoch in range(self.config.ppo_epochs):
            # 展开 prompt_ids 以匹配 response_ids 的 batch 维度
            num_responses = experience.response_ids.size(0)
            prompt_len = experience.prompt_ids.size(1)
            expanded_prompts = experience.prompt_ids[experience.group_indices]

            # 计算当前策略的 logprobs
            new_logprobs = self.compute_token_logprobs(
                self.policy_model,
                expanded_prompts,
                experience.response_ids,
            )

            # 计算 GRPO 损失
            metrics = self.loss_fn(
                logprobs=new_logprobs,
                old_logprobs=experience.old_logprobs.detach(),
                ref_logprobs=experience.ref_logprobs.detach(),
                advantages=advantages.detach(),
                attention_mask=experience.response_mask,
            )

            # 梯度更新
            self.optimizer.zero_grad()
            metrics["total_loss"].backward()
            nn.utils.clip_grad_norm_(
                self.policy_model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

            all_metrics.append(
                {
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in metrics.items()
                }
            )

        result = all_metrics[-1]
        result["mean_reward"] = experience.rewards.mean().item()
        result["reward_std"] = experience.rewards.std().item()
        result["mean_advantage"] = advantages.mean().item()
        return result


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=== GRPO (Group Relative Policy Optimization) 测试 ===\n")

    # 1. 测试组内优势计算
    print("--- 组内优势计算 ---")
    # 假设 2 个 prompt，每个采样 4 个回答
    rewards = torch.tensor([0.8, 0.2, 0.5, 1.0, -0.3, 0.7, 0.1, 0.9])
    group_indices = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    advantages = compute_group_advantages(rewards, group_indices)
    print(f"  奖励: {rewards.tolist()}")
    print(f"  分组: {group_indices.tolist()}")
    print(f"  优势: {[f'{a:.3f}' for a in advantages.tolist()]}")
    print(f"  组0 均值={rewards[:4].mean():.2f}, 组1 均值={rewards[4:].mean():.2f}")

    # 验证: 组内均值为 0
    for g in [0, 1]:
        mask = group_indices == g
        print(f"  组{g} 优势均值: {advantages[mask].mean():.6f} (应接近 0)")

    # 2. 测试规则奖励
    print("\n--- 规则奖励 ---")
    reward_fn = RuleBasedReward()
    responses = [
        "<think>Let me solve this...</think> The answer is 42.",
        "The answer is 42.",
        "<think>Hmm...</think> The answer is 99.",
        "I don't know.",
    ]
    ground_truths = ["42", "42", "42", "42"]
    rewards = reward_fn(responses, ground_truths)
    for resp, r in zip(responses, rewards.tolist()):
        print(f"  [{r:+.3f}] {resp[:50]}...")

    # 3. 测试 GRPO 损失
    print("\n--- GRPO 损失 ---")
    config = GRPOConfig(group_size=4)
    loss_fn = GRPOLoss(config)

    N, T = 8, 10  # 8 个回答, 每个 10 tokens
    logprobs = torch.randn(N, T) * 0.1 - 5
    old_logprobs = logprobs + torch.randn(N, T) * 0.01
    ref_logprobs = logprobs + torch.randn(N, T) * 0.05
    advantages = torch.randn(N)
    mask = torch.ones(N, T)

    metrics = loss_fn(logprobs, old_logprobs, ref_logprobs, advantages, mask)
    for k, v in metrics.items():
        print(f"  {k}: {v.item():.4f}")

    # 4. 测试 GRPO 训练器
    print("\n--- GRPO 训练器 ---")

    class SimpleLM(nn.Module):
        def __init__(self, vocab_size=100, hidden_size=64):
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
            return self.lm_head(self.layers(self.embedding(x)))

    policy = SimpleLM()
    ref = SimpleLM()
    ref.load_state_dict(policy.state_dict())

    config = GRPOConfig(group_size=4, lr=1e-4)
    trainer = GRPOTrainer(policy, ref, reward_fn=None, config=config)  # type: ignore

    # 构造假经验数据
    batch_size = 2
    G = config.group_size
    prompt_len = 5
    response_len = 10

    experience = GRPOExperience(
        prompt_ids=torch.randint(0, 100, (batch_size, prompt_len)),
        response_ids=torch.randint(0, 100, (batch_size * G, response_len)),
        response_mask=torch.ones(batch_size * G, response_len),
        old_logprobs=torch.randn(batch_size * G, response_len) * 0.1 - 5,
        ref_logprobs=torch.randn(batch_size * G, response_len) * 0.1 - 5,
        rewards=torch.randn(batch_size * G),  # 随机奖励
        group_indices=torch.tensor([i // G for i in range(batch_size * G)]),
    )

    for step in range(3):
        metrics = trainer.train_step(experience)
        print(
            f"  Step {step}: loss={metrics['total_loss']:.4f}, "
            f"kl={metrics['kl']:.4f}, "
            f"reward={metrics['mean_reward']:.3f}"
        )

    # 5. PPO vs DPO vs GRPO 对比
    print("\n" + "=" * 60)
    print("=== PPO vs DPO vs GRPO 对比 ===")
    print(f"{'':>15} | {'PPO':>10} | {'DPO':>10} | {'GRPO':>10}")
    print("-" * 55)
    comparisons = [
        ("需要模型数", "4", "2", "2"),
        ("需要奖励模型", "是", "否", "可选"),
        ("需要 Critic", "是", "否", "否"),
        ("数据要求", "在线采样", "离线偏好对", "在线采样"),
        ("优势估计", "GAE", "隐式", "组内归一化"),
        ("训练稳定性", "较差", "好", "好"),
        ("超参敏感度", "高", "低", "中"),
        ("适用场景", "通用", "偏好对齐", "推理任务"),
        ("显存占用", "高", "低", "中"),
        ("代表工作", "InstructGPT", "Zephyr", "DeepSeek-R1"),
    ]
    for name, ppo, dpo, grpo in comparisons:
        print(f"{name:>15} | {ppo:>10} | {dpo:>10} | {grpo:>10}")

    print("\n✓ GRPO 所有模块测试通过!")
