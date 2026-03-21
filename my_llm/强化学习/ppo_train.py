"""
PPO 训练 Qwen2.5-0.5B-Instruct 实战

场景: 训练模型在回答问题时更加"有帮助"
  - 构造简单的 prompt 数据集
  - 用基于规则的奖励函数模拟奖励模型 (避免需要额外训练 RM)
  - 完整走一遍 PPO 训练流程

训练涉及 4 个模型:
  1. Actor (π_θ): Qwen2.5-0.5B-Instruct，要优化的模型
  2. Critic (V_ψ): Actor 的 backbone + value_head，估计状态价值
  3. Reference (π_ref): 冻结的 Actor 副本，计算 KL 散度
  4. Reward: 基于规则的奖励函数 (实际中用训练好的奖励模型)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import copy
import os

# ============================================================
# 第一步: 配置
# ============================================================
@dataclass
class PPOTrainConfig:
    # 模型路径
    model_path: str = os.path.join(os.path.dirname(__file__), "model", "Qwen2.5-0.5B-Instruct")

    # PPO 超参
    clip_eps: float = 0.2          # 策略裁剪范围
    value_clip_eps: float = 0.2    # 价值裁剪范围
    ppo_epochs: int = 2            # 每批数据迭代几次 PPO
    gamma: float = 1.0             # 折扣因子 (语言任务只有最后有奖励，用 1.0)
    gae_lambda: float = 0.95       # GAE lambda
    kl_coef: float = 0.05          # KL 惩罚系数 β
    kl_target: float = 6.0         # 自适应 KL 目标
    value_loss_coef: float = 0.5   # Critic 损失权重
    entropy_coef: float = 0.01     # 熵奖励权重

    # 训练参数
    lr: float = 5e-6               # 学习率
    total_steps: int = 30          # 总训练步数
    batch_size: int = 2            # 每步的 prompt 数量
    max_prompt_len: int = 64       # prompt 最大长度
    max_response_len: int = 128    # 生成回答最大长度
    max_grad_norm: float = 1.0     # 梯度裁剪
    log_interval: int = 1          # 打印间隔

    # 生成参数
    temperature: float = 0.7
    top_k: int = 50


# ============================================================
# 第二步: 构造 Prompt 数据集
# ============================================================
PROMPTS = [
    "请用简单的语言解释什么是人工智能。",
    "如何保持健康的生活方式？",
    "推荐三本值得阅读的书籍。",
    "解释一下什么是机器学习。",
    "如何提高学习效率？",
    "写一首关于春天的短诗。",
    "请解释什么是深度学习。",
    "如何学好编程？",
    "用简单的话解释量子计算。",
    "给出三个节省时间的小技巧。",
    "解释区块链的基本原理。",
    "如何培养良好的阅读习惯？",
    "介绍一下Python编程语言的优点。",
    "如何有效管理时间？",
    "请描述一下太阳系的结构。",
    "什么是自然语言处理？",
    "如何写好一篇文章？",
    "解释一下什么是神经网络。",
    "给出保护环境的五个建议。",
    "如何克服拖延症？",
]


def get_prompt_batch(tokenizer, batch_size, max_len):
    """随机抽取一批 prompt 并编码"""
    import random
    selected = random.choices(PROMPTS, k=batch_size)

    # 用 chat template 格式化
    formatted = []
    for p in selected:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted.append(text)

    encodings = tokenizer(
        formatted,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return selected, encodings["input_ids"], encodings["attention_mask"]


# ============================================================
# 第三步: 基于规则的奖励函数 (模拟 Reward Model)
# ============================================================
def rule_based_reward(prompt: str, response: str) -> float:
    """
    基于规则的奖励函数，给回答打分
    实际 RLHF 中这里是一个训练好的奖励模型

    奖励标准:
      +1.0  回答长度适中 (50~200 字)
      +0.5  包含条理性标记 (1. 2. 3. 或 - 等)
      +0.3  包含具体例子或解释
      -1.0  回答太短 (<20 字)
      -0.5  重复内容多
      -0.5  包含无意义内容
    """
    reward = 0.0

    # 1. 长度奖励
    resp_len = len(response)
    if resp_len < 20:
        reward -= 1.0
    elif 50 <= resp_len <= 300:
        reward += 1.0
    elif resp_len > 500:
        reward -= 0.3  # 太啰嗦

    # 2. 条理性奖励
    structure_markers = ["1.", "2.", "3.", "- ", "首先", "其次", "最后", "第一", "第二"]
    if any(m in response for m in structure_markers):
        reward += 0.5

    # 3. 具体性奖励
    detail_markers = ["例如", "比如", "举例", "具体来说", "也就是说"]
    if any(m in response for m in detail_markers):
        reward += 0.3

    # 4. 重复惩罚 (检测连续重复的片段)
    for seg_len in [10, 20]:
        if resp_len > seg_len * 2:
            segments = [response[i:i+seg_len] for i in range(0, resp_len - seg_len, seg_len)]
            unique_ratio = len(set(segments)) / max(len(segments), 1)
            if unique_ratio < 0.5:
                reward -= 0.5
                break

    # 5. 无意义内容惩罚
    junk_markers = ["嗯嗯嗯", "哈哈哈", "啊啊啊", "......" * 3]
    if any(m in response for m in junk_markers):
        reward -= 0.5

    return reward


# ============================================================
# 第四步: Value Head (Critic 的价值头)
# ============================================================
class ValueHead(nn.Module):
    """
    在 LLM backbone 上加一个线性层，输出每个 token 位置的标量价值 V(s)
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_size)
        x = torch.tanh(self.dense(hidden_states))
        return self.value_head(x).squeeze(-1)  # (batch, seq_len)


# ============================================================
# 第五步: 核心 PPO 算法函数
# ============================================================
def compute_logprobs_from_logits(logits, labels):
    """
    从 logits 计算指定 token 的 log 概率
    logits: (batch, seq_len, vocab_size)
    labels: (batch, seq_len)
    返回: (batch, seq_len)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def compute_gae(token_rewards, values, gamma, gae_lambda, mask):
    """
    Generalized Advantage Estimation
    逐 token 从后往前递推:
      delta_t = r_t + gamma * V(t+1) - V(t)
      A_t = delta_t + gamma * lambda * A(t+1)
    """
    batch, seq_len = token_rewards.shape
    advantages = torch.zeros_like(token_rewards)
    last_gae = torch.zeros(batch, device=token_rewards.device)

    for t in reversed(range(seq_len)):
        next_val = values[:, t + 1] if t < seq_len - 1 else torch.zeros_like(values[:, 0])
        delta = token_rewards[:, t] + gamma * next_val - values[:, t]
        last_gae = delta + gamma * gae_lambda * last_gae
        last_gae = last_gae * mask[:, t]
        advantages[:, t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_loss(logprobs, old_logprobs, advantages, values, old_values, returns, mask, config):
    """
    PPO 三个损失:
    1. 策略损失: -min(ratio * A, clip(ratio) * A)
    2. 价值损失: clipped MSE
    3. 熵奖励: 鼓励探索
    """
    # 策略损失
    ratio = torch.exp(logprobs - old_logprobs)
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    loss1 = ratio * adv_norm
    loss2 = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * adv_norm
    policy_loss = -torch.min(loss1, loss2)
    policy_loss = (policy_loss * mask).sum() / mask.sum()

    # 价值损失
    v_clipped = old_values + torch.clamp(values - old_values, -config.value_clip_eps, config.value_clip_eps)
    vl1 = (values - returns).pow(2)
    vl2 = (v_clipped - returns).pow(2)
    value_loss = 0.5 * (torch.max(vl1, vl2) * mask).sum() / mask.sum()

    # 熵
    entropy = -(logprobs * mask).sum() / mask.sum()

    total = policy_loss + config.value_loss_coef * value_loss + config.entropy_coef * (-entropy)

    # 统计
    clip_frac = ((ratio - 1.0).abs() > config.clip_eps).float().mean().item()
    approx_kl = (old_logprobs - logprobs).mean().item()

    return total, {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "clip_fraction": clip_frac,
        "approx_kl": approx_kl,
    }


# ============================================================
# 第六步: PPO 训练主循环
# ============================================================
def train():
    config = PPOTrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ---------- 加载模型和 tokenizer ----------
    print(f"\n[1/4] 加载模型: {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"  # decoder-only 模型生成时用左填充
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Actor: 要训练的策略模型
    actor = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)

    # Reference: 冻结的副本，用于计算 KL 散度
    ref_model = copy.deepcopy(actor)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Critic: Value Head
    hidden_size = actor.config.hidden_size
    value_head = ValueHead(hidden_size).to(device)

    print(f"  Actor 参数量: {sum(p.numel() for p in actor.parameters()) / 1e6:.1f}M")
    print(f"  Value Head 参数量: {sum(p.numel() for p in value_head.parameters()) / 1e6:.2f}M")

    # ---------- 优化器 ----------
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(value_head.parameters()),
        lr=config.lr,
    )

    # ---------- KL 控制器 ----------
    kl_coef = config.kl_coef

    # ---------- 训练循环 ----------
    print(f"\n[2/4] 开始 PPO 训练 (共 {config.total_steps} 步)")
    print("=" * 80)

    for step in range(1, config.total_steps + 1):

        # ======== 阶段 1: Rollout (生成回答) ========
        actor.eval()
        prompts_text, prompt_ids, prompt_mask = get_prompt_batch(
            tokenizer, config.batch_size, config.max_prompt_len
        )
        prompt_ids = prompt_ids.to(device)

        # 生成回答
        with torch.no_grad():
            gen_output = actor.generate(
                prompt_ids,
                max_new_tokens=config.max_response_len,
                temperature=config.temperature,
                top_k=config.top_k,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 分离 prompt 和 response
        prompt_len = prompt_ids.shape[1]
        response_ids = gen_output[:, prompt_len:]  # (batch, resp_len)
        resp_len = response_ids.shape[1]

        if resp_len == 0:
            print(f"  Step {step}: 生成长度为 0，跳过")
            continue

        # 解码回答文本
        responses_text = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # response 的 attention mask (非 pad 位置)
        resp_mask = (response_ids != tokenizer.pad_token_id).float()

        # ======== 阶段 2: 评估 (计算奖励、logprobs、values) ========
        with torch.no_grad():
            # Actor 的 logprobs
            full_ids = gen_output  # prompt + response
            actor_out = actor(full_ids)
            actor_logits = actor_out.logits[:, prompt_len - 1:-1, :]  # 对齐 response
            actor_logits = actor_logits[:, :resp_len, :]
            old_logprobs = compute_logprobs_from_logits(actor_logits, response_ids)

            # Reference 的 logprobs
            ref_out = ref_model(full_ids)
            ref_logits = ref_out.logits[:, prompt_len - 1:-1, :]
            ref_logits = ref_logits[:, :resp_len, :]
            ref_logprobs = compute_logprobs_from_logits(ref_logits, response_ids)

            # Critic 的 values
            actor_hidden = actor_out.hidden_states if hasattr(actor_out, "hidden_states") else None
            # 重新跑一次获取 hidden states
            actor_out2 = actor(full_ids, output_hidden_states=True)
            hidden = actor_out2.hidden_states[-1][:, prompt_len:, :]
            hidden = hidden[:, :resp_len, :]
            old_values = value_head(hidden)

            # 基于规则的奖励
            rewards = torch.tensor([
                rule_based_reward(p, r) for p, r in zip(prompts_text, responses_text)
            ], device=device, dtype=torch.float32)

        # ======== 阶段 3: 计算优势函数 ========
        # KL 惩罚奖励: 每个 token 加 -beta * KL, 最后一个 token 加上序列奖励
        kl_per_token = old_logprobs - ref_logprobs  # 逐 token KL 估计
        token_rewards = -kl_coef * kl_per_token
        # 在最后一个有效 token 加上序列级奖励
        resp_lengths = resp_mask.sum(dim=-1).long()
        for i in range(config.batch_size):
            last_pos = resp_lengths[i] - 1
            if last_pos >= 0:
                token_rewards[i, last_pos] += rewards[i]

        advantages, returns = compute_gae(
            token_rewards, old_values.detach(), config.gamma, config.gae_lambda, resp_mask
        )

        # ======== 阶段 4: PPO 梯度更新 ========
        actor.train()
        all_metrics = []

        for ppo_epoch in range(config.ppo_epochs):
            # 重新计算 logprobs 和 values (因为参数更新了)
            out = actor(full_ids, output_hidden_states=True)
            new_logits = out.logits[:, prompt_len - 1:-1, :]
            new_logits = new_logits[:, :resp_len, :]
            new_logprobs = compute_logprobs_from_logits(new_logits, response_ids)

            new_hidden = out.hidden_states[-1][:, prompt_len:, :]
            new_hidden = new_hidden[:, :resp_len, :]
            new_values = value_head(new_hidden)

            # PPO 损失
            loss, metrics = ppo_loss(
                new_logprobs, old_logprobs, advantages.detach(), new_values,
                old_values.detach(), returns.detach(), resp_mask, config,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(value_head.parameters()),
                config.max_grad_norm,
            )
            optimizer.step()
            all_metrics.append(metrics)

        # ======== 阶段 5: 自适应 KL 调整 ========
        mean_kl = sum(m["approx_kl"] for m in all_metrics) / len(all_metrics)
        proportional_error = (mean_kl - config.kl_target) / config.kl_target
        kl_coef *= (1.0 + 0.1 * proportional_error)
        kl_coef = max(0.001, min(10.0, kl_coef))

        # ======== 打印日志 ========
        if step % config.log_interval == 0:
            m = all_metrics[-1]
            mean_reward = rewards.mean().item()
            mean_resp_len = resp_lengths.float().mean().item()
            kl_val = kl_per_token.mean().item()

            print(f"Step {step:3d}/{config.total_steps} | "
                  f"reward: {mean_reward:+.2f} | "
                  f"kl: {kl_val:.3f} | "
                  f"policy_loss: {m['policy_loss']:.4f} | "
                  f"value_loss: {m['value_loss']:.4f} | "
                  f"entropy: {m['entropy']:.2f} | "
                  f"clip_frac: {m['clip_fraction']:.2f} | "
                  f"β: {kl_coef:.4f} | "
                  f"resp_len: {mean_resp_len:.0f}")

            # 打印一个生成样例
            if step % 5 == 0 or step == 1:
                print(f"  Prompt:   {prompts_text[0][:60]}")
                resp_preview = responses_text[0][:120].replace("\n", " ")
                print(f"  Response: {resp_preview}")
                print(f"  Reward:   {rewards[0].item():+.2f}")
                print()

    # ---------- 训练结束 ----------
    print("=" * 80)
    print("[3/4] 训练完成! 对比训练前后的生成效果:\n")

    test_prompt = "如何提高学习效率？"
    messages = [{"role": "user", "content": test_prompt}]
    test_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    test_ids = tokenizer(test_input, return_tensors="pt")["input_ids"].to(device)

    # 训练后的模型生成
    actor.eval()
    with torch.no_grad():
        trained_output = actor.generate(
            test_ids, max_new_tokens=150, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.pad_token_id,
        )
    trained_text = tokenizer.decode(trained_output[0][test_ids.shape[1]:], skip_special_tokens=True)

    # 参考模型 (训练前) 生成
    with torch.no_grad():
        ref_output = ref_model.generate(
            test_ids, max_new_tokens=150, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.pad_token_id,
        )
    ref_text = tokenizer.decode(ref_output[0][test_ids.shape[1]:], skip_special_tokens=True)

    print(f"Prompt: {test_prompt}")
    print(f"\n[训练前] {ref_text[:300]}")
    print(f"\n[训练后] {trained_text[:300]}")

    # ---------- 保存 ----------
    save_dir = os.path.join(os.path.dirname(__file__), "model", "Qwen2.5-0.5B-PPO")
    print(f"\n[4/4] 保存模型到: {save_dir}")
    actor.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Done!")


if __name__ == "__main__":
    train()
