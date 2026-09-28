# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO/RL 训练的核心算法实现，供 ray_trainer 等 trainer 调用。

本文件包含的算法/组件一览
==========================

【KL 系数控制器】
  - AdaptiveKLController : 根据当前 KL 与目标 KL 的比例自适应调整系数
  - FixedKLController      : 固定 KL 系数，不随训练变化
  - get_kl_controller      : 根据配置创建上述控制器

【Advantage 估计器】（由 algorithm.adv_estimator 选择，决定如何计算优势函数）
  - compute_gae_advantage_return                          : GAE，标准 PPO + Critic 使用
  - compute_grpo_outcome_advantage                        : GRPO，组内相对 reward 归一化
  - compute_reinforce_plus_plus_baseline_outcome_advantage: REINFORCE++-baseline，组内均值 baseline + whitening
  - compute_rloo_outcome_advantage                        : RLOO (Leave-One-Out baseline)
  - compute_reinforce_plus_plus_outcome_advantage         : REINFORCE++，折扣回报 + whitening
  - compute_remax_outcome_advantage                       : ReMax，greedy baseline 减基线

【Reward / Loss 工具】
  - compute_rewards    : 在 token reward 上减去 KL penalty
  - agg_loss           : 将 token 级 loss 聚合为标量（支持 Dr.GRPO 的 seq-mean-token-sum-norm 模式）

【PPO 策略/价值损失】
  - compute_policy_loss : Dual-clip PPO 策略梯度损失
  - compute_entropy_loss: 策略熵正则项
  - compute_value_loss  : Critic 价值函数 clipped MSE 损失
  - kl_penalty          : 多种 KL 散度近似形式（kl / abs / mse / low_var_kl）
"""

from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


# ---------------------------------------------------------------------------
# KL 系数控制器：控制 reward 中 KL penalty 的强度
# ---------------------------------------------------------------------------

class AdaptiveKLController:
    """
    自适应 KL 控制器（参考 https://arxiv.org/pdf/1909.08593.pdf）。
    当实际 KL 偏离 target 时，按比例调整 kl_coef，使策略更新幅度保持在合理范围。
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """固定 KL 系数，整个训练过程中 kl_coef 不变。"""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    """根据配置 type 字段返回 fixed 或 adaptive KL 控制器。"""
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Advantage 估计器：不同 RL 算法在此选用不同的优势函数计算方式
# ---------------------------------------------------------------------------

def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """GAE (Generalized Advantage Estimation) — 标准 PPO 使用，需要 Critic 的 value 估计。

    从 response 末尾反向递推 TD-error，结合 gamma/lambda 得到 token 级 advantage 和 return。
    最后对 advantage 做 masked whitening（零均值、单位方差）。

    参考: https://arxiv.org/abs/1506.02438
    改编自: https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        returns_reversed = []
        gen_len = token_level_rewards.shape[-1]

        # padding / EOS 之后的 token 不参与 bootstrap：强制 gamma=1, lambda=1
        gamma_masked = response_mask * gamma + 1 - response_mask
        lam_masked = response_mask * lam + 1 - response_mask
        nextvalues_skip_obs = 0
        returns_gt = 0

        # 从最后一个 token 反向递推 GAE
        for t in reversed(range(gen_len)):
            next_step_mask = response_mask[:, t + 1] if t < gen_len - 1 else 1.0
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            nextvalues_skip_obs = (1 - next_step_mask) * nextvalues_skip_obs + next_step_mask * nextvalues
            this_step_gamma = gamma_masked[:, t]
            this_step_lam = lam_masked[:, t]
            delta = token_level_rewards[:, t] + this_step_gamma * nextvalues_skip_obs - values[:, t]
            delta *= response_mask[:, t]
            lastgaelam = delta + this_step_gamma * this_step_lam * lastgaelam
            advantages_reversed.append(lastgaelam)

            returns_gt = this_step_gamma * returns_gt + response_mask[:, t] * token_level_rewards[:, t]
            returns_reversed.append(returns_gt)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = torch.stack(returns_reversed[::-1], dim=1)
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): 以下 outcome-based 方法均假设每条 response 只有一个标量 reward（通常在末 token）。
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """GRPO (Group Relative Policy Optimization) 优势估计。

    对同一 prompt（由 index/uid 标识）采样的多条 response，在组内做 (score - mean) / std 归一化，
    得到相对优势。不需要 Critic，只需 rollout.n > 1 做 group sampling。

    norm_adv_by_std_in_grpo=True  为原始 GRPO；False 为 Dr.GRPO（不做 std 缩放）。

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)  # 每条 response 的总 reward

    # 按 prompt uid 分组，计算组内 mean/std
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # 组内只有 1 条 response 时，advantage 置 0（无法做相对比较）
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        # 标量 advantage 广播到所有有效 token
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6
):
    """REINFORCE++-baseline 优势估计（https://arxiv.org/abs/2501.03262）。

    与 GRPO 类似用组内均值作 baseline，但不除以 std；之后对 token 级 advantage 做 masked whitening。
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask)

    return scores, scores


def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6
):
    """RLOO (Reward Leave-One-Out) 优势估计（https://arxiv.org/abs/2402.14740）。

    对组内 n 条 response，用 leave-one-out 公式消除自身对 baseline 的影响：
    advantage_i = score_i * n/(n-1) - mean * n/(n-1)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                # leave-one-out：放大自身 score 并减去组均值，等价于用其余 n-1 条的均值作 baseline
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor
):
    """REINFORCE++ 优势估计（https://arxiv.org/abs/2501.03262）。

    从末 token 反向计算折扣回报 (discounted return)，再 masked whitening。
    与 RF++-baseline 不同：不使用 group baseline，而是时序折扣。
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0
        gamma_masked = response_mask * gamma + 1 - response_mask

        # 反向累积折扣回报 G_t = r_t + gamma * G_{t+1}
        for t in reversed(range(token_level_rewards.shape[1])):
            this_step_gamma = gamma_masked[:, t]
            running_return = token_level_rewards[:, t] + this_step_gamma * running_return
            returns[:, t] = running_return

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor
):
    """ReMax 优势估计（https://arxiv.org/abs/2310.10505）。

    用 greedy decoding 得到的 reward_baselines 作为基线：
    advantage = cumulative_return - baseline
    """

    with torch.no_grad():
        # 从末 token 向前累积 reward 得到 return
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


# ---------------------------------------------------------------------------
# Reward 处理 & Loss 聚合
# ---------------------------------------------------------------------------

def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """在原始 score 上减去 KL penalty，得到用于 advantage 计算的 token_level_rewards。"""
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        # Dr.GRPO 使用的 loss 聚合：先对每条序列的 token loss 求和，再除以固定 max_response_length
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


# ---------------------------------------------------------------------------
# PPO 策略 / 价值 / 熵 / KL 损失
# ---------------------------------------------------------------------------

def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode="token-mean",
):
    """Dual-clip PPO 策略梯度损失（https://arxiv.org/abs/1707.06347, https://arxiv.org/pdf/1912.09729）。

    ratio = exp(log_prob - old_log_prob)
    - 标准 clip: max(-ratio*A, -clip(ratio)*A)
    - 当 A < 0 时额外 dual-clip: min(-ratio*A, -clip_ratio_c*A)

    改编自: https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob  # 一阶 KL 近似
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # 未 clip 的策略梯度 loss
    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    # dual-clip：advantage 为负时限制 ratio 下界，防止策略崩溃
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask):
    """策略熵正则项，鼓励探索；返回 batch 内有效 token 的平均熵。

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=response_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value):
    """Critic 价值函数 clipped MSE 损失（PPO 标准组件，GAE 模式下使用）。

    改编自: https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), response_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """计算当前策略与参考策略之间的 KL 惩罚项（GRPO 等无 Critic 方法常用）。

    支持多种近似形式:
      - kl         : 一阶近似 log pi - log pi_ref
      - abs        : 绝对值
      - mse        : 二阶 MSE
      - low_var_kl : Schulman 低方差 KL 近似（GRPO 默认 kl_loss_type）
      - full       : 全词表 KL（未实现）

    参考: https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # Schulman 低方差 KL 近似: exp(kl) - kl - 1
    # http://joschu.net/blog/kl-approx.html
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
