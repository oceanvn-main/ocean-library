"""
Proximal Policy Optimization (PPO) RLHF Learner for LLMs
Implements Phase 3 of Post-Training Alignment (Jurafsky & Martin Ch. 10.3.2 & Schulman et al. 2017).
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import IAlignmentStrategy, RolloutBuffer, compute_sequence_log_probs
from reward_model import RewardModel

def compute_gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Generalized Advantage Estimation (GAE) and Targets.
    Args:
        rewards: Tensor [batch_size, seq_len] or [batch_size]
        values: Tensor [batch_size, seq_len] or [batch_size]
        gamma: Discount factor
        lam: GAE lambda parameter
    Returns:
        advantages: Tensor matching rewards shape
        returns: Tensor (advantages + values)
    """
    if rewards.dim() == 1:
        # Sequence-level single-step return
        advantages = rewards - values
        returns = rewards
        return advantages, returns

    batch_size, seq_len = rewards.size()
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0.0

    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = 0.0
        else:
            next_value = values[:, t + 1]
            
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        advantages[:, t] = last_gae_lam = delta + gamma * lam * last_gae_lam

    returns = advantages + values
    return advantages, returns

class PPOLearner(IAlignmentStrategy):
    """
    PPO Learner for LLM Reinforcement Learning from Human Feedback (RLHF).
    Features per-token KL divergence penalty and GAE Advantage estimation.
    """
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        value_model: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        value_optimizer: torch.optim.Optimizer,
        clip_eps: float = 0.2,
        kl_coef: float = 0.1,
        vf_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        device: str = "cpu"
    ):
        self.policy_model = policy_model.to(device)
        self.ref_model = ref_model.to(device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.value_model = value_model.to(device)
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer

        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

    def train_step(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Executes one PPO optimization step on a rollout trajectory buffer.
        """
        self.policy_model.train()
        self.value_model.train()

        prompts = buffer.prompts.to(self.device)
        responses = buffer.responses.to(self.device)
        old_log_probs = buffer.old_log_probs.to(self.device)
        rewards_raw = buffer.rewards.to(self.device)
        values_old = buffer.values.to(self.device)

        full_sequence = torch.cat([prompts, responses], dim=1)
        prompt_len = prompts.size(1)

        # 1. Compute reference log-probs to evaluate per-token KL divergence
        with torch.no_grad():
            ref_outputs = self.ref_model(full_sequence)
            ref_log_probs = compute_sequence_log_probs(
                ref_outputs.logits, full_sequence, prompt_lengths=torch.tensor([prompt_len]*prompts.size(0), device=self.device)
            )

        # 2. Compute current policy log-probs
        policy_outputs = self.policy_model(full_sequence)
        new_log_probs = compute_sequence_log_probs(
            policy_outputs.logits, full_sequence, prompt_lengths=torch.tensor([prompt_len]*prompts.size(0), device=self.device)
        )

        # 3. Add KL Penalty: R_total = R_raw - gamma * (log pi_theta - log pi_ref)
        kl_divergence = new_log_probs - ref_log_probs
        adjusted_rewards = rewards_raw - self.kl_coef * kl_divergence.detach()

        # 4. Compute GAE Advantages and Targets
        advantages, returns = compute_gae_advantages(adjusted_rewards, values_old)
        returns = returns.detach()
        # Normalize advantages
        advantages_norm = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).detach()

        # 5. PPO Clipped Policy Surrogate Loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages_norm
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_norm
        policy_loss = -torch.min(surr1, surr2).mean()

        # 6. Value Function Loss
        current_values = self.value_model(full_sequence)
        value_loss = 0.5 * F.mse_loss(current_values, returns)

        # Update Policy Network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        # Update Value Network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.value_model.parameters(), self.max_grad_norm)
        self.value_optimizer.step()

        return {
            "ppo_policy_loss": policy_loss.item(),
            "ppo_value_loss": value_loss.item(),
            "kl_divergence_mean": kl_divergence.mean().item(),
            "adjusted_reward_mean": adjusted_rewards.mean().item(),
            "ratio_mean": ratio.mean().item()
        }

    def evaluate(self, eval_dataloader: Any) -> Dict[str, float]:
        """Evaluates policy divergence on evaluation prompts."""
        self.policy_model.eval()
        total_kl = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                full_sequence = batch['input_ids'].to(self.device)
                prompt_len = batch['prompt_len']

                pol_logits = self.policy_model(full_sequence).logits
                ref_logits = self.ref_model(full_sequence).logits

                prompt_lens_tensor = torch.tensor([prompt_len]*full_sequence.size(0), device=self.device)
                pol_logprobs = compute_sequence_log_probs(pol_logits, full_sequence, prompt_lengths=prompt_lens_tensor)
                ref_logprobs = compute_sequence_log_probs(ref_logits, full_sequence, prompt_lengths=prompt_lens_tensor)

                kl = (pol_logprobs - ref_logprobs).mean().item()
                total_kl += kl
                total_batches += 1

        return {
            "eval_kl_divergence": total_kl / max(1, total_batches)
        }
