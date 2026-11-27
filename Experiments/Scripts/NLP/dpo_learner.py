"""
Direct Preference Optimization (DPO) Learner
Implements DPO algorithm from Rafailov et al. (2023) and Jurafsky & Martin Ch 10.3.2.
Eliminates explicit RL reward modeling by directly optimizing policy against reference model.
"""

from typing import Dict, Any, Optional, Tuple
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import IAlignmentStrategy, PreferenceBatch, compute_sequence_log_probs


class DPOLearner(IAlignmentStrategy):
    """
    Direct Preference Optimization (DPO) Strategy.
    Optimizes language model policy directly on pairwise preference data without explicit reward model training.
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        beta: float = 0.1,
        max_grad_norm: float = 1.0
    ):
        """
        Args:
            policy_model: Trainable policy neural network
            ref_model: Frozen reference model (SFT checkpoint)
            optimizer: PyTorch optimizer for policy model
            beta: DPO temperature / KL weight parameter beta
            max_grad_norm: Gradient clipping threshold
        """
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        
        # Ensure reference model is in evaluation mode and gradients disabled
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def _get_batch_log_probs(
        self,
        model: nn.Module,
        batch: PreferenceBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes chosen and rejected sequence log probabilities for a given model.
        """
        prompt_len = batch.prompt_ids.size(1)
        batch_size = batch.prompt_ids.size(0)
        prompt_lengths = torch.full(
            (batch_size,), prompt_len, dtype=torch.long, device=batch.prompt_ids.device
        )
        
        # Forward pass for chosen sequence
        chosen_logits = model(batch.chosen_ids)
        chosen_log_probs = compute_sequence_log_probs(
            chosen_logits, batch.chosen_ids, batch.chosen_mask, prompt_lengths
        )
        
        # Forward pass for rejected sequence
        rejected_logits = model(batch.rejected_ids)
        rejected_log_probs = compute_sequence_log_probs(
            rejected_logits, batch.rejected_ids, batch.rejected_mask, prompt_lengths
        )
        
        return chosen_log_probs, rejected_log_probs

    def compute_dpo_loss(
        self,
        policy_chosen_log_probs: torch.Tensor,
        policy_rejected_log_probs: torch.Tensor,
        ref_chosen_log_probs: torch.Tensor,
        ref_rejected_log_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes DPO loss, implicit rewards, reward margin, and preference accuracy.
        
        Formula:
            L_DPO = - log sigma( beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x)) )
            r_hat(x, y) = beta * log (pi(y|x) / pi_ref(y|x))
        """
        pi_logratios = policy_chosen_log_probs - policy_rejected_log_probs
        ref_logratios = ref_chosen_log_probs - ref_rejected_log_probs
        
        logits = pi_logratios - ref_logratios
        
        # Negative log-sigmoid DPO loss
        loss = -F.logsigmoid(self.beta * logits).mean()
        
        # Calculate implicit rewards for policy diagnostic tracking
        chosen_rewards = self.beta * (policy_chosen_log_probs - ref_chosen_log_probs).detach()
        rejected_rewards = self.beta * (policy_rejected_log_probs - ref_rejected_log_probs).detach()
        reward_margin = chosen_rewards - rejected_rewards
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        return loss, chosen_rewards.mean(), rejected_rewards.mean(), reward_margin.mean(), accuracy

    def train_step(self, batch: PreferenceBatch) -> Dict[str, float]:
        """
        Executes single DPO training step.
        """
        self.policy_model.train()
        
        # Compute policy log-probs (with gradient tracking)
        policy_chosen_log_probs, policy_rejected_log_probs = self._get_batch_log_probs(
            self.policy_model, batch
        )
        
        # Compute reference log-probs (no gradients)
        with torch.no_grad():
            ref_chosen_log_probs, ref_rejected_log_probs = self._get_batch_log_probs(
                self.ref_model, batch
            )
            
        loss, chosen_r, rejected_r, margin, acc = self.compute_dpo_loss(
            policy_chosen_log_probs,
            policy_rejected_log_probs,
            ref_chosen_log_probs,
            ref_rejected_log_probs
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            "dpo_loss": loss.item(),
            "chosen_reward": chosen_r.item(),
            "rejected_reward": rejected_r.item(),
            "reward_margin": margin.item(),
            "accuracy": acc.item()
        }

    def evaluate(self, eval_dataloader: Any) -> Dict[str, float]:
        """
        Evaluates DPO metrics across evaluation dataset.
        """
        self.policy_model.eval()
        total_loss = 0.0
        total_margin = 0.0
        total_acc = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                policy_chosen_log_probs, policy_rejected_log_probs = self._get_batch_log_probs(
                    self.policy_model, batch
                )
                ref_chosen_log_probs, ref_rejected_log_probs = self._get_batch_log_probs(
                    self.ref_model, batch
                )
                
                loss, _, _, margin, acc = self.compute_dpo_loss(
                    policy_chosen_log_probs,
                    policy_rejected_log_probs,
                    ref_chosen_log_probs,
                    ref_rejected_log_probs
                )
                
                total_loss += loss.item()
                total_margin += margin.item()
                total_acc += acc.item()
                n_batches += 1
                
        if n_batches == 0:
            return {}
            
        return {
            "eval_dpo_loss": total_loss / n_batches,
            "eval_reward_margin": total_margin / n_batches,
            "eval_accuracy": total_acc / n_batches
        }
