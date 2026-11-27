"""
Bradley-Terry Pairwise Preference Reward Model
Implements Phase 2 of Post-Training Alignment (Jurafsky & Martin Ch. 10.3.1).
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import IAlignmentStrategy, PreferenceBatch

class RewardModel(nn.Module):
    """
    Scalar Reward Model r_psi(x, y).
    Wraps a transformer backbone and adds a linear projection to scalar reward.
    """
    def __init__(self, backbone: nn.Module, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        self.v_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass returning scalar reward for sequence.
        Returns:
            rewards: Tensor [batch_size] containing reward score for last non-padding token
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Extract last hidden layer states [batch_size, seq_len, hidden_size]
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]
        else:
            # Fallback for standard CausalLM output
            last_hidden_state = outputs[0]

        # Compute token-level scalar rewards [batch_size, seq_len]
        token_rewards = self.v_head(last_hidden_state).squeeze(-1)

        # Extract reward at the end-of-sequence / last non-padded token
        if attention_mask is not None:
            last_token_indices = attention_mask.sum(dim=-1).long() - 1
            last_token_indices = torch.clamp(last_token_indices, min=0)
            rewards = token_rewards.gather(dim=1, index=last_token_indices.unsqueeze(-1)).squeeze(-1)
        else:
            rewards = token_rewards[:, -1]

        return rewards

class RewardModelTrainer(IAlignmentStrategy):
    """
    Trainer for Bradley-Terry Pairwise Preference Reward Model.
    Loss: - E [ log sigmoid( r_psi(x, y_w) - r_psi(x, y_l) ) ]
    """
    def __init__(
        self,
        reward_model: RewardModel,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float = 1.0,
        device: str = "cpu"
    ):
        self.model = reward_model.to(device)
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.device = device

    def train_step(self, batch: PreferenceBatch) -> Dict[str, float]:
        """
        Executes one Bradley-Terry reward model training step.
        """
        self.model.train()
        self.optimizer.zero_grad()

        chosen_ids = batch.chosen_ids.to(self.device)
        rejected_ids = batch.rejected_ids.to(self.device)
        chosen_mask = batch.chosen_mask.to(self.device) if batch.chosen_mask is not None else None
        rejected_mask = batch.rejected_mask.to(self.device) if batch.rejected_mask is not None else None

        # Compute rewards for preferred (chosen) and dispreferred (rejected) responses
        r_chosen = self.model(chosen_ids, chosen_mask)
        r_rejected = self.model(rejected_ids, rejected_mask)

        # Bradley-Terry Preference Loss
        margin = r_chosen - r_rejected
        loss = -F.logsigmoid(margin).mean()

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Compute accuracy (how often r_chosen > r_rejected)
        accuracy = (margin > 0).float().mean().item()

        return {
            "rm_loss": loss.item(),
            "reward_chosen_mean": r_chosen.mean().item(),
            "reward_rejected_mean": r_rejected.mean().item(),
            "margin_mean": margin.mean().item(),
            "preference_accuracy": accuracy
        }

    def evaluate(self, eval_dataloader: Any) -> Dict[str, float]:
        """Evaluates mean Bradley-Terry loss and preference accuracy."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                chosen_ids = batch.chosen_ids.to(self.device)
                rejected_ids = batch.rejected_ids.to(self.device)
                chosen_mask = batch.chosen_mask.to(self.device) if batch.chosen_mask is not None else None
                rejected_mask = batch.rejected_mask.to(self.device) if batch.rejected_mask is not None else None

                r_chosen = self.model(chosen_ids, chosen_mask)
                r_rejected = self.model(rejected_ids, rejected_mask)
                margin = r_chosen - r_rejected

                loss = -F.logsigmoid(margin).mean()
                total_loss += loss.item() * chosen_ids.size(0)
                total_correct += (margin > 0).float().sum().item()
                total_samples += chosen_ids.size(0)

        return {
            "eval_rm_loss": total_loss / max(1, total_samples),
            "eval_preference_accuracy": total_correct / max(1, total_samples)
        }
