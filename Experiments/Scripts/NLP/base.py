"""
Base Contracts and Data Structures for Post-Training Alignment Engine
Implements Ocean Library Strategy Pattern & PyTorch Token Log-Prob Calculation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class PreferenceBatch:
    """Dataclass holding pairwise preference triplets."""
    prompt_ids: torch.Tensor
    chosen_ids: torch.Tensor
    rejected_ids: torch.Tensor
    chosen_mask: Optional[torch.Tensor] = None
    rejected_mask: Optional[torch.Tensor] = None

@dataclass
class RolloutBuffer:
    """Buffer storing PPO RLHF rollout trajectory steps."""
    prompts: torch.Tensor
    responses: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None

def compute_sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    prompt_lengths: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes cumulative log-probabilities sum_{t} log pi(y_t | x, y_<t}) for response tokens.
    
    Args:
        logits: Model output tensor [batch_size, seq_len, vocab_size]
        labels: Input token IDs [batch_size, seq_len]
        mask: Attention mask [batch_size, seq_len]
        prompt_lengths: Tensor of prompt lengths [batch_size] to mask out prompt tokens
    Returns:
        Tensor of per-sequence log probabilities [batch_size]
    """
    if hasattr(logits, 'logits'):
        logits = logits.logits
        
    # Shift logits and labels for autoregressive language modeling
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # Compute log probabilities for all vocabulary tokens
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log-probs for actual label tokens
    per_token_log_probs = torch.gather(
        log_probs, dim=2, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Construct response mask
    if mask is not None:
        shift_mask = mask[:, 1:].contiguous().float()
    else:
        shift_mask = torch.ones_like(shift_labels, dtype=torch.float32)
        
    # If prompt_lengths provided, zero out prompt token positions
    if prompt_lengths is not None:
        seq_len = shift_labels.size(1)
        positions = torch.arange(seq_len, device=labels.device).unsqueeze(0)
        # Prompt positions are <= prompt_lengths - 1 in shifted coordinates
        response_mask = positions >= (prompt_lengths.unsqueeze(1) - 1)
        shift_mask = shift_mask * response_mask.float()
        
    # Sum log probabilities across response tokens
    sequence_log_probs = (per_token_log_probs * shift_mask).sum(dim=-1)
    return sequence_log_probs

class IAlignmentStrategy(ABC):
    """Abstract Strategy interface for Post-Training Alignment algorithms."""
    
    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Executes a single optimization step and returns metrics dictionary."""
        pass

    @abstractmethod
    def evaluate(self, eval_dataloader: Any) -> Dict[str, float]:
        """Evaluates model performance on validation data."""
        pass
