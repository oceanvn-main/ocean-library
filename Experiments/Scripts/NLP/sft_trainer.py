"""
Supervised Fine-Tuning (SFT) Trainer
Implements Phase 1 of Post-Training Alignment (Jurafsky & Martin Ch. 10.1).
"""

from typing import Dict, Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import IAlignmentStrategy, compute_sequence_log_probs

class SFTTrainer(IAlignmentStrategy):
    """
    Supervised Fine-Tuning Trainer.
    Fine-tunes a pretrained base model on (instruction, response) pairs.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float = 1.0,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.device = device

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Executes one SFT training step.
        Batch dictionary keys: 'input_ids', 'attention_mask', 'prompt_lengths'
        """
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        prompt_lengths = batch.get('prompt_lengths', None)
        if prompt_lengths is not None:
            prompt_lengths = prompt_lengths.to(self.device)

        # Forward pass through Causal LM
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Compute autoregressive cross-entropy loss over response tokens
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Flat Cross Entropy
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        token_losses = loss_fct(flat_logits, flat_labels).view(shift_labels.size())

        # Mask out prompt tokens if prompt_lengths provided
        if prompt_lengths is not None:
            seq_len = shift_labels.size(1)
            positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
            response_mask = positions >= (prompt_lengths.unsqueeze(1) - 1)
            if attention_mask is not None:
                response_mask = response_mask & (attention_mask[:, 1:] > 0)
            
            mask_float = response_mask.float()
            loss = (token_losses * mask_float).sum() / torch.clamp(mask_float.sum(), min=1.0)
        else:
            if attention_mask is not None:
                mask_float = attention_mask[:, 1:].float()
                loss = (token_losses * mask_float).sum() / torch.clamp(mask_float.sum(), min=1.0)
            else:
                loss = token_losses.mean()

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        perplexity = math.exp(min(loss.item(), 20.0))

        return {
            "sft_loss": loss.item(),
            "perplexity": perplexity
        }

    def evaluate(self, eval_dataloader: Any) -> Dict[str, float]:
        """Evaluates mean validation loss and perplexity."""
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        return {
            "eval_sft_loss": avg_loss,
            "eval_perplexity": math.exp(min(avg_loss, 20.0))
        }
