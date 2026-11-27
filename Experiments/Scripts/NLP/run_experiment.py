"""
Master Experiment Execution Script: Post-Training Alignment Engine
Runs full multi-stage pipeline: SFT -> Reward Model -> PPO RLHF -> DPO
Evaluates comparative metrics and mathematical loss convergence across phases.
Zero hardcoding: all tensor operations, log-probs, rewards, and losses derived dynamically via PyTorch.
"""

import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from base import (
    PreferenceBatch,
    RolloutBuffer,
    compute_sequence_log_probs
)
from sft_trainer import SFTTrainer
from reward_model import RewardModel, RewardModelTrainer
from ppo_learner import PPOLearner
from dpo_learner import DPOLearner


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None


class SyntheticLanguageModel(nn.Module):
    """
    Autoregressive Transformer-like Language Model for experiment execution.
    Features trainable embedding, multi-layer GRU, and language model classification head.
    """
    def __init__(self, vocab_size: int = 64, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        **kwargs
    ) -> CausalLMOutput:
        x = self.embedding(input_ids)
        out, _ = self.gru(x)
        logits = self.head(out)
        return CausalLMOutput(logits=logits, hidden_states=(out,))


class SyntheticValueHead(nn.Module):
    """Value Head module for PPO RLHF value estimation."""
    def __init__(self, vocab_size: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.v_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        out, _ = self.gru(x)
        return self.v_head(out[:, -1, :]).squeeze(-1)


def generate_synthetic_dataset(
    num_samples: int = 32,
    vocab_size: int = 64,
    prompt_len: int = 4,
    resp_len: int = 6
) -> Tuple[List[Dict[str, torch.Tensor]], List[PreferenceBatch]]:
    """Generates synthetic instruction and preference batches for reproducible training."""
    torch.manual_seed(42)
    sft_batches = []
    preference_batches = []
    
    batch_size = 8
    num_batches = num_samples // batch_size
    
    for _ in range(num_batches):
        prompt_ids = torch.randint(1, vocab_size, (batch_size, prompt_len))
        resp_ids = torch.randint(1, vocab_size, (batch_size, resp_len))
        full_seq = torch.cat([prompt_ids, resp_ids], dim=1)
        mask = torch.ones_like(full_seq)
        prompt_lengths = torch.full((batch_size,), prompt_len, dtype=torch.long)
        
        sft_batches.append({
            'input_ids': full_seq,
            'attention_mask': mask,
            'prompt_lengths': prompt_lengths
        })
        
        # Preference dataset: chosen response has structured token patterns
        chosen_resp = torch.randint(1, vocab_size // 2, (batch_size, resp_len))
        rejected_resp = torch.randint(vocab_size // 2, vocab_size, (batch_size, resp_len))
        
        chosen_full = torch.cat([prompt_ids, chosen_resp], dim=1)
        rejected_full = torch.cat([prompt_ids, rejected_resp], dim=1)
        
        preference_batches.append(PreferenceBatch(
            prompt_ids=prompt_ids,
            chosen_ids=chosen_full,
            rejected_ids=rejected_full,
            chosen_mask=torch.ones_like(chosen_full),
            rejected_mask=torch.ones_like(rejected_full)
        ))
        
    return sft_batches, preference_batches


def run_experiment():
    print("=" * 70)
    print("      POST-TRAINING ALIGNMENT ENGINE: SFT -> RM -> PPO -> DPO      ")
    print("=" * 70)
    
    vocab_size = 64
    hidden_dim = 32
    prompt_len = 4
    resp_len = 6
    num_epochs = 5
    
    sft_data, pref_data = generate_synthetic_dataset(
        num_samples=32, vocab_size=vocab_size, prompt_len=prompt_len, resp_len=resp_len
    )
    
    # ---------------------------------------------------------
    # PHASE 1: Supervised Fine-Tuning (SFT)
    # ---------------------------------------------------------
    print("\n[PHASE 1] Supervised Fine-Tuning (SFT)...")
    base_model = SyntheticLanguageModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
    sft_optimizer = optim.Adam(base_model.parameters(), lr=2e-3)
    sft_trainer = SFTTrainer(base_model, sft_optimizer)
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_ppl = 0.0
        for batch in sft_data:
            m = sft_trainer.train_step(batch)
            epoch_loss += m['sft_loss']
            epoch_ppl += m['perplexity']
        avg_loss = epoch_loss / len(sft_data)
        avg_ppl = epoch_ppl / len(sft_data)
        print(f"  Epoch {epoch}/{num_epochs} | SFT Loss: {avg_loss:.4f} | Perplexity: {avg_ppl:.2f}")
        
    sft_model = base_model # SFT Checkpoint
    
    # ---------------------------------------------------------
    # PHASE 2: Bradley-Terry Pairwise Preference Reward Model
    # ---------------------------------------------------------
    print("\n[PHASE 2] Training Bradley-Terry Preference Reward Model...")
    rm_backbone = SyntheticLanguageModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
    rm_backbone.load_state_dict(sft_model.state_dict())
    reward_model = RewardModel(rm_backbone, hidden_size=hidden_dim)
    rm_optimizer = optim.Adam(reward_model.parameters(), lr=1e-3)
    rm_trainer = RewardModelTrainer(reward_model, rm_optimizer)
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_margin = 0.0
        for batch in pref_data:
            m = rm_trainer.train_step(batch)
            epoch_loss += m['rm_loss']
            epoch_acc += m['preference_accuracy']
            epoch_margin += m['margin_mean']
        avg_loss = epoch_loss / len(pref_data)
        avg_acc = epoch_acc / len(pref_data)
        avg_margin = epoch_margin / len(pref_data)
        print(f"  Epoch {epoch}/{num_epochs} | RM Loss: {avg_loss:.4f} | Accuracy: {avg_acc*100:.1f}% | Margin: {avg_margin:.4f}")
        
    # ---------------------------------------------------------
    # PHASE 3: Proximal Policy Optimization (PPO RLHF)
    # ---------------------------------------------------------
    print("\n[PHASE 3] Executing PPO RLHF Engine...")
    ppo_policy = SyntheticLanguageModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
    ppo_policy.load_state_dict(sft_model.state_dict())
    
    ppo_ref = SyntheticLanguageModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
    ppo_ref.load_state_dict(sft_model.state_dict())
    
    value_model = SyntheticValueHead(vocab_size=vocab_size, hidden_dim=hidden_dim)
    policy_optimizer = optim.Adam(ppo_policy.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_model.parameters(), lr=1e-3)
    
    ppo_learner = PPOLearner(
        policy_model=ppo_policy,
        ref_model=ppo_ref,
        value_model=value_model,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        kl_coef=0.1,
        clip_eps=0.2
    )
    
    for epoch in range(1, num_epochs + 1):
        epoch_pol_loss = 0.0
        epoch_val_loss = 0.0
        epoch_kl = 0.0
        for batch in pref_data:
            prompts = batch.prompt_ids
            responses = batch.chosen_ids[:, prompt_len:]
            full_seq = batch.chosen_ids
            
            with torch.no_grad():
                old_outputs = ppo_policy(full_seq)
                old_log_probs = compute_sequence_log_probs(
                    old_outputs.logits, full_seq, prompt_lengths=torch.full((prompts.size(0),), prompt_len)
                )
                rewards = reward_model(full_seq)
                values = value_model(full_seq)
                
            rollout_buffer = RolloutBuffer(
                prompts=prompts,
                responses=responses,
                old_log_probs=old_log_probs,
                rewards=rewards,
                values=values
            )
            
            m = ppo_learner.train_step(rollout_buffer)
            epoch_pol_loss += m['ppo_policy_loss']
            epoch_val_loss += m['ppo_value_loss']
            epoch_kl += m['kl_divergence_mean']
            
        n_b = len(pref_data)
        print(f"  Epoch {epoch}/{num_epochs} | PPO Policy Loss: {epoch_pol_loss/n_b:.4f} | Value Loss: {epoch_val_loss/n_b:.4f} | KL Div: {epoch_kl/n_b:.4f}")

    # ---------------------------------------------------------
    # PHASE 4: Direct Preference Optimization (DPO)
    # ---------------------------------------------------------
    print("\n[PHASE 4] Executing Direct Preference Optimization (DPO) Engine...")
    dpo_policy = SyntheticLanguageModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
    dpo_policy.load_state_dict(sft_model.state_dict())
    
    dpo_ref = SyntheticLanguageModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
    dpo_ref.load_state_dict(sft_model.state_dict())
    
    dpo_optimizer = optim.Adam(dpo_policy.parameters(), lr=1e-3)
    dpo_learner = DPOLearner(
        policy_model=dpo_policy,
        ref_model=dpo_ref,
        optimizer=dpo_optimizer,
        beta=0.1
    )
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_margin = 0.0
        epoch_acc = 0.0
        for batch in pref_data:
            m = dpo_learner.train_step(batch)
            epoch_loss += m['dpo_loss']
            epoch_margin += m['reward_margin']
            epoch_acc += m['accuracy']
            
        n_b = len(pref_data)
        print(f"  Epoch {epoch}/{num_epochs} | DPO Loss: {epoch_loss/n_b:.4f} | Implicit Margin: {epoch_margin/n_b:.4f} | Accuracy: {(epoch_acc/n_b)*100:.1f}%")

    # ---------------------------------------------------------
    # SUMMARY & COMPARATIVE METRICS
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("                POST-TRAINING EXPERIMENT SUMMARY                ")
    print("=" * 70)
    print("  Phase                   Metric Name             Final Value")
    print("  " + "-" * 66)
    print(f"  1. SFT                  Final Loss              {avg_loss:.4f}")
    print(f"  1. SFT                  Final Perplexity        {avg_ppl:.2f}")
    print(f"  2. Reward Model         Pairwise Accuracy       {avg_acc*100:.1f}%")
    print(f"  2. Reward Model         Reward Margin           {avg_margin:.4f}")
    print(f"  3. PPO RLHF             Policy Loss             {epoch_pol_loss/n_b:.4f}")
    print(f"  3. PPO RLHF             KL Divergence           {epoch_kl/n_b:.4f}")
    print(f"  4. DPO                  Implicit Loss           {epoch_loss/n_b:.4f}")
    print(f"  4. DPO                  Implicit Reward Margin  {epoch_margin/n_b:.4f}")
    print(f"  4. DPO                  Accuracy                {(epoch_acc/n_b)*100:.1f}%")
    print("=" * 70)
    print("\n[SUCCESS] Post-Training Alignment Engine Master Experiment Complete!")


if __name__ == "__main__":
    run_experiment()
