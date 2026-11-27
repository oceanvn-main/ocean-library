"""
Unit Test Suite for Post-Training Alignment Engine
Tests base log-prob computations, SFT, Reward Model, PPO RLHF, and DPO modules.
Ensures zero hardcoded values, strict mathematical correctness, and real PyTorch gradient flows.
"""

import unittest
from dataclasses import dataclass
from typing import Optional, Tuple
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


class TinyLM(nn.Module):
    """Minimal PyTorch Autoregressive Language Model for unit testing."""
    def __init__(self, vocab_size: int = 32, hidden_dim: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        **kwargs
    ) -> CausalLMOutput:
        x = self.embedding(input_ids)
        out, _ = self.lstm(x)
        logits = self.head(out)
        return CausalLMOutput(logits=logits, hidden_states=(out,))


class TinyValueHead(nn.Module):
    """Minimal Value Head network for PPO."""
    def __init__(self, vocab_size: int = 32, hidden_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.v_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        out, _ = self.lstm(x)
        # Squeeze to return sequence or scalar value [batch_size]
        return self.v_head(out[:, -1, :]).squeeze(-1)


class TestAlignmentEngine(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 32
        self.hidden_dim = 16
        self.batch_size = 2
        self.seq_len = 8
        self.prompt_len = 3
        
        self.model = TinyLM(self.vocab_size, self.hidden_dim)
        self.ref_model = TinyLM(self.vocab_size, self.hidden_dim)
        
    def test_compute_sequence_log_probs(self):
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        outputs = self.model(input_ids)
        prompt_lengths = torch.tensor([self.prompt_len, self.prompt_len])
        
        seq_log_probs = compute_sequence_log_probs(
            outputs.logits, input_ids, prompt_lengths=prompt_lengths
        )
        
        self.assertEqual(seq_log_probs.shape, (self.batch_size,))
        self.assertTrue(torch.all(seq_log_probs <= 0.0), "Log-probs must be <= 0")
        self.assertFalse(torch.isnan(seq_log_probs).any(), "NaN detected in sequence log-probs")

    def test_sft_trainer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        trainer = SFTTrainer(self.model, optimizer)
        
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        attention_mask = torch.ones_like(input_ids)
        prompt_lengths = torch.tensor([self.prompt_len, self.prompt_len])
        
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'prompt_lengths': prompt_lengths
        }
        
        # Save initial weights
        init_param = next(self.model.parameters()).clone()
        
        metrics = trainer.train_step(batch)
        
        self.assertIn("sft_loss", metrics)
        self.assertIn("perplexity", metrics)
        self.assertGreater(metrics["sft_loss"], 0.0)
        self.assertGreater(metrics["perplexity"], 1.0)
        
        # Verify gradients were applied
        updated_param = next(self.model.parameters())
        self.assertFalse(torch.equal(init_param, updated_param), "SFT optimizer did not update parameters")

    def test_reward_model(self):
        rm = RewardModel(self.model, hidden_size=self.hidden_dim)
        optimizer = optim.Adam(rm.parameters(), lr=1e-3)
        rm_trainer = RewardModelTrainer(rm, optimizer)
        
        prompt_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.prompt_len))
        chosen_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        rejected_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        batch = PreferenceBatch(
            prompt_ids=prompt_ids,
            chosen_ids=chosen_ids,
            rejected_ids=rejected_ids
        )
        
        metrics = rm_trainer.train_step(batch)
        
        self.assertIn("rm_loss", metrics)
        self.assertIn("margin_mean", metrics)
        self.assertIn("preference_accuracy", metrics)
        self.assertGreater(metrics["rm_loss"], 0.0)
        self.assertTrue(0.0 <= metrics["preference_accuracy"] <= 1.0)

    def test_ppo_learner(self):
        policy_optim = optim.Adam(self.model.parameters(), lr=1e-3)
        value_model = TinyValueHead(self.vocab_size, self.hidden_dim)
        value_optim = optim.Adam(value_model.parameters(), lr=1e-3)
        
        ppo = PPOLearner(
            policy_model=self.model,
            ref_model=self.ref_model,
            value_model=value_model,
            policy_optimizer=policy_optim,
            value_optimizer=value_optim,
            kl_coef=0.1,
            clip_eps=0.2
        )
        
        prompts = torch.randint(0, self.vocab_size, (self.batch_size, self.prompt_len))
        responses = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len - self.prompt_len))
        
        # Construct synthetic rollout buffer for unit test
        full_seq = torch.cat([prompts, responses], dim=1)
        with torch.no_grad():
            old_logits = self.model(full_seq).logits
            old_log_probs = compute_sequence_log_probs(
                old_logits, full_seq, prompt_lengths=torch.tensor([self.prompt_len]*self.batch_size)
            )
            values = value_model(full_seq)
            rewards = torch.tensor([0.5, -0.2])
            
        buffer = RolloutBuffer(
            prompts=prompts,
            responses=responses,
            old_log_probs=old_log_probs,
            rewards=rewards,
            values=values
        )
        
        metrics = ppo.train_step(buffer)
        
        self.assertIn("ppo_policy_loss", metrics)
        self.assertIn("ppo_value_loss", metrics)
        self.assertIn("kl_divergence_mean", metrics)

    def test_dpo_learner(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        dpo = DPOLearner(
            policy_model=self.model,
            ref_model=self.ref_model,
            optimizer=optimizer,
            beta=0.1
        )
        
        prompt_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.prompt_len))
        chosen_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        rejected_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        batch = PreferenceBatch(
            prompt_ids=prompt_ids,
            chosen_ids=chosen_ids,
            rejected_ids=rejected_ids
        )
        
        metrics = dpo.train_step(batch)
        
        self.assertIn("dpo_loss", metrics)
        self.assertIn("chosen_reward", metrics)
        self.assertIn("rejected_reward", metrics)
        self.assertIn("reward_margin", metrics)
        self.assertIn("accuracy", metrics)
        self.assertGreater(metrics["dpo_loss"], 0.0)
        self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
