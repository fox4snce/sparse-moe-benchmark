"""
Continue-Thinking Heads for Phase 3 Loop Control

Implements binary classification heads that determine whether
specialists should continue thinking (iterate) or stop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ThinkingHead(nn.Module):
    """
    Binary classifier that determines if a specialist should continue thinking.
    
    Outputs a probability of "need more thinking" based on the current
    hidden state. Used to enable iterative reasoning without external scaffolding.
    The threshold is a learnable parameter (default 0.7, but can be tuned/lowered).
    """
    
    def __init__(self, d_model: int = 256, threshold: float = 0.7, learnable_threshold: bool = True):
        super().__init__()
        self.d_model = d_model
        # Make threshold a learnable parameter if requested
        if learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        else:
            self.register_buffer('threshold', torch.tensor(threshold, dtype=torch.float32))
        
        # Simple binary classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
        
        # Optional: Add layer norm for stability
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to determine continue-thinking probability.
        
        Args:
            hidden_state: Hidden state from specialist [batch_size, seq_len, d_model]
            
        Returns:
            prob: Continue-thinking probability [batch_size, seq_len]
            continue_thinking: Boolean mask for continue decision [batch_size, seq_len]
        """
        # Apply layer norm for stability
        normalized_state = self.layer_norm(hidden_state)
        
        # Get continue-thinking probability
        logits = self.classifier(normalized_state)  # [batch_size, seq_len, 1]
        prob = torch.sigmoid(logits).squeeze(-1)   # [batch_size, seq_len]
        
        # Determine continue decision using (possibly learnable) threshold
        continue_thinking = prob > self.threshold
        
        return prob, continue_thinking
    
    def get_thinking_stats(self, hidden_state: torch.Tensor) -> dict:
        """
        Get statistics about thinking decisions.
        
        Args:
            hidden_state: Hidden state from specialist
            
        Returns:
            Dictionary with thinking statistics
        """
        prob, continue_thinking = self.forward(hidden_state)
        
        return {
            "mean_prob": prob.mean().item(),
            "max_prob": prob.max().item(),
            "continue_ratio": continue_thinking.float().mean().item(),
            "continue_count": continue_thinking.sum().item(),
            "total_tokens": continue_thinking.numel(),
            "threshold": float(self.threshold.detach().cpu().item())
        }


class MultiThinkingHeads(nn.Module):
    """
    Collection of thinking heads for multiple specialists.
    
    Manages thinking heads for all specialists and provides
    unified interface for loop control decisions.
    """
    
    def __init__(self, d_model: int = 256, n_specialists: int = 3, threshold: float = 0.7, learnable_threshold: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_specialists = n_specialists
        self.threshold = threshold
        self.learnable_threshold = learnable_threshold
        
        # Create thinking head for each specialist
        self.thinking_heads = nn.ModuleList([
            ThinkingHead(d_model, threshold, learnable_threshold) for _ in range(n_specialists)
        ])
    
    def forward(self, specialist_outputs: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for all thinking heads.
        
        Args:
            specialist_outputs: List of hidden states from each specialist
                              Each element: [batch_size, seq_len, d_model]
            
        Returns:
            all_probs: Continue-thinking probabilities for all specialists
                      [n_specialists, batch_size, seq_len]
            all_decisions: Continue decisions for all specialists
                          [n_specialists, batch_size, seq_len]
        """
        all_probs = []
        all_decisions = []
        
        for i, (thinking_head, hidden_state) in enumerate(zip(self.thinking_heads, specialist_outputs)):
            prob, decision = thinking_head(hidden_state)
            all_probs.append(prob)
            all_decisions.append(decision)
        
        # Stack into tensors
        all_probs = torch.stack(all_probs, dim=0)      # [n_specialists, batch_size, seq_len]
        all_decisions = torch.stack(all_decisions, dim=0)  # [n_specialists, batch_size, seq_len]
        
        return all_probs, all_decisions
    
    def should_continue_any(self, specialist_outputs: list) -> torch.Tensor:
        """
        Determine if ANY specialist wants to continue thinking.
        
        Args:
            specialist_outputs: List of hidden states from each specialist
            
        Returns:
            continue_any: Boolean tensor indicating if any specialist wants to continue
                        [batch_size, seq_len]
        """
        _, all_decisions = self.forward(specialist_outputs)
        
        # If any specialist wants to continue, the system continues
        continue_any = all_decisions.any(dim=0)  # [batch_size, seq_len]
        
        return continue_any
    
    def get_thinking_summary(self, specialist_outputs: list) -> dict:
        """
        Get summary statistics for all thinking heads.
        
        Args:
            specialist_outputs: List of hidden states from each specialist
            
        Returns:
            Dictionary with summary statistics
        """
        all_probs, all_decisions = self.forward(specialist_outputs)
        
        summary = {
            "total_continue_ratio": all_decisions.float().mean().item(),
            "any_continue_ratio": all_decisions.any(dim=0).float().mean().item(),
            "mean_prob_per_specialist": all_probs.mean(dim=1).mean(dim=1).tolist(),
            "max_prob_per_specialist": all_probs.max(dim=1)[0].max(dim=1)[0].tolist(),
            "continue_count_per_specialist": all_decisions.sum(dim=1).sum(dim=1).tolist(),
            "thresholds": [float(h.threshold.detach().cpu().item()) for h in self.thinking_heads]
        }
        
        return summary


def create_thinking_head_for_specialist(specialist_name: str, d_model: int = 256, learnable_threshold: bool = True) -> ThinkingHead:
    """
    Factory function to create thinking head for a specific specialist.
    
    Args:
        specialist_name: Name of the specialist ("math", "creative", "memory")
        d_model: Hidden dimension size
        learnable_threshold: Whether the threshold is a learnable parameter
        
    Returns:
        ThinkingHead configured for the specialist
    """
    # Can customize thresholds per specialist if needed
    thresholds = {
        "math": 0.7,      # Math problems often need iteration
        "creative": 0.6,   # Creative tasks may need refinement
        "memory": 0.8      # Memory queries usually direct
    }
    
    threshold = thresholds.get(specialist_name, 0.7)
    
    return ThinkingHead(d_model, threshold, learnable_threshold) 