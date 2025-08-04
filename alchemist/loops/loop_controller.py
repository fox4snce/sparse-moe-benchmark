"""
Loop Control System for Phase 3 Iterative Reasoning

Implements the loop controller that manages iterative inference,
convergence monitoring, and early stopping for the Alchemist system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import math


class LoopController:
    """
    Manages iterative reasoning loops with convergence monitoring.
    
    Controls the number of iterations, monitors convergence,
    and implements early stopping to prevent infinite loops.
    Logs exit reason and tracks delta-probability of thinking head.
    """
    
    def __init__(self, max_cycles: int = 4, convergence_threshold: float = 0.95):
        super().__init__()
        self.max_cycles = max_cycles
        self.convergence_threshold = convergence_threshold
        
        # Statistics tracking
        self.iteration_stats = {
            "total_iterations": 0,
            "convergence_count": 0,
            "max_cycle_count": 0,
            "avg_cycles_per_query": 0.0,
            "exit_reasons": {"converged": 0, "max_cycles": 0, "all_stop": 0}
        }
    
    def compute_convergence(self, current_state: torch.Tensor, previous_state: torch.Tensor) -> float:
        """
        Compute convergence metric between current and previous states.
        
        Args:
            current_state: Current hidden state [batch_size, seq_len, d_model]
            previous_state: Previous hidden state [batch_size, seq_len, d_model]
            
        Returns:
            convergence_score: Cosine similarity between states
        """
        # Flatten for cosine similarity computation
        current_flat = current_state.view(current_state.size(0), -1)
        previous_flat = previous_state.view(previous_state.size(0), -1)
        
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(current_flat, previous_flat, dim=1)
        
        return cos_sim.mean().item()
    
    def compute_delta_prob(self, current_prob: torch.Tensor, previous_prob: torch.Tensor) -> float:
        """
        Compute mean absolute change in thinking head probability between cycles.
        """
        return (current_prob - previous_prob).abs().mean().item()
    
    def has_converged(self, current_state: torch.Tensor, previous_state: torch.Tensor) -> bool:
        """
        Check if the system has converged.
        
        Args:
            current_state: Current hidden state
            previous_state: Previous hidden state
            
        Returns:
            True if converged, False otherwise
        """
        convergence_score = self.compute_convergence(current_state, previous_state)
        return convergence_score > self.convergence_threshold
    
    def run_iterative_inference(
        self,
        query: str,
        specialists: List[nn.Module],
        router: nn.Module,
        thinking_heads: nn.Module,
        tokenizer,
        max_cycles: Optional[int] = None
    ) -> Dict:
        """
        Run iterative inference with loop control.
        Logs exit reason and tracks delta-probability of thinking head.
        
        Args:
            query: Input query string
            specialists: List of specialist models
            router: Crucible router
            thinking_heads: MultiThinkingHeads instance
            tokenizer: Tokenizer for encoding
            max_cycles: Override for max cycles (optional)
            
        Returns:
            Dictionary with results and statistics
        """
        if max_cycles is None:
            max_cycles = self.max_cycles
        
        # Encode query
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Move to device
        device = next(specialists[0].parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Initialize tracking
        current_hidden = None
        previous_prob = None
        iteration_outputs = []
        thinking_probs = []
        convergence_scores = []
        delta_probs = []
        exit_reason = None
        
        for cycle in range(max_cycles):
            # Get specialist outputs
            specialist_outputs = []
            for specialist in specialists:
                with torch.no_grad():
                    outputs = specialist(input_ids, attention_mask)
                    hidden_states = outputs["last_hidden_state"]
                    specialist_outputs.append(hidden_states)
            
            # Get thinking head decisions
            all_probs, all_decisions = thinking_heads(specialist_outputs)
            continue_any = thinking_heads.should_continue_any(specialist_outputs)
            
            # Track mean probability for main specialist (0)
            current_prob = all_probs[0].mean().item()
            thinking_probs.append(current_prob)
            
            # Track delta-probability
            if previous_prob is not None:
                delta = abs(current_prob - previous_prob)
                delta_probs.append(delta)
            previous_prob = current_prob
            
            # Get router decision
            router_result = router(specialist_outputs[0], return_entropy=True)
            router_weights = router_result["weights"]
            
            # Store iteration data
            iteration_outputs.append({
                "cycle": cycle,
                "router_weights": router_weights.detach().cpu(),
                "thinking_probs": all_probs.detach().cpu(),
                "continue_decisions": all_decisions.detach().cpu(),
                "continue_any": continue_any.detach().cpu(),
                "mean_prob": current_prob,
                "delta_prob": delta_probs[-1] if delta_probs else 0.0
            })
            
            # Check convergence if we have a previous state
            if current_hidden is not None:
                convergence_score = self.compute_convergence(
                    specialist_outputs[0], current_hidden
                )
                convergence_scores.append(convergence_score)
                
                if self.has_converged(specialist_outputs[0], current_hidden):
                    self.iteration_stats["convergence_count"] += 1
                    exit_reason = "converged"
                    break
            
            # Update current hidden state
            current_hidden = specialist_outputs[0].detach()
            
            # Check if any specialist wants to continue
            if not continue_any.any():
                exit_reason = "all_stop"
                break
        
        # If we hit max cycles
        if exit_reason is None:
            exit_reason = "max_cycles"
        self.iteration_stats["exit_reasons"][exit_reason] += 1
        
        # Update statistics
        cycles_used = len(iteration_outputs)
        self.iteration_stats["total_iterations"] += cycles_used
        if cycles_used >= max_cycles:
            self.iteration_stats["max_cycle_count"] += 1
        
        # Compute final result (weighted combination of specialist outputs)
        final_output = self.combine_specialist_outputs(
            specialist_outputs, router_weights
        )
        
        result = {
            "final_output": final_output,
            "iterations": iteration_outputs,
            "cycles_used": cycles_used,
            "convergence_scores": convergence_scores,
            "delta_probs": delta_probs,
            "thinking_summary": thinking_heads.get_thinking_summary(specialist_outputs),
            "router_weights": router_weights.detach().cpu(),
            "exit_reason": exit_reason,
            "stats": self.iteration_stats.copy(),
            "num_cycles": cycle + 1 if 'cycle' in locals() else 1
        }
        return result
    
    def combine_specialist_outputs(
        self, 
        specialist_outputs: List[torch.Tensor], 
        router_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine specialist outputs using router weights.
        
        Args:
            specialist_outputs: List of specialist hidden states
            router_weights: Router weight distribution [batch_size, seq_len, n_specialists]
            
        Returns:
            Combined output [batch_size, seq_len, d_model]
        """
        # Stack specialist outputs: [batch, n_specialists, seq, d_model]
        stacked_outputs = torch.stack(specialist_outputs, dim=1)  # [B, N, S, D]
        # Permute router weights to [B, N, S]
        weights_perm = router_weights.permute(0, 2, 1)  # [B, N, S]
        weights_expanded = weights_perm.unsqueeze(-1)   # [B, N, S, 1]
        # Weighted combination
        weighted_outputs = stacked_outputs * weights_expanded  # [B, N, S, D]
        combined_output = weighted_outputs.sum(dim=1)  # [B, S, D]
        return combined_output
    
    def get_loop_statistics(self) -> Dict:
        """
        Get comprehensive loop statistics.
        
        Returns:
            Dictionary with loop performance statistics
        """
        stats = self.iteration_stats.copy()
        
        if stats["total_iterations"] > 0:
            stats["avg_cycles_per_query"] = stats["total_iterations"] / max(1, stats["total_iterations"] // 4)
            stats["convergence_rate"] = stats["convergence_count"] / max(1, stats["total_iterations"])
            stats["max_cycle_rate"] = stats["max_cycle_count"] / max(1, stats["total_iterations"])
        
        return stats
    
    def reset_statistics(self):
        """Reset all loop statistics."""
        self.iteration_stats = {
            "total_iterations": 0,
            "convergence_count": 0,
            "max_cycle_count": 0,
            "avg_cycles_per_query": 0.0,
            "exit_reasons": {"converged": 0, "max_cycles": 0, "all_stop": 0}
        }


class LoopTrainingController(LoopController):
    """
    Extended loop controller for training scenarios.
    
    Adds training-specific features like differentiable unrolling
    and loss computation for loop training.
    """
    
    def __init__(self, max_cycles: int = 4, convergence_threshold: float = 0.95):
        super().__init__(max_cycles, convergence_threshold)
        
        # Training-specific parameters
        self.efficiency_penalty_weight = 0.1
        self.convergence_bonus_weight = 0.05
        # Placeholder for regularizer/kl-divergence weights
        self.memory_downweight_reg = 0.05
        self.kl_stop_continue_weight = 0.01
    
    def compute_loop_training_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        num_cycles: int,
        convergence_score: float,
        stop_probs: Optional[torch.Tensor] = None,
        router_entropy: Optional[torch.Tensor] = None,
        memory_weights: Optional[torch.Tensor] = None,
        continue_flags: Optional[torch.Tensor] = None,
        ce_history: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute training loss with loop-specific terms.
        Includes hooks for regularizer and KL-divergence.
        
        Args:
            outputs: Model outputs
            targets: Target outputs
            num_cycles: Number of cycles used
            convergence_score: Convergence metric
            stop_probs: Thinking head stop probabilities (optional)
            router_entropy: Router entropy (optional)
            memory_weights: Router weights for memory specialist (optional)
            continue_flags: Boolean mask for continue-thinking (optional)
            
        Returns:
            Total loss with loop penalties/bonuses
        """
        # Standard cross-entropy loss
        # Flatten outputs and targets for CE
        batch, seq, vocab = outputs.shape
        ce_loss = F.cross_entropy(outputs.reshape(-1, vocab), targets.reshape(-1))
        
        # --- CE-PER-CYCLE REWARD SHAPING ---
        loop_reward = 0.0
        energy_cost = 0.0
        if ce_history is not None and len(ce_history) > 1:
            # Calculate CE improvement from each cycle
            ce_improvements = []
            for i in range(1, len(ce_history)):
                delta_ce = ce_history[i-1] - ce_history[i]  # Positive = improvement
                ce_improvements.append(delta_ce)
            
            # Reward only positive improvements (CE actually dropped)
            for delta in ce_improvements:
                loop_reward += torch.clamp(delta, min=0.0)  # Only reward improvements
            
            # Energy cost for each extra cycle (make looping expensive)
            energy_cost = 0.02 * (len(ce_history) - 1)
        
        # Traditional efficiency penalty (discourage excessive looping)
        efficiency_penalty = self.efficiency_penalty_weight * (num_cycles - 1)
        
        # Convergence bonus (reward reaching stable state)
        convergence_bonus = self.convergence_bonus_weight * convergence_score
        
        # --- REGULARIZER: Down-weight memory when continue-thinking is false ---
        memory_reg = 0.0
        if memory_weights is not None and continue_flags is not None:
            if continue_flags.dim() == 2:
                # Reduce sequence dimension: if any token in the sample wants to continue, treat as continue
                continue_flags = continue_flags.any(dim=1)
            assert memory_weights.dim() == 1, f"memory_weights expected [B], got {memory_weights.shape}"
            assert continue_flags.dim() == 1, f"continue_flags expected [B], got {continue_flags.shape}"
            # Penalize high memory weight when NOT continuing
            memory_reg = self.memory_downweight_reg * (memory_weights * (1 - continue_flags.float())).mean()
        
        # --- KL-divergence: Encourage stop when router entropy is low & answer prob is high ---
        kl_term = 0.0
        if stop_probs is not None and router_entropy is not None:
            # Target: stop_prob should be high when entropy is low
            target_stop = (router_entropy < 0.2).float()
            # Add epsilon to prevent log(0) = -inf
            eps = 1e-8
            stop_probs_safe = stop_probs.float().clamp(min=eps, max=1-eps)
            kl_term = self.kl_stop_continue_weight * F.kl_div(stop_probs_safe.log(), target_stop, reduction='batchmean')
        
        total_loss = ce_loss + efficiency_penalty - convergence_bonus + memory_reg + kl_term + energy_cost - loop_reward
        
        return total_loss
    
    def differentiable_unroll(
        self,
        query: str,
        specialists: List[nn.Module],
        router: nn.Module,
        thinking_heads: nn.Module,
        tokenizer,
        targets: torch.Tensor
    ) -> Dict:
        """
        Differentiable unrolling for training.
        
        Args:
            query: Input query
            specialists: List of specialists
            router: Router model
            thinking_heads: Thinking heads
            tokenizer: Tokenizer
            targets: Target outputs
            
        Returns:
            Training results with loss
        """
        # Run iterative inference
        results = self.run_iterative_inference(
            query, specialists, router, thinking_heads, tokenizer
        )
        
        # Compute training loss
        final_output = results["final_output"]
        cycles_used = results["cycles_used"]
        convergence_score = results["convergence_scores"][-1] if results["convergence_scores"] else 0.0
        # TODO: Pass stop_probs, router_entropy, memory_weights, continue_flags as needed
        loss = self.compute_loop_training_loss(
            final_output, targets, cycles_used, convergence_score
        )
        
        results["loss"] = loss
        return results 