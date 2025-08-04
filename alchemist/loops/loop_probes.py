"""
Loop-specific sanity probes and monitoring tools for Phase 3 training.

This module implements the diagnostic tools suggested by GPT:
- Cycle histogram analysis
- Stuck-loop detector
- Latency vs accuracy curves
- Loop decision accuracy tracking
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time
from dataclasses import dataclass


@dataclass
class LoopProbeResults:
    """Results from loop-specific probes."""
    cycle_histogram: Dict[int, int]
    stuck_loops: List[str]
    latency_vs_accuracy: List[Tuple[int, float, float]]  # (cycles, accuracy, latency)
    loop_decision_accuracy: float
    flops_per_prompt: List[float]
    router_weight_heatmap: np.ndarray  # (prompt_idx, cycle, specialist)


class LoopProbeMonitor:
    """Monitor loop behavior during training and evaluation."""
    
    def __init__(self, max_cycles: int = 4, stuck_threshold: float = 1e-4):
        self.max_cycles = max_cycles
        self.stuck_threshold = stuck_threshold
        self.reset()
    
    def reset(self):
        """Reset all monitoring state."""
        self.cycle_counts = []
        self.stuck_loop_prompts = []
        self.router_weights_history = []
        self.loop_decisions = []  # (prompt, cycle, p_continue, final_correct)
        self.flops_per_prompt = []
        self.latency_per_prompt = []
    
    def log_cycle(self, prompt_idx: int, cycle: int, p_continue: float, 
                  router_weights: torch.Tensor, logits_change: Optional[float] = None):
        """Log a single cycle for analysis."""
        self.cycle_counts.append(cycle)
        self.router_weights_history.append(router_weights.detach().cpu().numpy())
        
        # Detect stuck loops
        if logits_change is not None and logits_change < self.stuck_threshold and p_continue > 0.5:
            self.stuck_loop_prompts.append(f"prompt_{prompt_idx}_cycle_{cycle}")
    
    def log_loop_decision(self, prompt: str, cycle: int, p_continue: float, 
                         final_correct: bool):
        """Log loop decision accuracy."""
        self.loop_decisions.append((prompt, cycle, p_continue, final_correct))
    
    def log_flops_and_latency(self, flops: float, latency_ms: float):
        """Log computational cost metrics."""
        self.flops_per_prompt.append(flops)
        self.latency_per_prompt.append(latency_ms)
    
    def get_cycle_histogram(self) -> Dict[int, int]:
        """Get histogram of cycle counts."""
        histogram = defaultdict(int)
        for cycle in self.cycle_counts:
            histogram[cycle] += 1
        return dict(histogram)
    
    def get_stuck_loop_count(self) -> int:
        """Get count of stuck loops detected."""
        return len(self.stuck_loop_prompts)
    
    def get_router_weight_heatmap(self) -> np.ndarray:
        """Get router weight heatmap across prompts and cycles."""
        if not self.router_weights_history:
            return np.array([])
        
        # Stack all router weights: (total_cycles, batch_size, n_specialists)
        stacked = np.stack(self.router_weights_history, axis=0)
        return stacked
    
    def get_loop_decision_accuracy(self) -> float:
        """Calculate accuracy of loop decisions (p_continue > threshold when actually needed)."""
        if not self.loop_decisions:
            return 0.0
        
        correct_decisions = 0
        total_decisions = 0
        
        for prompt, cycle, p_continue, final_correct in self.loop_decisions:
            # Consider it a correct decision if:
            # - p_continue > threshold and final answer was wrong (needed more thinking)
            # - p_continue < threshold and final answer was correct (stopped appropriately)
            if p_continue > 0.5:  # Continue threshold
                if not final_correct:
                    correct_decisions += 1  # Correctly continued when needed
            else:
                if final_correct:
                    correct_decisions += 1  # Correctly stopped when done
            
            total_decisions += 1
        
        return correct_decisions / total_decisions if total_decisions > 0 else 0.0
    
    def get_flops_stats(self) -> Dict[str, float]:
        """Get FLOP statistics."""
        if not self.flops_per_prompt:
            return {"mean": 0.0, "p95": 0.0, "p99": 0.0}
        
        flops_array = np.array(self.flops_per_prompt)
        return {
            "mean": float(np.mean(flops_array)),
            "p95": float(np.percentile(flops_array, 95)),
            "p99": float(np.percentile(flops_array, 99))
        }
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.latency_per_prompt:
            return {"mean": 0.0, "p95": 0.0, "p99": 0.0}
        
        latency_array = np.array(self.latency_per_prompt)
        return {
            "mean": float(np.mean(latency_array)),
            "p95": float(np.percentile(latency_array, 95)),
            "p99": float(np.percentile(latency_array, 99))
        }
    
    def get_results(self) -> LoopProbeResults:
        """Get comprehensive probe results."""
        return LoopProbeResults(
            cycle_histogram=self.get_cycle_histogram(),
            stuck_loops=self.stuck_loop_prompts,
            latency_vs_accuracy=[],  # Will be populated by external analysis
            loop_decision_accuracy=self.get_loop_decision_accuracy(),
            flops_per_prompt=self.flops_per_prompt,
            router_weight_heatmap=self.get_router_weight_heatmap()
        )


class LatencyVsAccuracyAnalyzer:
    """Analyze latency vs accuracy trade-offs for different loop counts."""
    
    def __init__(self, max_cycles: int = 4):
        self.max_cycles = max_cycles
        self.results = defaultdict(list)  # cycle -> [(accuracy, latency), ...]
    
    def add_result(self, cycle: int, accuracy: float, latency_ms: float):
        """Add a result for a specific cycle count."""
        self.results[cycle].append((accuracy, latency_ms))
    
    def get_curve(self) -> List[Tuple[int, float, float]]:
        """Get latency vs accuracy curve."""
        curve = []
        for cycle in range(self.max_cycles + 1):
            if cycle in self.results:
                accuracies = [acc for acc, _ in self.results[cycle]]
                latencies = [lat for _, lat in self.results[cycle]]
                avg_accuracy = np.mean(accuracies)
                avg_latency = np.mean(latencies)
                curve.append((cycle, avg_accuracy, avg_latency))
        return curve
    
    def get_area_under_curve(self) -> float:
        """Calculate area under the latency-accuracy curve."""
        curve = self.get_curve()
        if len(curve) < 2:
            return 0.0
        
        # Sort by latency
        curve.sort(key=lambda x: x[2])
        
        # Calculate area using trapezoidal rule
        area = 0.0
        for i in range(1, len(curve)):
            lat1, acc1 = curve[i-1][2], curve[i-1][1]
            lat2, acc2 = curve[i][2], curve[i][1]
            area += (lat2 - lat1) * (acc1 + acc2) / 2
        
        return area


def run_loop_sanity_check(loop_controller, test_prompts: List[str], 
                         specialists: List, router, thinking_heads, tokenizer,
                         max_cycles: int = 4) -> LoopProbeResults:
    """
    Run comprehensive loop sanity checks on test prompts.
    
    Args:
        loop_controller: LoopController instance
        test_prompts: List of test prompts
        specialists: List of specialist models
        router: Router model
        thinking_heads: Thinking heads
        tokenizer: Tokenizer
        max_cycles: Maximum cycles to test
    
    Returns:
        LoopProbeResults with comprehensive analysis
    """
    monitor = LoopProbeMonitor(max_cycles=max_cycles)
    
    for i, prompt in enumerate(test_prompts):
        start_time = time.time()
        
        # Run iterative inference
        result = loop_controller.run_iterative_inference(
            prompt, specialists, router, thinking_heads, tokenizer, max_cycles
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Log cycle information
        cycles = result.get("num_cycles", 1)
        monitor.log_cycle(i, cycles, result.get("thinking_probs", [0.0])[0], 
                         result.get("router_weights", torch.zeros(1, 3)))
        
        # Log FLOPs (simplified - would need actual FLOP counting in practice)
        estimated_flops = cycles * 1e9  # Rough estimate
        monitor.log_flops_and_latency(estimated_flops, latency_ms)
    
    return monitor.get_results()


def analyze_hyperparameter_sweep(thresholds: List[float], loop_weights: List[float], 
                                max_cycles_list: List[int], test_data: List[Tuple[str, str]]) -> Dict:
    """
    Analyze hyperparameter sweep results.
    
    Args:
        thresholds: List of continue thresholds to test
        loop_weights: List of loop loss weights to test
        max_cycles_list: List of max cycles to test
        test_data: List of (prompt, expected_answer) tuples
    
    Returns:
        Dictionary with best hyperparameters and results
    """
    results = []
    
    for tau in thresholds:
        for lambda_loop in loop_weights:
            for max_cycles in max_cycles_list:
                # This would run actual training/evaluation with these params
                # For now, return placeholder structure
                result = {
                    "threshold": tau,
                    "loop_weight": lambda_loop,
                    "max_cycles": max_cycles,
                    "accuracy": 0.0,  # Would be actual accuracy
                    "latency_ms": 0.0,  # Would be actual latency
                    "area_under_curve": 0.0  # Would be actual AUC
                }
                results.append(result)
    
    # Find best configuration (highest area under curve)
    best_result = max(results, key=lambda x: x["area_under_curve"])
    
    return {
        "best_config": best_result,
        "all_results": results
    } 