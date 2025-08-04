"""
Evaluation harness for the Alchemist Architecture.

Implements benchmarks and metrics for assessing Phase 1 performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import json
import csv
import os
from tqdm import tqdm
import time

from alchemist.specialists.base_specialist import SpecialistModel, create_specialist, count_parameters, estimate_flops
from alchemist.routing.crucible_router import CrucibleRouter, create_router
from alchemist.routing.projection_heads import MultiProjectionHeads, create_projection_heads
from alchemist.foundation.tokenizer import SharedTokenizer


class AlchemistEvaluator:
    """
    Evaluator for the Alchemist Architecture.
    
    Implements comprehensive evaluation metrics for Phase 1.
    """
    
    def __init__(
        self,
        tokenizer: SharedTokenizer,
        device: str = "cuda",
        batch_size: int = 32
    ):
        """
        Initialize the evaluator.
        
        Args:
            tokenizer: Shared tokenizer
            device: Evaluation device
            batch_size: Batch size for evaluation
        """
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        
        print(f"Alchemist Evaluator initialized:")
        print(f"  Device: {device}")
        print(f"  Batch size: {batch_size}")
    
    def evaluate_gsm8k(
        self,
        model: nn.Module,
        test_loader: Any,
        max_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Evaluate GSM8K mathematical reasoning performance.
        
        Args:
            model: Model to evaluate
            test_loader: GSM8K test data loader
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary of GSM8K metrics
        """
        print("Evaluating GSM8K performance...")
        
        model.eval()
        model.to(self.device)
        
        correct = 0
        total = 0
        total_loss = 0.0
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.get_pad_token_id())
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="GSM8K Evaluation")):
                if i * self.batch_size >= max_samples:
                    break
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                logits = outputs["logits"]
                
                # Shift for language modeling
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                # Compute loss
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                total_loss += loss.item()
                
                # Compute accuracy (simplified)
                predictions = shift_logits.argmax(dim=-1)
                correct += (predictions == shift_labels).sum().item()
                total += shift_labels.numel()
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / (i + 1) if i >= 0 else 0.0
        
        return {
            "gsm8k_accuracy": accuracy,
            "gsm8k_loss": avg_loss,
            "gsm8k_samples": total
        }
    
    def evaluate_story_generation(
        self,
        model: nn.Module,
        test_loader: Any,
        max_samples: int = 500
    ) -> Dict[str, float]:
        """
        Evaluate creative story generation performance.
        
        Args:
            model: Model to evaluate
            test_loader: Story generation test data loader
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary of story generation metrics
        """
        print("Evaluating story generation performance...")
        
        model.eval()
        model.to(self.device)
        
        total_bleu = 0.0
        total_samples = 0
        total_loss = 0.0
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.get_pad_token_id())
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Story Generation Evaluation")):
                if i * self.batch_size >= max_samples:
                    break
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                logits = outputs["logits"]
                
                # Shift for language modeling
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                # Compute loss
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                total_loss += loss.item()
                
                # Compute BLEU score (simplified)
                predictions = shift_logits.argmax(dim=-1)
                bleu_score = self._compute_bleu_score(predictions, shift_labels)
                total_bleu += bleu_score
                total_samples += 1
        
        avg_bleu = total_bleu / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / (i + 1) if i >= 0 else 0.0
        
        return {
            "story_bleu": avg_bleu,
            "story_loss": avg_loss,
            "story_samples": total_samples
        }
    
    def _compute_bleu_score(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """
        Compute simplified BLEU score.
        
        Args:
            predictions: Predicted token IDs
            targets: Target token IDs
            
        Returns:
            BLEU score
        """
        # Simplified BLEU computation
        # In practice, you'd want to use a proper BLEU implementation
        correct = (predictions == targets).sum().item()
        total = targets.numel()
        
        if total == 0:
            return 0.0
        
        return correct / total
    
    def evaluate_routing_system(
        self,
        specialists: List[SpecialistModel],
        router: CrucibleRouter,
        projections: MultiProjectionHeads,
        test_loader: Any,
        max_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Evaluate the complete routing system.
        
        Args:
            specialists: List of specialist models
            router: Crucible router
            projections: Projection heads
            test_loader: Test data loader
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary of routing system metrics
        """
        print("Evaluating routing system performance...")
        
        # Move all models to device
        for specialist in specialists:
            specialist.eval()
            specialist.to(self.device)
        
        router.eval()
        router.to(self.device)
        
        projections.eval()
        projections.to(self.device)
        
        total_loss = 0.0
        total_entropy = 0.0
        total_active_params = 0.0
        total_flops = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Routing System Evaluation")):
                if i * self.batch_size >= max_samples:
                    break
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Get specialist outputs
                specialist_outputs = []
                for specialist in specialists:
                    outputs = specialist(input_ids, attention_mask)
                    hidden_states = outputs["last_hidden_state"]
                    specialist_outputs.append(hidden_states)
                
                # Project to shared space
                projected_outputs = projections.forward_all(specialist_outputs)
                
                # Get router decision
                router_result = router(projected_outputs[0], return_entropy=True)
                
                # Compute metrics
                routing_loss = router.compute_routing_loss(
                    router_result["weights"],
                    router_result["entropy"]
                )
                
                # Calculate active parameters
                active_params = self._calculate_active_parameters(
                    specialists,
                    router_result["active_mask"]
                )
                
                # Estimate FLOPs
                flops = self._estimate_system_flops(
                    input_ids,
                    specialists,
                    router_result["active_mask"]
                )
                
                total_loss += routing_loss.item()
                total_entropy += router_result["entropy"].mean().item()
                total_active_params += active_params
                total_flops += flops
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_entropy = total_entropy / num_batches if num_batches > 0 else 0.0
        avg_active_params = total_active_params / num_batches if num_batches > 0 else 0.0
        avg_flops = total_flops / num_batches if num_batches > 0 else 0.0
        
        return {
            "routing_loss": avg_loss,
            "routing_entropy": avg_entropy,
            "active_params_mil": avg_active_params / 1e6,  # Convert to millions
            "flops_G": avg_flops / 1e9  # Convert to GFLOPs
        }
    
    def _calculate_active_parameters(
        self,
        specialists: List[SpecialistModel],
        active_mask: torch.Tensor
    ) -> float:
        """
        Calculate active parameters based on routing decisions.
        
        Args:
            specialists: List of specialist models
            active_mask: Active expert mask from router
            
        Returns:
            Number of active parameters
        """
        total_active_params = 0.0
        
        for i, specialist in enumerate(specialists):
            # Count parameters for this specialist
            specialist_params = count_parameters(specialist)
            
            # Calculate active fraction based on routing mask
            active_fraction = active_mask[:, :, i].float().mean().item()
            
            total_active_params += specialist_params * active_fraction
        
        return total_active_params
    
    def _estimate_system_flops(
        self,
        input_ids: torch.Tensor,
        specialists: List[SpecialistModel],
        active_mask: torch.Tensor
    ) -> float:
        """
        Estimate FLOPs for the complete system.
        
        Args:
            input_ids: Input token IDs
            specialists: List of specialist models
            active_mask: Active expert mask from router
            
        Returns:
            Estimated FLOPs
        """
        total_flops = 0.0
        
        for i, specialist in enumerate(specialists):
            # Estimate FLOPs for this specialist
            specialist_flops = estimate_flops(specialist, input_ids)
            
            # Calculate active fraction based on routing mask
            active_fraction = active_mask[:, :, i].float().mean().item()
            
            total_flops += specialist_flops * active_fraction
        
        return total_flops
    
    def compare_with_baseline(
        self,
        router_system_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compare router system with baseline.
        
        Args:
            router_system_metrics: Metrics from router system
            baseline_metrics: Metrics from baseline model
            
        Returns:
            Dictionary of comparison metrics
        """
        comparisons = {}
        
        # Accuracy comparisons
        if "gsm8k_accuracy" in router_system_metrics and "gsm8k_accuracy" in baseline_metrics:
            router_acc = router_system_metrics["gsm8k_accuracy"]
            baseline_acc = baseline_metrics["gsm8k_accuracy"]
            comparisons["gsm8k_accuracy_diff"] = router_acc - baseline_acc
            comparisons["gsm8k_accuracy_ratio"] = router_acc / baseline_acc if baseline_acc > 0 else 0
        
        if "story_bleu" in router_system_metrics and "story_bleu" in baseline_metrics:
            router_bleu = router_system_metrics["story_bleu"]
            baseline_bleu = baseline_metrics["story_bleu"]
            comparisons["story_bleu_diff"] = router_bleu - baseline_bleu
            comparisons["story_bleu_ratio"] = router_bleu / baseline_bleu if baseline_bleu > 0 else 0
        
        # Efficiency comparisons
        if "active_params_mil" in router_system_metrics and "active_params_mil" in baseline_metrics:
            router_params = router_system_metrics["active_params_mil"]
            baseline_params = baseline_metrics["active_params_mil"]
            comparisons["param_reduction"] = (baseline_params - router_params) / baseline_params
            comparisons["param_efficiency"] = router_params / baseline_params
        
        if "flops_G" in router_system_metrics and "flops_G" in baseline_metrics:
            router_flops = router_system_metrics["flops_G"]
            baseline_flops = baseline_metrics["flops_G"]
            comparisons["flop_reduction"] = (baseline_flops - router_flops) / baseline_flops
            comparisons["flop_efficiency"] = router_flops / baseline_flops
        
        return comparisons


def run_comprehensive_evaluation(
    model_paths: Dict[str, str],
    tokenizer_path: str,
    test_data_path: str,
    output_path: str = "evaluation_results.csv"
) -> None:
    """
    Run comprehensive evaluation for Phase 1.
    
    Args:
        model_paths: Dictionary of model paths
        tokenizer_path: Path to tokenizer
        test_data_path: Path to test data
        output_path: Path to save results
    """
    print("Running comprehensive Phase 1 evaluation...")
    
    # Load tokenizer
    tokenizer = SharedTokenizer(tokenizer_path)
    
    # Create evaluator
    evaluator = AlchemistEvaluator(tokenizer)
    
    # Load models
    specialists = []
    for domain in ["math", "creative"]:
        if f"{domain}_specialist" in model_paths:
            specialist = create_specialist(domain)
            specialist.load_checkpoint(model_paths[f"{domain}_specialist"])
            specialists.append(specialist)
    
    router = create_router(d_model=512, n_experts=2)
    if "router" in model_paths:
        router.load_state_dict(torch.load(model_paths["router"]))
    
    projections = create_projection_heads(d_model=512, d_shared=512, num_specialists=2)
    if "projections" in model_paths:
        projections.load_state_dict(torch.load(model_paths["projections"]))
    
    # Create baseline model (single 120M model)
    baseline_model = create_specialist("general")  # Use general specialist as baseline
    
    # Load test data
    # This would load actual test data in practice
    # For now, we'll create dummy data loaders
    
    # Evaluate router system
    router_metrics = evaluator.evaluate_routing_system(
        specialists,
        router,
        projections,
        None,  # test_loader would be provided in practice
        max_samples=100
    )
    
    # Evaluate baseline
    baseline_metrics = evaluator.evaluate_gsm8k(
        baseline_model,
        None,  # test_loader would be provided in practice
        max_samples=100
    )
    
    # Compare results
    comparisons = evaluator.compare_with_baseline(router_metrics, baseline_metrics)
    
    # Combine all metrics
    all_metrics = {
        "model": "router_system",
        **router_metrics,
        **comparisons
    }
    
    baseline_all_metrics = {
        "model": "baseline_120m",
        **baseline_metrics
    }
    
    # Save results
    results = [all_metrics, baseline_all_metrics]
    
    with open(output_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"Evaluation results saved to: {output_path}")
    
    # Print summary
    print("\n=== Phase 1 Evaluation Summary ===")
    print(f"Router System:")
    for key, value in all_metrics.items():
        if key != "model":
            print(f"  {key}: {value:.4f}")
    
    print(f"\nBaseline System:")
    for key, value in baseline_all_metrics.items():
        if key != "model":
            print(f"  {key}: {value:.4f}")


def test_evaluation_harness() -> None:
    """Test the evaluation harness functionality."""
    print("Testing evaluation harness...")
    
    # Create dummy models and evaluator
    tokenizer = SharedTokenizer("dummy_path")  # Would be real path in practice
    evaluator = AlchemistEvaluator(tokenizer)
    
    # Create dummy specialists
    math_specialist = create_specialist("math")
    creative_specialist = create_specialist("creative")
    specialists = [math_specialist, creative_specialist]
    
    # Create router and projections
    router = create_router(d_model=512, n_experts=2)
    projections = create_projection_heads(d_model=512, d_shared=512, num_specialists=2)
    
    # Test routing system evaluation
    dummy_metrics = evaluator.evaluate_routing_system(
        specialists,
        router,
        projections,
        None,  # No test loader for testing
        max_samples=10
    )
    
    print("Routing system evaluation test completed!")
    print("Dummy metrics:", dummy_metrics) 