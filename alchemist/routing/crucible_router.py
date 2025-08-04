"""
Crucible Router for the Alchemist Architecture.

Learns dynamic mixing weights for specialist outputs with sparse activation
and entropy regularization for routing decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class CrucibleRouter(nn.Module):
    """
    Crucible Router for dynamic specialist routing.
    
    Learns to route queries to appropriate specialists using a two-layer MLP
    with entropy regularization and sparse activation.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_experts: int = 3,  # PHASE 2: Math, Creative, Memory
        hidden_size: int = 256,
        entropy_beta: float = 0.005,  # Reduced from 0.02 to allow specialization
        skip_threshold: float = 0.15  # Increased from 0.05 to make skipping painful
    ):
        """
        Initialize the Crucible Router.
        
        Args:
            d_model: Input dimension (shared vector space)
            n_experts: Number of specialists
            hidden_size: Hidden layer size for MLP
            entropy_beta: Entropy regularization coefficient
            skip_threshold: Threshold for sparse activation
        """
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.hidden_size = hidden_size
        self.entropy_beta = entropy_beta
        self.skip_threshold = skip_threshold
        
        # Two-layer MLP for routing decisions
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, n_experts)
        )
        
        # Initialize weights
        self._init_weights()
        
        print(f"Crucible Router initialized:")
        print(f"  Input dimension: {d_model}")
        print(f"  Number of experts: {n_experts}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Entropy beta: {entropy_beta}")
        print(f"  Skip threshold: {skip_threshold}")
    
    def _init_weights(self) -> None:
        """Initialize weights for the router."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        return_entropy: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the router.
        
        Args:
            query_embedding: Query embedding [batch_size, seq_len, d_model]
            return_entropy: Whether to return entropy for regularization
            
        Returns:
            Dictionary containing routing weights and active mask
        """
        # Get routing logits
        logits = self.mlp(query_embedding)  # [batch_size, seq_len, n_experts]
        
        # Apply softmax to get routing weights
        weights = F.softmax(logits, dim=-1)  # [batch_size, seq_len, n_experts]
        
        # Create sparse activation mask
        active_mask = weights > self.skip_threshold  # [batch_size, seq_len, n_experts]
        
        # Normalize weights for active experts only
        masked_weights = weights * active_mask.float()
        weight_sums = masked_weights.sum(dim=-1, keepdim=True)
        normalized_weights = masked_weights / (weight_sums + 1e-8)
        
        result = {
            "weights": normalized_weights,
            "active_mask": active_mask,
            "raw_weights": weights
        }
        
        if return_entropy:
            # Calculate entropy for regularization
            entropy = self._compute_entropy(weights)
            result["entropy"] = entropy
        
        return result
    
    def _compute_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of routing weights.
        
        Args:
            weights: Routing weights [batch_size, seq_len, n_experts]
            
        Returns:
            Entropy values [batch_size, seq_len]
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        log_weights = torch.log(weights + eps)
        
        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(weights * log_weights, dim=-1)
        
        return entropy
    
    def compute_routing_loss(
        self,
        weights: torch.Tensor,
        entropy: torch.Tensor,
        target_entropy: float = 1.0,
        entropy_beta: Optional[float] = None,
        confidence_gamma: float = 0.1  # 10x boost - MAXIMUM CONFIDENCE!
    ) -> torch.Tensor:
        """
        Compute routing loss with entropy regularization.
        
        Args:
            weights: Routing weights [batch_size, seq_len, n_experts]
            entropy: Entropy values [batch_size, seq_len]
            target_entropy: Target entropy for regularization
            
        Returns:
            Routing loss
        """
        # Entropy regularization: encourage diverse routing (with optional annealing)
        beta = entropy_beta if entropy_beta is not None else self.entropy_beta
        entropy_loss = beta * F.mse_loss(entropy.mean(), torch.tensor(target_entropy, device=entropy.device))
        
        # Load balancing loss: DISABLED - was fighting specialization
        # expert_usage = weights.mean(dim=[0, 1])  # [n_experts]
        # target_usage = torch.ones_like(expert_usage) / self.n_experts
        # load_balance_loss = F.mse_loss(expert_usage, target_usage)
        lb_weight = 0.0  # Kill the load balancer!
        load_balance_loss = torch.tensor(0.0, device=weights.device)
        
        # Expert confidence reward: encourage clear winners
        max_weights, _ = weights.max(dim=-1)  # [batch_size, seq_len]
        sorted_weights, _ = weights.sort(dim=-1, descending=True)
        second_weights = sorted_weights[..., 1]  # Second highest weight
        
        confidence_bonus = (max_weights - second_weights).detach()  # Stop gradients
        confidence_reward = confidence_gamma * confidence_bonus.mean()
        
        # Combine losses (load balancer KILLED!)
        total_loss = entropy_loss + lb_weight * load_balance_loss - confidence_reward
        
        return total_loss
    
    def get_routing_stats(self, weights: torch.Tensor, active_mask: torch.Tensor) -> Dict[str, float]:
        """
        Get routing statistics for analysis.
        
        Args:
            weights: Routing weights [batch_size, seq_len, n_experts]
            active_mask: Active expert mask [batch_size, seq_len, n_experts]
            
        Returns:
            Dictionary of routing statistics
        """
        # Expert usage statistics
        expert_usage = weights.mean(dim=[0, 1]).detach().cpu().numpy()
        
        # Sparsity statistics
        sparsity = (1 - active_mask.float().mean()).item()
        
        # Entropy statistics
        entropy = self._compute_entropy(weights).mean().item()
        
        # Routing diversity
        max_expert_usage = expert_usage.max()
        min_expert_usage = expert_usage.min()
        usage_ratio = max_expert_usage / (min_expert_usage + 1e-8)
        
        return {
            "expert_usage": expert_usage.tolist(),
            "sparsity": sparsity,
            "entropy": entropy,
            "usage_ratio": usage_ratio,
            "max_usage": max_expert_usage,
            "min_usage": min_expert_usage
        }


class EnhancedRouter(CrucibleRouter):
    """
    Enhanced router with additional features for Phase 2+.
    
    Includes memory integration and negative signal routing.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_experts: int = 3,  # +1 for memory
        hidden_size: int = 256,
        entropy_beta: float = 0.02,
        skip_threshold: float = 0.05,
        memory_weight: float = 0.3
    ):
        """
        Initialize the enhanced router.
        
        Args:
            d_model: Input dimension
            n_experts: Number of experts (including memory)
            hidden_size: Hidden layer size
            entropy_beta: Entropy regularization coefficient
            skip_threshold: Sparse activation threshold
            memory_weight: Weight for memory integration
        """
        super().__init__(d_model, n_experts, hidden_size, entropy_beta, skip_threshold)
        self.memory_weight = nn.Parameter(torch.tensor(memory_weight))
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        memory_vector: Optional[torch.Tensor] = None,
        return_entropy: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional memory integration.
        
        Args:
            query_embedding: Query embedding
            memory_vector: Optional memory vector
            return_entropy: Whether to return entropy
            
        Returns:
            Dictionary containing routing weights and active mask
        """
        # Standard routing
        result = super().forward(query_embedding, return_entropy)
        
        # Integrate memory if provided
        if memory_vector is not None:
            # Combine query and memory embeddings
            combined_embedding = query_embedding + self.memory_weight * memory_vector
            
            # Recompute routing with memory
            memory_result = super().forward(combined_embedding, return_entropy)
            
            # Update result with memory-influenced routing
            result.update(memory_result)
            result["memory_influenced"] = True
        
        return result


def create_router(
    d_model: int = 512,
    n_experts: int = 2,
    enhanced: bool = False,
    **kwargs
) -> CrucibleRouter:
    """
    Create a router instance.
    
    Args:
        d_model: Input dimension
        n_experts: Number of experts
        enhanced: Whether to use enhanced router
        **kwargs: Additional arguments
        
    Returns:
        Router instance
    """
    if enhanced:
        return EnhancedRouter(d_model, n_experts, **kwargs)
    else:
        return CrucibleRouter(d_model, n_experts, **kwargs)


def test_router() -> None:
    """Test the Crucible Router functionality."""
    print("Testing Crucible Router...")
    
    # Create router
    d_model = 512
    n_experts = 2
    
    router = create_router(d_model, n_experts)
    
    # Create dummy query embedding
    batch_size = 4
    seq_len = 128
    
    query_embedding = torch.randn(batch_size, seq_len, d_model)
    
    # Test forward pass
    result = router.forward(query_embedding, return_entropy=True)
    
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Routing weights shape: {result['weights'].shape}")
    print(f"Active mask shape: {result['active_mask'].shape}")
    print(f"Entropy shape: {result['entropy'].shape}")
    
    # Test routing statistics
    stats = router.get_routing_stats(result['weights'], result['active_mask'])
    
    print(f"Routing statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test routing loss
    loss = router.compute_routing_loss(
        result['weights'],
        result['entropy']
    )
    
    print(f"Routing loss: {loss.item():.4f}")
    
    print("Crucible Router test completed successfully!")


def analyze_routing_behavior(
    router: CrucibleRouter,
    test_embeddings: torch.Tensor,
    num_samples: int = 100
) -> Dict[str, np.ndarray]:
    """
    Analyze routing behavior across different inputs.
    
    Args:
        router: Router to analyze
        test_embeddings: Test embeddings
        num_samples: Number of samples to analyze
        
    Returns:
        Dictionary of analysis results
    """
    router.eval()
    
    all_weights = []
    all_entropies = []
    all_sparsities = []
    
    with torch.no_grad():
        for i in range(0, min(num_samples, len(test_embeddings)), 4):
            batch = test_embeddings[i:i+4]
            result = router.forward(batch, return_entropy=True)
            
            all_weights.append(result['weights'].cpu().numpy())
            all_entropies.append(result['entropy'].cpu().numpy())
            
            # Calculate sparsity
            sparsity = (1 - result['active_mask'].float().mean()).item()
            all_sparsities.append(sparsity)
    
    # Aggregate results
    weights_array = np.concatenate(all_weights, axis=0)
    entropies_array = np.concatenate(all_entropies, axis=0)
    
    return {
        "weights": weights_array,
        "entropies": entropies_array,
        "sparsities": np.array(all_sparsities),
        "mean_entropy": entropies_array.mean(),
        "mean_sparsity": np.mean(all_sparsities),
        "expert_usage": weights_array.mean(axis=(0, 1))
    } 