"""
Projection heads for the Alchemist Architecture.

Maps specialist hidden states to a shared vector space for routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ProjectionHead(nn.Module):
    """
    Projection head for mapping specialist hidden states to shared space.
    
    Maps R^512 → R^512_shared with weight normalization for stability.
    """
    
    def __init__(self, d_model: int = 512, d_shared: int = 512):
        """
        Initialize the projection head.
        
        Args:
            d_model: Input dimension (specialist hidden size)
            d_shared: Output dimension (shared vector space)
        """
        super().__init__()
        self.d_model = d_model
        self.d_shared = d_shared
        
        # Linear projection with weight normalization
        self.proj = nn.Linear(d_model, d_shared)
        self.norm = nn.LayerNorm(d_shared)  # Stability for contrastive loss
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights for stability."""
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to shared space.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, d_model]
            
        Returns:
            Projected vectors [batch_size, seq_len, d_shared]
        """
        # Apply linear projection
        projected = self.proj(hidden_states)
        
        # Apply layer normalization for stability
        normalized = self.norm(projected)
        
        return normalized
    
    def get_projection_dim(self) -> int:
        """Get the output dimension of the projection."""
        return self.d_shared


class MultiProjectionHeads(nn.Module):
    """
    Collection of projection heads for multiple specialists.
    
    Each specialist gets its own projection head to map to shared space.
    """
    
    def __init__(self, d_model: int = 512, d_shared: int = 512, num_specialists: int = 2):
        """
        Initialize multiple projection heads.
        
        Args:
            d_model: Input dimension (specialist hidden size)
            d_shared: Output dimension (shared vector space)
            num_specialists: Number of specialists
        """
        super().__init__()
        self.d_model = d_model
        self.d_shared = d_shared
        self.num_specialists = num_specialists
        
        # Create projection heads for each specialist
        self.projections = nn.ModuleDict({
            f"specialist_{i}": ProjectionHead(d_model, d_shared)
            for i in range(num_specialists)
        })
    
    def forward(self, hidden_states: torch.Tensor, specialist_id: int) -> torch.Tensor:
        """
        Project hidden states for a specific specialist.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, d_model]
            specialist_id: ID of the specialist (0 to num_specialists-1)
            
        Returns:
            Projected vectors [batch_size, seq_len, d_shared]
        """
        projection_key = f"specialist_{specialist_id}"
        
        if projection_key not in self.projections:
            raise ValueError(f"Specialist {specialist_id} not found in projections")
        
        return self.projections[projection_key](hidden_states)
    
    def forward_all(self, hidden_states_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Project hidden states for all specialists.
        
        Args:
            hidden_states_list: List of hidden states for each specialist
            
        Returns:
            List of projected vectors for each specialist
        """
        if len(hidden_states_list) != self.num_specialists:
            raise ValueError(f"Expected {self.num_specialists} specialists, got {len(hidden_states_list)}")
        
        projected_list = []
        for i, hidden_states in enumerate(hidden_states_list):
            projected = self.forward(hidden_states, i)
            projected_list.append(projected)
        
        return projected_list
    
    def get_projection_dim(self) -> int:
        """Get the output dimension of the projections."""
        return self.d_shared


class ContrastiveProjectionHead(nn.Module):
    """
    Projection head optimized for contrastive learning.
    
    Uses temperature scaling and normalization for better contrastive loss.
    """
    
    def __init__(self, d_model: int = 512, d_shared: int = 512, temperature: float = 0.1):
        """
        Initialize contrastive projection head.
        
        Args:
            d_model: Input dimension
            d_shared: Output dimension
            temperature: Temperature for contrastive learning
        """
        super().__init__()
        self.d_model = d_model
        self.d_shared = d_shared
        self.temperature = temperature
        
        # Multi-layer projection for better representation
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_shared),
            nn.ReLU(),
            nn.Linear(d_shared, d_shared)
        )
        
        # Normalization for contrastive learning
        self.norm = nn.LayerNorm(d_shared)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights for contrastive learning."""
        for layer in self.proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states with contrastive optimization.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, d_model]
            
        Returns:
            Projected vectors [batch_size, seq_len, d_shared]
        """
        # Apply multi-layer projection
        projected = self.proj(hidden_states)
        
        # Apply normalization
        normalized = self.norm(projected)
        
        # Apply temperature scaling
        scaled = normalized / self.temperature
        
        return scaled
    
    def compute_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between projected vectors.
        
        Args:
            vec1: First projected vector [batch_size, seq_len, d_shared]
            vec2: Second projected vector [batch_size, seq_len, d_shared]
            
        Returns:
            Similarity scores [batch_size, seq_len]
        """
        # Normalize vectors
        vec1_norm = F.normalize(vec1, p=2, dim=-1)
        vec2_norm = F.normalize(vec2, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.sum(vec1_norm * vec2_norm, dim=-1)
        
        return similarity


def create_projection_heads(
    d_model: int = 512,
    d_shared: int = 512,
    num_specialists: int = 3,  # PHASE 2: Math, Creative, Memory
    use_contrastive: bool = False
) -> MultiProjectionHeads:
    """
    Create projection heads for specialists.
    
    Args:
        d_model: Input dimension
        d_shared: Output dimension
        num_specialists: Number of specialists
        use_contrastive: Whether to use contrastive projection heads
        
    Returns:
        MultiProjectionHeads instance
    """
    if use_contrastive:
        # For contrastive learning, we'd need a different approach
        # This is a placeholder for future implementation
        raise NotImplementedError("Contrastive projection heads not yet implemented")
    
    return MultiProjectionHeads(d_model, d_shared, num_specialists)


def test_projection_heads() -> None:
    """Test the projection heads functionality."""
    print("Testing projection heads...")
    
    # Create projection heads
    d_model = 512
    d_shared = 512
    num_specialists = 2
    
    projections = create_projection_heads(d_model, d_shared, num_specialists)
    
    # Create dummy hidden states
    batch_size = 4
    seq_len = 128
    
    hidden_states_1 = torch.randn(batch_size, seq_len, d_model)
    hidden_states_2 = torch.randn(batch_size, seq_len, d_model)
    
    # Test individual projection
    projected_1 = projections.forward(hidden_states_1, 0)
    projected_2 = projections.forward(hidden_states_2, 1)
    
    print(f"Input shape: {hidden_states_1.shape}")
    print(f"Output shape: {projected_1.shape}")
    print(f"Projection dimension: {projections.get_projection_dim()}")
    
    # Test batch projection
    hidden_states_list = [hidden_states_1, hidden_states_2]
    projected_list = projections.forward_all(hidden_states_list)
    
    print(f"Number of projected outputs: {len(projected_list)}")
    print(f"All outputs have same shape: {all(p.shape == projected_1.shape for p in projected_list)}")
    
    print("Projection heads test completed successfully!") 