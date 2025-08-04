"""
Memory Specialist: Encoder-decoder for knowledge graph interaction.

Takes JSON tuples like {"user":"Jeff", "text":"I play ukulele"} and produces
256-dimensional vectors for routing and knowledge retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import json
from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """Configuration for Memory Specialist (10M parameters target)."""
    vocab_size: int = 650  # Match Phase 1 tokenizer
    hidden_size: int = 256  # Match projection dimension
    encoder_layers: int = 4  # Smaller than main specialists
    decoder_layers: int = 4
    num_attention_heads: int = 8
    intermediate_size: int = 512
    max_position_embeddings: int = 512
    dropout: float = 0.1
    memory_dim: int = 256  # Output dimension for memory vectors


class MemoryEncoder(nn.Module):
    """Encoder for processing user context and queries."""
    
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode input sequence to hidden states.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            encoded_states: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        token_embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        
        # Create attention mask for transformer (inverted)
        if attention_mask is not None:
            # Transformer expects True for positions to attend to
            transformer_mask = attention_mask.bool()
        else:
            transformer_mask = None
            
        # Encode
        encoded = self.transformer(embeddings, src_key_padding_mask=~transformer_mask if transformer_mask is not None else None)
        
        return encoded


class MemoryDecoder(nn.Module):
    """Decoder for generating persona vectors and responses."""
    
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        
        # Memory projection head
        self.memory_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.memory_dim),
            nn.LayerNorm(config.memory_dim)
        )
        
        # Optional: Language modeling head for response generation
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, encoded_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Decode encoded states to memory vectors and optionally logits.
        
        Args:
            encoded_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            Dict with 'memory_vector' and optionally 'logits'
        """
        # Use last token representation for memory vector (like BERT [CLS])
        if attention_mask is not None:
            # Find last non-padded token for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            last_hidden = encoded_states[torch.arange(encoded_states.size(0)), seq_lengths]
        else:
            # Use last token
            last_hidden = encoded_states[:, -1, :]
        
        # Generate memory vector
        memory_vector = self.memory_projection(last_hidden)
        
        # Optional: generate logits for language modeling
        logits = self.lm_head(encoded_states)
        
        return {
            'memory_vector': memory_vector,  # [batch_size, memory_dim]
            'logits': logits,  # [batch_size, seq_len, vocab_size]
            'last_hidden_state': last_hidden.unsqueeze(1)  # For compatibility with specialists
        }


class MemorySpecialist(nn.Module):
    """
    Complete Memory Specialist: Encoder-Decoder for knowledge graph interaction.
    
    Target: ~10M parameters
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        super().__init__()
        self.config = config or MemoryConfig()
        
        self.encoder = MemoryEncoder(self.config)
        self.decoder = MemoryDecoder(self.config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Memory Specialist initialized:")
        print(f"  Parameters: {total_params:,}")
        print(f"  Target: ~10M parameters")
        print(f"  Hidden size: {self.config.hidden_size}")
        print(f"  Memory dimension: {self.config.memory_dim}")
        
    def _init_weights(self, module):
        """Initialize weights (similar to GPT-2)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through memory specialist.
        
        Args:
            input_ids: [batch_size, seq_len] - tokenized input
            attention_mask: [batch_size, seq_len] - attention mask
            
        Returns:
            Dict with memory_vector, logits, and last_hidden_state
        """
        # Encode
        encoded_states = self.encoder(input_ids, attention_mask)
        
        # Decode
        outputs = self.decoder(encoded_states, attention_mask)
        
        return outputs
    
    def encode_json_input(self, json_data: Dict[str, Any], tokenizer) -> torch.Tensor:
        """
        Encode JSON input like {"user":"Jeff", "text":"I play ukulele"} to memory vector.
        
        Args:
            json_data: Dictionary with user context
            tokenizer: Tokenizer to use
            
        Returns:
            memory_vector: [1, memory_dim]
        """
        # Convert JSON to text
        if isinstance(json_data, dict):
            # Format: "User: Jeff. Text: I play ukulele."
            text_parts = []
            if 'user' in json_data:
                text_parts.append(f"User: {json_data['user']}")
            if 'text' in json_data:
                text_parts.append(f"Text: {json_data['text']}")
            text = ". ".join(text_parts) + "."
        else:
            text = str(json_data)
        
        # Tokenize
        tokens = tokenizer.encode(text)
        input_ids = torch.tensor([tokens], device=next(self.parameters()).device)
        attention_mask = torch.ones_like(input_ids)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs['memory_vector']
    
    def freeze_parameters(self):
        """Freeze all parameters for training other components."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_parameters(self):
        """Unfreeze all parameters for training."""
        for param in self.parameters():
            param.requires_grad = True


def create_memory_specialist(config: Optional[MemoryConfig] = None) -> MemorySpecialist:
    """Factory function to create memory specialist."""
    return MemorySpecialist(config)


# Parameter estimation
def estimate_memory_params():
    """Estimate parameter count for memory specialist."""
    config = MemoryConfig()
    
    # Encoder parameters
    embedding_params = config.vocab_size * config.hidden_size  # Token embeddings
    position_params = config.max_position_embeddings * config.hidden_size  # Position embeddings
    
    # Transformer encoder (4 layers)
    # Each layer: self-attention + feed-forward
    attention_params_per_layer = (
        3 * config.hidden_size * config.hidden_size +  # Q, K, V projections
        config.hidden_size * config.hidden_size +      # Output projection
        4 * config.hidden_size                         # Layer norms and biases
    )
    feedforward_params_per_layer = (
        config.hidden_size * config.intermediate_size +  # Up projection
        config.intermediate_size * config.hidden_size +  # Down projection
        config.intermediate_size + config.hidden_size    # Biases
    )
    encoder_params = config.encoder_layers * (attention_params_per_layer + feedforward_params_per_layer)
    
    # Decoder parameters
    memory_projection_params = (
        config.hidden_size * config.intermediate_size +  # First linear
        config.intermediate_size * config.memory_dim +   # Second linear
        config.intermediate_size + config.memory_dim +   # Biases
        config.memory_dim * 2                           # Layer norm
    )
    lm_head_params = config.hidden_size * config.vocab_size
    
    total = embedding_params + position_params + encoder_params + memory_projection_params + lm_head_params
    
    print(f"Memory Specialist Parameter Estimation:")
    print(f"  Embeddings: {embedding_params:,}")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Decoder: {memory_projection_params + lm_head_params:,}")
    print(f"  Total: {total:,}")
    print(f"  Target: ~10M")
    
    return total


if __name__ == "__main__":
    # Test the memory specialist
    estimate_memory_params()
    
    # Create and test
    memory_specialist = create_memory_specialist()
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 650, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    outputs = memory_specialist(input_ids, attention_mask)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Memory vector shape: {outputs['memory_vector'].shape}")
    print(f"  Logits shape: {outputs['logits'].shape}")