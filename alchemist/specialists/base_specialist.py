"""
Base specialist model for the Alchemist Architecture.

Implements the Mistral-style transformer architecture with 60M parameters
and freeze capability for routing experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from transformers import AutoConfig, AutoModelForCausalLM


class SpecialistConfig:
    """
    Configuration for specialist models.
    
    Based on Mistral architecture with 60M parameters for Phase 1.
    """
    
    def __init__(
        self,
        vocab_size: int = 650,  # Maximum for dummy datasets
        hidden_size: int = 256,  # Reduced for 6GB GPU
        intermediate_size: int = 512,  # Reduced for 6GB GPU
        num_hidden_layers: int = 6,  # Reduced for 6GB GPU
        num_attention_heads: int = 4,  # Reduced for 6GB GPU
        max_position_embeddings: int = 1024,  # Reduced for 6GB GPU
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict] = None,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        classifier_dropout: float = 0.0,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.torch_dtype = torch_dtype
        
        # Calculate parameters for 60M model
        self.total_params = self._calculate_parameters()
    
    def _calculate_parameters(self) -> int:
        """Calculate total parameters for the configuration."""
        # Embedding layers
        embedding_params = self.vocab_size * self.hidden_size
        
        # Transformer layers
        layer_params = 0
        for _ in range(self.num_hidden_layers):
            # Self-attention
            layer_params += 4 * self.hidden_size * self.hidden_size  # Q, K, V, O
            layer_params += 4 * self.hidden_size  # Layer norms
            
            # MLP
            layer_params += self.hidden_size * self.intermediate_size * 2  # Up and down projections
            layer_params += self.intermediate_size + self.hidden_size  # Biases
        
        # Final layer norm
        layer_params += self.hidden_size
        
        # LM head
        lm_head_params = self.vocab_size * self.hidden_size
        
        total = embedding_params + layer_params + lm_head_params
        return total


class SpecialistModel(nn.Module):
    """
    Base specialist model for the Alchemist Architecture.
    
    Implements a Mistral-style transformer with 60M parameters and
    freeze capability for routing experiments.
    """
    
    def __init__(self, config: SpecialistConfig):
        """
        Initialize the specialist model.
        
        Args:
            config: Specialist configuration
        """
        super().__init__()
        self.config = config
        self.freeze_flag = False  # Debug knob for routing failures
        
        # Create the transformer model
        self.transformer = self._create_transformer()
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"Specialist model initialized:")
        print(f"  Parameters: {self._count_parameters():,}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Attention heads: {config.num_attention_heads}")
    
    def _create_transformer(self) -> nn.Module:
        """Create the transformer model using HuggingFace."""
        # Create config for HuggingFace
        hf_config = AutoConfig.from_pretrained(
            "microsoft/DialoGPT-medium",  # Use as base, we'll modify
            vocab_size=self.config.vocab_size,
            n_positions=self.config.max_position_embeddings,
            n_embd=self.config.hidden_size,
            n_layer=self.config.num_hidden_layers,
            n_head=self.config.num_attention_heads,
            pad_token_id=self.config.pad_token_id,
            bos_token_id=self.config.bos_token_id,
            eos_token_id=self.config.eos_token_id,
            tie_word_embeddings=self.config.tie_word_embeddings,
            torch_dtype=self.config.torch_dtype,
        )
        
        # Create model
        model = AutoModelForCausalLM.from_config(hf_config)
        
        return model
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for the model."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _count_parameters(self) -> int:
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        freeze_all_but_final_ln: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the specialist model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            freeze_all_but_final_ln: Whether to freeze all layers except final layer norm
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing logits and hidden states
        """
        if freeze_all_but_final_ln:
            # Freeze all parameters except final layer norm
            for name, param in self.named_parameters():
                if "ln_f" not in name:  # Final layer norm
                    param.requires_grad = False
        
        # Forward pass through transformer (with hidden states output)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Extract last hidden state from hidden_states if available
        last_hidden_state = None
        if outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]  # Last layer
        
        return {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,
            "last_hidden_state": last_hidden_state
        }
    
    def get_hidden_states(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Get hidden states from a specific layer.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            layer_idx: Layer index (-1 for last layer)
            
        Returns:
            Hidden states from the specified layer
        """
        with torch.no_grad():
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            if layer_idx == -1:
                return outputs.last_hidden_state
            else:
                return outputs.hidden_states[layer_idx]
    
    def freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.freeze_flag = True
    
    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self.freeze_flag = False
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'freeze_flag': self.freeze_flag
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.freeze_flag = checkpoint.get('freeze_flag', False)

    def gradient_checkpointing_enable(self):
        print("⚠️  skipping grad-ckpt; not supported for this backbone")


class MathSpecialist(SpecialistModel):
    """Mathematical reasoning specialist trained on GSM8K."""
    
    def __init__(self, config: Optional[SpecialistConfig] = None):
        if config is None:
            config = SpecialistConfig()
        super().__init__(config)
        self.domain = "math"


class CreativeSpecialist(SpecialistModel):
    """Creative writing specialist trained on story generation."""
    
    def __init__(self, config: Optional[SpecialistConfig] = None):
        if config is None:
            config = SpecialistConfig()
        super().__init__(config)
        self.domain = "creative"


def create_specialist(domain: str, config: Optional[SpecialistConfig] = None) -> SpecialistModel:
    """
    Create a specialist model for the specified domain.
    
    Args:
        domain: Domain name ('math', 'creative', 'general')
        config: Optional configuration
        
    Returns:
        Specialist model instance
    """
    if config is None:
        config = SpecialistConfig()
    
    if domain == "math":
        return MathSpecialist(config)
    elif domain == "creative":
        return CreativeSpecialist(config)
    elif domain == "general":
        return SpecialistModel(config)  # Generic specialist
    else:
        raise ValueError(f"Unknown domain: {domain}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(
    model: nn.Module,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor] = None
) -> int:
    """
    Estimate FLOPs for a forward pass.
    
    This is a rough estimation based on model architecture.
    For precise measurement, use PyTorch profiler.
    """
    batch_size, seq_len = input_ids.shape
    config = model.config
    
    # Embedding FLOPs
    embedding_flops = batch_size * seq_len * config.hidden_size
    
    # Transformer layer FLOPs
    layer_flops = 0
    for _ in range(config.num_hidden_layers):
        # Self-attention
        layer_flops += 4 * batch_size * seq_len * config.hidden_size * config.hidden_size  # Q, K, V, O
        layer_flops += batch_size * seq_len * seq_len * config.hidden_size  # Attention computation
        
        # MLP
        layer_flops += 2 * batch_size * seq_len * config.hidden_size * config.intermediate_size
    
    # Final layer norm and LM head
    final_flops = batch_size * seq_len * config.hidden_size * config.vocab_size
    
    total_flops = embedding_flops + layer_flops + final_flops
    return total_flops 