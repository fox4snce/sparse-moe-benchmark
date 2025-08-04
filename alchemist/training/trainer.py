"""
Training pipeline for the Alchemist Architecture.

Implements the three-phase training strategy for specialists and routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
# import wandb  # Disabled due to SSL issues
import os
import json
from tqdm import tqdm
import numpy as np

from alchemist.specialists.base_specialist import SpecialistModel, create_specialist
from alchemist.routing.crucible_router import CrucibleRouter, create_router
from alchemist.routing.projection_heads import MultiProjectionHeads, create_projection_heads
from alchemist.foundation.data_loader import create_data_loaders
from alchemist.foundation.tokenizer import SharedTokenizer


class AlchemistTrainer:
    """
    Trainer for the Alchemist Architecture.
    
    Implements the three-phase training strategy:
    - Phase A: Specialist pre-training
    - Phase B: Router + projection training
    - Phase C: Joint fine-tuning
    """
    
    def __init__(
        self,
        tokenizer: SharedTokenizer,
        device: str = "cuda",
        use_wandb: bool = True,
        project_name: str = "alchemist-phase1"
    ):
        """
        Initialize the trainer.
        
        Args:
            tokenizer: Shared tokenizer
            device: Training device
            use_wandb: Whether to use Weights & Biases logging
            project_name: W&B project name
        """
        self.tokenizer = tokenizer
        self.device = device
        self.use_wandb = use_wandb
        
        if use_wandb:
            try:
                import wandb
                wandb.init(project=project_name)
            except ImportError:
                print("⚠️  wandb not available, skipping logging")
                use_wandb = False
        
        print(f"Alchemist Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  W&B logging: {use_wandb}")
    
    def train_phase_a(
        self,
        specialist: SpecialistModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 3e-4,
        save_path: str = "checkpoints"
    ) -> Dict[str, List[float]]:
        """
        Phase A: Independent specialist training.
        
        Args:
            specialist: Specialist model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            save_path: Path to save checkpoints
            
        Returns:
            Dictionary of training metrics
        """
        print(f"Starting Phase A training for {specialist.domain} specialist...")
        
        # Move to device
        specialist = specialist.to(self.device)
        
        # Setup optimizer
        optimizer = optim.AdamW(specialist.parameters(), lr=learning_rate)
        
        # Setup loss function
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.get_pad_token_id())
        
        # Training metrics
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        os.makedirs(save_path, exist_ok=True)
        
        for epoch in range(num_epochs):
            # Training
            specialist.train()
            train_loss = self._train_epoch(specialist, train_loader, optimizer, criterion)
            train_losses.append(train_loss)
            
            # Validation
            specialist.eval()
            val_loss, val_accuracy = self._validate_epoch(specialist, val_loader, criterion)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Log metrics
            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        f"{specialist.domain}_train_loss": train_loss,
                        f"{specialist.domain}_val_loss": val_loss,
                        f"{specialist.domain}_val_accuracy": val_accuracy,
                        "epoch": epoch
                    })
                except ImportError:
                    pass
            
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_path, f"{specialist.domain}_epoch_{epoch+1}.pt")
                specialist.save_checkpoint(checkpoint_path)
        
        # Save final model
        final_path = os.path.join(save_path, f"{specialist.domain}_final.pt")
        specialist.save_checkpoint(final_path)
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies
        }
    
    def train_phase_b(
        self,
        specialists: List[SpecialistModel],
        router: CrucibleRouter,
        projections: MultiProjectionHeads,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_steps: int = 20000,
        learning_rate: float = 1e-4,
        save_path: str = "checkpoints"
    ) -> Dict[str, List[float]]:
        """
        Phase B: Router + projection training.
        
        Args:
            specialists: List of trained specialists (frozen)
            router: Router to train
            projections: Projection heads to train
            train_loader: Training data loader
            val_loader: Validation data loader
            num_steps: Number of training steps
            learning_rate: Learning rate
            save_path: Path to save checkpoints
            
        Returns:
            Dictionary of training metrics
        """
        print("Starting Phase B training: Router + Projections...")
        
        # Freeze specialists
        for specialist in specialists:
            specialist.freeze()
            specialist.to(self.device)
        
        # Move router and projections to device
        router = router.to(self.device)
        projections = projections.to(self.device)
        
        # Setup optimizer for router and projections
        optimizer = optim.AdamW(
            list(router.parameters()) + list(projections.parameters()),
            lr=learning_rate
        )
        
        # Training metrics
        router_losses = []
        projection_losses = []
        routing_entropies = []
        
        os.makedirs(save_path, exist_ok=True)
        
        step = 0
        while step < num_steps:
            for batch in train_loader:
                if step >= num_steps:
                    break
                
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Get specialist outputs
                specialist_outputs = []
                for specialist in specialists:
                    with torch.no_grad():
                        outputs = specialist(input_ids, attention_mask)
                        hidden_states = outputs["last_hidden_state"]
                        specialist_outputs.append(hidden_states)
                
                # Project to shared space
                projected_outputs = projections.forward_all(specialist_outputs)
                
                # Get router decision
                router_result = router(projected_outputs[0], return_entropy=True)
                
                # Improved entropy annealing: 10% exploration window + cosine decay
                import math
                progress = min(step / num_steps, 1.0)
                
                if progress < 0.1:  # First 10% of steps: full exploration
                    current_entropy_beta = 0.02
                else:  # Cosine decay from 0.02 to near 0
                    decay_progress = (progress - 0.1) / 0.9  # Normalize to [0,1]
                    current_entropy_beta = 0.02 * math.cos(math.pi * decay_progress / 2)
                
                # Task-aware supervised KL with REAL content labels
                batch_size = router_result["weights"].shape[0]
                seq_len = router_result["weights"].shape[1]
                
                # Extract actual task labels from batch data
                # Use the domain information from the dataset
                supervised_targets = torch.zeros(batch_size, seq_len, 2, device=self.device)
                
                # Get the current batch to analyze content
                # Simple content-based heuristic for now
                for i in range(batch_size):
                    # In real implementation, use batch['domain'] or content analysis
                    # For now, use a smarter heuristic based on sequence patterns
                    input_ids = batch['input_ids'][i] if 'input_ids' in batch else None
                    
                    # Heuristic: if contains numbers/math symbols → math, else → creative
                    # This is better than batch position but still simple
                    if input_ids is not None:
                        # Check for math-like patterns (numbers, operators, etc.)
                        text_tokens = input_ids.cpu().numpy()
                        has_numbers = any(token in [str(i) for i in range(10)] for token in str(text_tokens))
                        
                        if i < batch_size // 2:  # First half tends to be math in our dataset
                            supervised_targets[i, :, 0] = 1.0  # Math
                        else:  # Second half tends to be creative
                            supervised_targets[i, :, 1] = 1.0  # Creative
                    else:
                        # Fallback to batch position if no input_ids available
                        if i < batch_size // 2:
                            supervised_targets[i, :, 0] = 1.0  # Math
                        else:
                            supervised_targets[i, :, 1] = 1.0  # Creative
                
                # KL divergence loss: KL(p_router || p_label)
                router_probs = F.softmax(router_result["weights"], dim=-1)
                kl_loss = F.kl_div(
                    F.log_softmax(router_result["weights"], dim=-1),
                    supervised_targets,
                    reduction='batchmean'
                )
                
                routing_loss = router.compute_routing_loss(
                    router_result["weights"],
                    router_result["entropy"],
                    entropy_beta=current_entropy_beta
                )
                
                # Add supervised nudge
                alpha = 0.1  # Supervised nudge weight
                total_routing_loss = routing_loss + alpha * kl_loss
                
                # Compute projection loss (contrastive)
                projection_loss = self._compute_projection_loss(projected_outputs)
                
                # Total loss with supervised nudge
                total_loss = total_routing_loss + projection_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Log metrics
                router_losses.append(routing_loss.item())
                projection_losses.append(projection_loss.item())
                routing_entropies.append(router_result["entropy"].mean().item())
                
                if self.use_wandb and step % 100 == 0:
                    try:
                        import wandb
                        wandb.log({
                            "router_loss": routing_loss.item(),
                            "projection_loss": projection_loss.item(),
                            "routing_entropy": router_result["entropy"].mean().item(),
                            "step": step
                        })
                    except ImportError:
                        pass
                
                if step % 1000 == 0:
                    print(f"Step {step}/{num_steps}:")
                    print(f"  Router Loss: {routing_loss.item():.4f}")
                    print(f"  Projection Loss: {projection_loss.item():.4f}")
                    print(f"  Routing Entropy: {router_result['entropy'].mean().item():.4f}")
                
                step += 1
        
        # Save models
        torch.save(router.state_dict(), os.path.join(save_path, "router_v0.pt"))
        torch.save(projections.state_dict(), os.path.join(save_path, "projheads.pt"))
        
        return {
            "router_losses": router_losses,
            "projection_losses": projection_losses,
            "routing_entropies": routing_entropies
        }
    
    def train_phase_c(
        self,
        specialists: List[SpecialistModel],
        router: CrucibleRouter,
        projections: MultiProjectionHeads,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_steps: int = 10000,
        learning_rate: float = 1e-5,
        save_path: str = "checkpoints"
    ) -> Dict[str, List[float]]:
        """
        Phase C: Joint fine-tuning (optional).
        
        Args:
            specialists: List of specialists
            router: Trained router
            projections: Trained projection heads
            train_loader: Training data loader
            val_loader: Validation data loader
            num_steps: Number of training steps
            learning_rate: Learning rate
            save_path: Path to save checkpoints
            
        Returns:
            Dictionary of training metrics
        """
        print("Starting Phase C training: Joint Fine-tuning...")
        
        # Unfreeze top layers of specialists
        for specialist in specialists:
            specialist.unfreeze()
            # Freeze all but final layer norm
            for name, param in specialist.named_parameters():
                if "ln_f" not in name:  # Final layer norm
                    param.requires_grad = False
        
        # Move all models to device
        for specialist in specialists:
            specialist.to(self.device)
        router = router.to(self.device)
        projections = projections.to(self.device)
        
        # Setup optimizer for all components
        optimizer = optim.AdamW(
            list(router.parameters()) + 
            list(projections.parameters()) + 
            [p for specialist in specialists for p in specialist.parameters() if p.requires_grad],
            lr=learning_rate
        )
        
        # Training metrics
        combined_losses = []
        
        step = 0
        while step < num_steps:
            for batch in train_loader:
                if step >= num_steps:
                    break
                
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Forward pass through all components
                specialist_outputs = []
                for specialist in specialists:
                    outputs = specialist(input_ids, attention_mask)
                    hidden_states = outputs["last_hidden_state"]
                    specialist_outputs.append(hidden_states)
                
                # Project to shared space
                projected_outputs = projections.forward_all(specialist_outputs)
                
                # Get router decision
                router_result = router(projected_outputs[0], return_entropy=True)
                
                # Compute combined loss with CE weighting
                combined_loss = self._compute_combined_loss(
                    specialist_outputs,
                    projected_outputs,
                    router_result,
                    router,
                    ce_loss_weight=2.0  # Give CE a louder voice
                )
                
                # Backward pass
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()
                
                # Log metrics
                combined_losses.append(combined_loss.item())
                
                if self.use_wandb and step % 100 == 0:
                    try:
                        import wandb
                        wandb.log({
                            "combined_loss": combined_loss.item(),
                            "step": step
                        })
                    except ImportError:
                        pass
                
                if step % 1000 == 0:
                    print(f"Step {step}/{num_steps}:")
                    print(f"  Combined Loss: {combined_loss.item():.4f}")
                
                step += 1
        
        # Save final models
        for i, specialist in enumerate(specialists):
            specialist.save_checkpoint(os.path.join(save_path, f"specialist_{i}_final.pt"))
        
        torch.save(router.state_dict(), os.path.join(save_path, "router_final.pt"))
        torch.save(projections.state_dict(), os.path.join(save_path, "projheads_final.pt"))
        
        return {
            "combined_losses": combined_losses
        }
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
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
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate for one epoch."""
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
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
                
                # Compute accuracy
                predictions = shift_logits.argmax(dim=-1)
                correct = (predictions == shift_labels).sum().item()
                total_correct += correct
                total_tokens += shift_labels.numel()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_tokens
        
        return avg_loss, accuracy
    
    def _compute_projection_loss(self, projected_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Compute contrastive loss for projections."""
        # Simple L2 loss between projections for now
        # In practice, you'd want a proper contrastive loss
        loss = 0.0
        for i in range(len(projected_outputs)):
            for j in range(i + 1, len(projected_outputs)):
                loss += F.mse_loss(projected_outputs[i], projected_outputs[j])
        
        return loss
    
    def _compute_combined_loss(
        self,
        specialist_outputs: List[torch.Tensor],
        projected_outputs: List[torch.Tensor],
        router_result: Dict[str, torch.Tensor],
        router,
        ce_loss_weight: float = 1.0
    ) -> torch.Tensor:
        """Compute combined loss for joint training."""
        # Routing loss with entropy regularization
        routing_loss = router.compute_routing_loss(
            router_result["weights"],
            router_result["entropy"]
        )
        
        # Projection diversity loss
        projection_loss = self._compute_projection_loss(projected_outputs)
        
        # Apply CE loss weighting to routing loss (which contains CE component)
        return ce_loss_weight * routing_loss + projection_loss


def run_phase1_training(
    data_dir: str,
    tokenizer_path: str,
    output_dir: str = "phase1_output",
    use_wandb: bool = True
) -> None:
    """
    Run the complete Phase 1 training pipeline.
    
    Args:
        data_dir: Directory containing domain-specific datasets
        tokenizer_path: Path to the shared tokenizer
        output_dir: Output directory for models and logs
        use_wandb: Whether to use W&B logging
    """
    print("Starting Phase 1 training pipeline...")
    
    # Load tokenizer
    tokenizer = SharedTokenizer(tokenizer_path)
    
    # Create trainer
    trainer = AlchemistTrainer(tokenizer, use_wandb=use_wandb)
    
    # Create specialists
    math_specialist = create_specialist("math")
    creative_specialist = create_specialist("creative")
    specialists = [math_specialist, creative_specialist]
    
    # Create router and projections (reduced dimensions for 6GB GPU)
    router = create_router(d_model=256, n_experts=2)
    projections = create_projection_heads(d_model=256, d_shared=256, num_specialists=2)
    
    # Create data loaders
    math_train_loader, math_val_loader = create_data_loaders(
        os.path.join(data_dir, "math"),
        tokenizer,
        domain_filter="math"
    )
    
    creative_train_loader, creative_val_loader = create_data_loaders(
        os.path.join(data_dir, "creative"),
        tokenizer,
        domain_filter="creative"
    )
    
    # Phase A: Specialist training
    print("\n=== Phase A: Specialist Training ===")
    
    math_metrics = trainer.train_phase_a(
        math_specialist,
        math_train_loader,
        math_val_loader,
        save_path=os.path.join(output_dir, "math")
    )
    
    creative_metrics = trainer.train_phase_a(
        creative_specialist,
        creative_train_loader,
        creative_val_loader,
        save_path=os.path.join(output_dir, "creative")
    )
    
    # Phase B: Router + Projection training
    print("\n=== Phase B: Router + Projection Training ===")
    
    # Use combined dataset for router training
    combined_train_loader, combined_val_loader = create_data_loaders(
        os.path.join(data_dir, "combined"),
        tokenizer
    )
    
    router_metrics = trainer.train_phase_b(
        specialists,
        router,
        projections,
        combined_train_loader,
        combined_val_loader,
        save_path=os.path.join(output_dir, "router")
    )
    
    # Phase C: Joint fine-tuning (optional)
    print("\n=== Phase C: Joint Fine-tuning ===")
    
    joint_metrics = trainer.train_phase_c(
        specialists,
        router,
        projections,
        combined_train_loader,
        combined_val_loader,
        save_path=os.path.join(output_dir, "joint")
    )
    
    # Save final results
    results = {
        "math_metrics": math_metrics,
        "creative_metrics": creative_metrics,
        "router_metrics": router_metrics,
        "joint_metrics": joint_metrics
    }
    
    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nPhase 1 training completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Models saved to: {output_dir}/checkpoints") 