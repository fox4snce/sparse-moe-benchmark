"""
Memory Integration for Alchemist Phase 2.

Combines Memory Specialist, Knowledge Graph, and 3-expert routing system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np

from ..foundation.tokenizer import SharedTokenizer
from ..specialists.base_specialist import create_specialist
from ..routing.projection_heads import create_projection_heads
from ..routing.crucible_router import create_router
from .memory_specialist import create_memory_specialist, MemoryConfig
from .simple_knowledge_graph import SimpleKnowledgeGraph, Triple


class AlchemistMemorySystem(nn.Module):
    """
    Complete Alchemist system with Memory integration.
    
    Combines:
    - Math Specialist (from Phase 1)
    - Creative Specialist (from Phase 1) 
    - Memory Specialist (new)
    - 3-expert Router
    - Knowledge Graph
    """
    
    def __init__(
        self,
        tokenizer: SharedTokenizer,
        math_specialist_path: Optional[str] = None,
        creative_specialist_path: Optional[str] = None,
        knowledge_graph_path: str = "memory_kg.pkl",
        device: str = "cuda"
    ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.device = device
        
        # Load specialists
        print("🧠 Initializing Alchemist Memory System...")
        
        # Math and Creative specialists (from Phase 1)
        self.math_specialist = create_specialist('math').to(device)
        self.creative_specialist = create_specialist('creative').to(device)
        
        if math_specialist_path:
            checkpoint = torch.load(math_specialist_path, weights_only=False)
            self.math_specialist.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded Math specialist from Phase 1")
        
        if creative_specialist_path:
            checkpoint = torch.load(creative_specialist_path, weights_only=False)
            self.creative_specialist.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded Creative specialist from Phase 1")
        
        # Memory specialist (new)
        memory_config = MemoryConfig(
            vocab_size=tokenizer.sp.get_piece_size(),
            hidden_size=256,  # Match other specialists
            memory_dim=256
        )
        self.memory_specialist = create_memory_specialist(memory_config).to(device)
        print("✅ Created Memory specialist")
        
        # 3-expert router and projections
        self.router = create_router(d_model=256, n_experts=3).to(device)
        self.projections = create_projection_heads(d_model=256, d_shared=256, num_specialists=3).to(device)
        print("✅ Created 3-expert router and projections")
        
        # Knowledge graph
        self.knowledge_graph = SimpleKnowledgeGraph(knowledge_graph_path, vector_dim=256)
        print("✅ Initialized knowledge graph")
        
        # Expert mapping
        self.expert_names = ["math", "creative", "memory"]
        
        print(f"🎉 Alchemist Memory System ready!")
        print(f"   Experts: {', '.join(self.expert_names)}")
        print(f"   Knowledge base: {self.knowledge_graph.count_triples()} triples")
    
    def forward(
        self,
        input_text: str,
        user: str = "user",
        return_routing_info: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass through the complete Alchemist system.
        
        Args:
            input_text: Input query/prompt
            user: Username for personalization
            return_routing_info: Whether to return routing details
            
        Returns:
            Dict with response, routing info, and memory updates
        """
        # Tokenize input
        tokens = self.tokenizer.encode(input_text)
        input_ids = torch.tensor([tokens], device=self.device)
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            # Get specialist outputs
            math_output = self.math_specialist(input_ids, attention_mask)
            creative_output = self.creative_specialist(input_ids, attention_mask)
            
            # Create memory context
            memory_context = {"user": user, "text": input_text}
            memory_output = self.memory_specialist.encode_json_input(memory_context, self.tokenizer)
            
            # Get hidden states for projection
            if math_output['last_hidden_state'] is not None:
                math_hidden = math_output['last_hidden_state'][:, -1, :]
                creative_hidden = creative_output['last_hidden_state'][:, -1, :]
                memory_hidden = memory_output  # Already [1, 256]
                
                # Project to shared space
                math_proj = self.projections.forward(math_hidden, 0)
                creative_proj = self.projections.forward(creative_hidden, 1)
                memory_proj = self.projections.forward(memory_hidden, 2)
                
                # Router decision (use math projection as query for now)
                router_result = self.router(math_proj, return_entropy=True)
                routing_weights = router_result["weights"].softmax(dim=-1)
                routing_entropy = router_result["entropy"]
                
                # Get routing decision
                expert_weights = routing_weights[0, 0].cpu().numpy()  # [3]
                top_expert_idx = expert_weights.argmax()
                top_expert_name = self.expert_names[top_expert_idx]
                
                # Generate response based on routing
                if top_expert_idx == 0:  # Math
                    response_logits = math_output["logits"]
                elif top_expert_idx == 1:  # Creative
                    response_logits = creative_output["logits"]
                else:  # Memory
                    # For memory, retrieve from knowledge graph
                    persona_vector = self.knowledge_graph.create_persona_vector(
                        user, self.memory_specialist, self.tokenizer
                    )
                    
                    # Search for relevant memories
                    if np.linalg.norm(persona_vector) > 0:
                        similar_triples = self.knowledge_graph.search_by_vector(persona_vector, k=3)
                        if similar_triples:
                            memory_facts = [triple.to_text() for triple, _ in similar_triples]
                            memory_response = f"Based on what I know: {', '.join(memory_facts[:2])}"
                        else:
                            memory_response = "I don't have specific information about that."
                    else:
                        memory_response = "I'm learning about your preferences."
                    
                    # For now, return text response for memory
                    return {
                        "response": memory_response,
                        "expert_used": top_expert_name,
                        "routing_weights": expert_weights,
                        "routing_entropy": routing_entropy.item(),
                        "memory_facts": similar_triples if 'similar_triples' in locals() else []
                    }
                
                # Generate text from logits (simplified)
                response_text = self._generate_from_logits(response_logits, input_text)
                
                result = {
                    "response": response_text,
                    "expert_used": top_expert_name,
                    "routing_weights": expert_weights,
                    "routing_entropy": routing_entropy.item()
                }
                
                if return_routing_info:
                    result.update({
                        "projections": {
                            "math": math_proj.cpu().numpy(),
                            "creative": creative_proj.cpu().numpy(), 
                            "memory": memory_proj.cpu().numpy()
                        },
                        "routing_details": router_result
                    })
                
                return result
            
            else:
                return {"error": "No hidden states available", "response": "Error processing input"}
    
    def _generate_from_logits(self, logits: torch.Tensor, input_text: str) -> str:
        """Simple text generation from logits (placeholder)."""
        # For now, just return a simple response based on the expert
        # In a full implementation, you'd use proper text generation
        return f"Generated response based on: {input_text}"
    
    def remember_fact(self, user: str, fact_text: str) -> bool:
        """
        Store a new fact in the knowledge graph.
        
        Args:
            user: Username
            fact_text: Fact to remember (e.g., "I play ukulele")
            
        Returns:
            success: Whether fact was stored successfully
        """
        try:
            # Simple parsing for now
            # In practice, you'd use NLP to extract subject-predicate-object
            if "like" in fact_text.lower():
                # Extract what they like
                parts = fact_text.lower().split("like")
                if len(parts) >= 2:
                    obj = parts[1].strip().strip(".")
                    triple = Triple(user, "likes", obj)
                    
                    # Create vector representation
                    context = {"user": user, "text": fact_text}
                    vector = self.memory_specialist.encode_json_input(context, self.tokenizer)
                    vector_np = vector.cpu().numpy().flatten()
                    
                    # Store in knowledge graph
                    self.knowledge_graph.add_triple(triple, vector_np)
                    print(f"💾 Remembered: {user} likes {obj}")
                    return True
            
            # Generic storage
            triple = Triple(user, "mentioned", fact_text)
            context = {"user": user, "text": fact_text}
            vector = self.memory_specialist.encode_json_input(context, self.tokenizer)
            vector_np = vector.cpu().numpy().flatten()
            
            self.knowledge_graph.add_triple(triple, vector_np)
            print(f"💾 Remembered: {user} mentioned {fact_text}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to remember fact: {e}")
            return False
    
    def get_user_facts(self, user: str) -> List[str]:
        """Get all known facts about a user."""
        triples = self.knowledge_graph.search_by_subject(user)
        return [triple.to_text() for triple in triples]
    
    def save_knowledge_graph(self):
        """Save knowledge graph to disk."""
        self.knowledge_graph.save_to_disk()
    
    def freeze_phase1_components(self):
        """Freeze math and creative specialists for Phase 2 training."""
        self.math_specialist.freeze_parameters()
        self.creative_specialist.freeze_parameters()
        print("🧊 Froze Phase 1 specialists (math, creative)")
    
    def unfreeze_all_components(self):
        """Unfreeze all components for joint training."""
        self.math_specialist.unfreeze_parameters()
        self.creative_specialist.unfreeze_parameters()
        self.memory_specialist.unfreeze_parameters()
        print("🔥 Unfroze all specialists for joint training")


def create_memory_system(
    tokenizer_path: str,
    math_specialist_path: Optional[str] = None,
    creative_specialist_path: Optional[str] = None,
    knowledge_graph_path: str = "memory_kg.pkl",
    device: str = "cuda"
) -> AlchemistMemorySystem:
    """Factory function to create complete memory system."""
    
    tokenizer = SharedTokenizer(tokenizer_path)
    
    return AlchemistMemorySystem(
        tokenizer=tokenizer,
        math_specialist_path=math_specialist_path,
        creative_specialist_path=creative_specialist_path,
        knowledge_graph_path=knowledge_graph_path,
        device=device
    )


if __name__ == "__main__":
    # Test the memory system
    print("🧪 Testing Alchemist Memory System...")
    
    # Create system (without Phase 1 checkpoints for now)
    system = create_memory_system(
        tokenizer_path="phase1_final/spm.model",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Test basic functionality
    result = system.forward("What is 2 + 2?", user="TestUser")
    print(f"Math query result: {result}")
    
    # Test memory
    system.remember_fact("TestUser", "I like pizza")
    facts = system.get_user_facts("TestUser")
    print(f"User facts: {facts}")
    
    memory_result = system.forward("What do I like?", user="TestUser")
    print(f"Memory query result: {memory_result}")
    
    print("✅ Memory system test complete!")