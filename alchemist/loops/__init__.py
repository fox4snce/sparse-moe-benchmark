"""
Phase 3: Loop Control Implementation

This module implements internal iterative reasoning capabilities:
- Continue-thinking probability heads on specialists
- Loop control system with max-cycle gate
- Convergence monitoring and early stopping
"""

from .thinking_heads import ThinkingHead
from .loop_controller import LoopController

__all__ = ["ThinkingHead", "LoopController"] 