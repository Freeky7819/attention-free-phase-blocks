"""
Resonant Models: Phase-Based Neural Architecture

Core components for training models that replace attention with resonance.

This package implements a novel neural architecture that replaces traditional
attention mechanisms with phase-based resonance. Instead of computing query-key
attention scores, the model stores hidden states in Phase Vector Memory (PVM)
and retrieves relevant information through phase-coherent readouts.

Core Components:
---------------
- AFRB (Adapter with Frequency Response and Buffering): Main adapter module
- PhaseEmbedding: Sinusoidal phase embeddings for positional encoding
- PhaseVectorMemory: Memory bank for storing phase-space representations
- PhaseBuffer: Sliding window buffer for recent phase states

Alignment Utilities:
-------------------
- ridge_fit: Ridge regression for PVM-to-logits calibration
- compute_query_key: Extract query vectors from hidden states
- collect_alignment_pairs: Collect (PVM, target) pairs for calibration

Diagnostic Tools:
----------------
- compute_phase_metrics: Curvature-based trajectory analysis

Mathematical Foundation:
-----------------------
The resonant model operates on the principle that semantically similar tokens
create coherent phase patterns in hidden state space. By storing these patterns
in PVM and using phase-weighted retrieval, the model can recall information
without explicit attention computation.

Key equation: Z @ W ≈ E
Where:
- Z: PVM readouts (phase-space representations)
- W: Ridge-calibrated transformation matrix
- E: Target embeddings (what we want to predict)

This allows PVM readouts to produce logits directly, bypassing attention.
"""

from .afrb import AFRB, PhaseEmbedding
from .phase_memory import PhaseVectorMemory, PhaseBuffer
from .utils_alignment import ridge_fit, compute_query_key, collect_alignment_pairs
from .phase_curvature import compute_phase_curvature_metrics, compute_phase_metrics

__version__ = "0.1.0"
__all__ = [
    "AFRB",
    "PhaseEmbedding",
    "PhaseVectorMemory",
    "PhaseBuffer",
    "ridge_fit",
    "compute_query_key",
    "collect_alignment_pairs",
    "compute_phase_curvature_metrics",
    "compute_phase_metrics",
]
