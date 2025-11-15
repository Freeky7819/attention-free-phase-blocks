"""
API Request/Response Schemas for Resonant Model Inference

Pydantic models for type validation, serialization, and API documentation.
All schemas include comprehensive field validation and documentation for
the world's first resonant model inference API.

Physical Interpretation:
    - temperature: Softmax temperature (higher = more random)
    - top_k/top_p: Nucleus sampling parameters
    - phase_coherence: Resonance alignment quality metric
    - memory_stats: PVM/PLM diagnostics

Usage:
    from api.schemas import GenerateRequest, GenerateResponse

    request = GenerateRequest(
        prompt="Once upon a time",
        max_tokens=100,
        temperature=0.8
    )
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class GenerateRequest(BaseModel):
    """
    Text generation request for resonant model inference.

    Fields:
        prompt: Input text to continue/complete
        max_tokens: Maximum number of tokens to generate (1-2048)
        temperature: Sampling temperature (0.0-2.0)
            - 0.0: Greedy (deterministic)
            - 0.7: Balanced creativity
            - 1.0: Standard sampling
            - 1.5+: High randomness
        top_k: Keep only top K most likely tokens (0 = disabled)
        top_p: Nucleus sampling cumulative probability (0.0-1.0)
        repetition_penalty: Penalty for repeating tokens (1.0 = none)
        stop_sequences: Stop generation when these strings appear
        seed: Random seed for reproducibility (optional)
        return_phase_metrics: Include phase coherence diagnostics
        return_memory_stats: Include PVM/PLM memory metrics

    Example:
        {
            "prompt": "The theory of resonant neural networks",
            "max_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_phase_metrics": true
        }
    """
    prompt: str = Field(
        ...,
        description="Input text prompt for generation",
        min_length=1,
        max_length=8192,
        example="Once upon a time in a neural network"
    )
    max_tokens: int = Field(
        default=100,
        ge=1,
        le=2048,
        description="Maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (higher = more random)"
    )
    top_k: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Top-K sampling (0 = disabled)"
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability threshold"
    )
    repetition_penalty: float = Field(
        default=1.0,
        ge=1.0,
        le=2.0,
        description="Penalty for token repetition"
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None,
        max_items=10,
        description="Stop generation at these sequences"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    return_phase_metrics: bool = Field(
        default=False,
        description="Include phase coherence and resonance metrics"
    )
    return_memory_stats: bool = Field(
        default=False,
        description="Include PVM/PLM memory diagnostics"
    )

    @validator("prompt")
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Resonant neural networks use phase dynamics to",
                "max_tokens": 100,
                "temperature": 0.8,
                "top_p": 0.9,
                "return_phase_metrics": True
            }
        }


class PhaseMetrics(BaseModel):
    """
    Phase coherence and resonance quality metrics.

    These metrics characterize the phase dynamics during generation:
    - phase_coherence: Average alignment across AFRB layers (0-1)
    - gamma_mean: Mean resonance depth (0-1)
    - omega_values: Resonant frequencies per layer
    - phase_offsets: Phase shifts per layer (radians)

    Physical Interpretation:
        - High coherence (>0.7): Strong phase binding
        - Low coherence (<0.3): Weak resonance
        - Gamma ~0.2: Balanced phase-data coupling
    """
    phase_coherence: float = Field(
        description="Average phase alignment across layers (0-1)"
    )
    gamma_mean: float = Field(
        description="Mean resonance depth (0-1)"
    )
    omega_values: Optional[List[float]] = Field(
        default=None,
        description="Resonant frequencies per AFRB layer"
    )
    phase_offsets: Optional[List[float]] = Field(
        default=None,
        description="Phase shifts per layer (radians)"
    )


class MemoryStats(BaseModel):
    """
    PVM/PLM memory diagnostics.

    Tracks memory state evolution during generation:
    - pvm_mem_norm: Average PVM memory magnitude
    - pvm_gate_strength: PVM gating strength (0-1)
    - plm_coherence: PLM lattice phase coherence
    - total_memory_kb: Memory footprint in kilobytes

    Usage:
        Monitor these metrics to debug memory behavior:
        - High mem_norm: Strong pattern accumulation
        - Low gate_strength: Memory weakly coupled
        - High plm_coherence: Stable lattice dynamics
    """
    pvm_mem_norm: Optional[float] = Field(
        default=None,
        description="Average PVM memory magnitude"
    )
    pvm_gate_strength: Optional[float] = Field(
        default=None,
        description="PVM gating strength (0-1)"
    )
    plm_coherence: Optional[float] = Field(
        default=None,
        description="PLM lattice phase coherence"
    )
    total_memory_kb: Optional[float] = Field(
        default=None,
        description="Total memory footprint in kilobytes"
    )


class GenerateResponse(BaseModel):
    """
    Text generation response with optional diagnostics.

    Fields:
        text: Generated text (continuation of prompt)
        prompt: Original input prompt (echo)
        tokens_generated: Number of tokens produced
        finish_reason: Why generation stopped
        timing_ms: Generation latency in milliseconds
        phase_metrics: Phase coherence diagnostics (optional)
        memory_stats: PVM/PLM metrics (optional)
        model_info: Model metadata

    Example:
        {
            "text": "use phase dynamics to replace attention...",
            "prompt": "Resonant neural networks",
            "tokens_generated": 87,
            "finish_reason": "max_tokens",
            "timing_ms": 342.5,
            "phase_metrics": {"phase_coherence": 0.76, ...}
        }
    """
    text: str = Field(
        description="Generated text output"
    )
    prompt: str = Field(
        description="Original input prompt (echo)"
    )
    tokens_generated: int = Field(
        description="Number of tokens generated"
    )
    finish_reason: str = Field(
        description="Reason for stopping (max_tokens, eos_token, stop_sequence)"
    )
    timing_ms: float = Field(
        description="Generation latency in milliseconds"
    )
    phase_metrics: Optional[PhaseMetrics] = Field(
        default=None,
        description="Phase coherence and resonance diagnostics"
    )
    memory_stats: Optional[MemoryStats] = Field(
        default=None,
        description="PVM/PLM memory state information"
    )
    model_info: Dict[str, Any] = Field(
        description="Model metadata (name, num_afrb_layers, etc.)"
    )


class RetrievalRequest(BaseModel):
    """
    Needle-in-haystack retrieval request.

    Tests the model's ability to retrieve specific patterns from long contexts
    using phase-based content addressing.

    Fields:
        context: Long text containing the needle
        needle: Pattern to retrieve (substring of context)
        query: Query text for retrieval (if different from needle)
        retrieval_mode: Method for retrieval
            - "pvm": Use Phase-Vector Memory content addressing
            - "plm": Use Phase-Lattice Memory spatial search
            - "hybrid": Combine PVM and PLM
        top_k: Return top K most similar positions
        return_positions: Include needle position indices
        return_similarity_scores: Include cosine similarities

    Example:
        {
            "context": "... [10000 tokens] ... The secret code is X7Z9 ...",
            "needle": "secret code is X7Z9",
            "retrieval_mode": "pvm",
            "top_k": 5
        }
    """
    context: str = Field(
        ...,
        description="Long text context containing the needle",
        min_length=1,
        max_length=100000
    )
    needle: str = Field(
        ...,
        description="Pattern to retrieve from context",
        min_length=1,
        max_length=1000
    )
    query: Optional[str] = Field(
        default=None,
        description="Alternative query text (defaults to needle)"
    )
    retrieval_mode: str = Field(
        default="pvm",
        description="Retrieval method: 'pvm', 'plm', or 'hybrid'"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of top matches to return"
    )
    return_positions: bool = Field(
        default=True,
        description="Include position indices in response"
    )
    return_similarity_scores: bool = Field(
        default=True,
        description="Include cosine similarity scores"
    )

    @validator("retrieval_mode")
    def validate_retrieval_mode(cls, v):
        allowed = ["pvm", "plm", "hybrid"]
        if v not in allowed:
            raise ValueError(f"retrieval_mode must be one of {allowed}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "context": "Random text... The key insight is resonance... More text...",
                "needle": "key insight is resonance",
                "retrieval_mode": "pvm",
                "top_k": 3
            }
        }


class RetrievalMatch(BaseModel):
    """
    Single retrieval match result.

    Represents one retrieved location matching the query:
    - text: Retrieved text snippet
    - position: Token position in context
    - similarity: Cosine similarity score (0-1)
    - confidence: Retrieval confidence (0-1)
    """
    text: str = Field(
        description="Retrieved text snippet"
    )
    position: Optional[int] = Field(
        default=None,
        description="Token position in original context"
    )
    similarity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Retrieval confidence"
    )


class RetrievalResponse(BaseModel):
    """
    Needle retrieval response with ranked matches.

    Fields:
        matches: List of retrieved text snippets (ranked by similarity)
        needle: Original needle pattern (echo)
        retrieval_mode: Method used for retrieval
        timing_ms: Retrieval latency in milliseconds
        phase_metrics: Phase coherence during retrieval (optional)
        memory_stats: PVM/PLM state diagnostics (optional)

    Example:
        {
            "matches": [
                {"text": "key insight is resonance", "similarity": 0.94},
                {"text": "insight in resonance theory", "similarity": 0.78}
            ],
            "needle": "key insight is resonance",
            "timing_ms": 45.2
        }
    """
    matches: List[RetrievalMatch] = Field(
        description="Retrieved matches ranked by similarity"
    )
    needle: str = Field(
        description="Original needle pattern (echo)"
    )
    retrieval_mode: str = Field(
        description="Retrieval method used"
    )
    timing_ms: float = Field(
        description="Retrieval latency in milliseconds"
    )
    phase_metrics: Optional[PhaseMetrics] = Field(
        default=None,
        description="Phase coherence during retrieval"
    )
    memory_stats: Optional[MemoryStats] = Field(
        default=None,
        description="Memory state diagnostics"
    )


class HealthResponse(BaseModel):
    """
    Health check response.

    Reports API server status and readiness:
    - status: "healthy" or "degraded"
    - model_loaded: Whether model is ready
    - uptime_seconds: Time since server start
    - version: API version string
    - timestamp: Current server time
    """
    status: str = Field(
        description="Server status: 'healthy' or 'degraded'"
    )
    model_loaded: bool = Field(
        description="Whether inference model is loaded and ready"
    )
    uptime_seconds: float = Field(
        description="Server uptime in seconds"
    )
    version: str = Field(
        description="API version string"
    )
    timestamp: str = Field(
        description="Current server timestamp (ISO 8601)"
    )


class ModelInfoResponse(BaseModel):
    """
    Model metadata and configuration.

    Returns detailed information about the loaded resonant model:
    - model_name: Base model identifier (e.g., "pythia-160m")
    - num_afrb_layers: Number of AFRB adapter layers
    - hidden_size: Model dimension
    - vocab_size: Tokenizer vocabulary size
    - max_context_length: Maximum sequence length
    - phase_features: Phase dynamics configuration
    - memory_features: PVM/PLM configuration

    Example:
        {
            "model_name": "pythia-160m-resonant",
            "num_afrb_layers": 8,
            "hidden_size": 768,
            "phase_features": {
                "adaptive_omega": true,
                "learnable_gamma": true
            }
        }
    """
    model_name: str = Field(
        description="Base model identifier"
    )
    num_afrb_layers: int = Field(
        description="Number of AFRB resonant layers"
    )
    hidden_size: int = Field(
        description="Model hidden dimension"
    )
    vocab_size: int = Field(
        description="Tokenizer vocabulary size"
    )
    max_context_length: int = Field(
        description="Maximum sequence length supported"
    )
    phase_features: Dict[str, Any] = Field(
        description="Phase dynamics configuration"
    )
    memory_features: Dict[str, Any] = Field(
        description="PVM/PLM memory configuration"
    )
    device: str = Field(
        description="Computation device (cuda, cpu, mps)"
    )


class ErrorResponse(BaseModel):
    """
    Error response for failed requests.

    Standard error format for all API failures:
    - error: Error type/category
    - message: Human-readable error description
    - details: Additional context (optional)
    - timestamp: When error occurred

    Error Types:
        - validation_error: Invalid request parameters
        - model_error: Model inference failure
        - resource_error: Out of memory or compute
        - internal_error: Unexpected server error
    """
    error: str = Field(
        description="Error type/category"
    )
    message: str = Field(
        description="Human-readable error description"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error context"
    )
    timestamp: str = Field(
        description="Error timestamp (ISO 8601)"
    )

    class Config:
        schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "max_tokens must be between 1 and 2048",
                "timestamp": "2025-11-14T12:34:56.789Z"
            }
        }
