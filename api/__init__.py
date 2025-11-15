"""
Resonant Model API Module

Production REST API for the world's first resonant neural model repository.

This package provides:
- FastAPI server with async endpoints
- Model inference engine with AFRB + PVM + PLM
- Pydantic schemas for request/response validation
- Comprehensive error handling and logging

Components:
    - server: FastAPI application with endpoints
    - inference: Model loading and inference logic
    - schemas: Pydantic models for API contracts

Usage:
    # Start API server
    from api.server import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # Use inference engine directly
    from api.inference import ResonantModelInference
    model = ResonantModelInference(checkpoint_path="model.pt")
    output = model.generate("Once upon a time", max_tokens=100)

Environment Variables:
    MODEL_CHECKPOINT: Path to trained model checkpoint
    MODEL_NAME: Base model identifier (default: EleutherAI/pythia-160m)
    DEVICE: Computation device (cuda, cpu, mps)
    API_HOST: Server host (default: 0.0.0.0)
    API_PORT: Server port (default: 8000)
    CORS_ORIGINS: Allowed CORS origins (comma-separated)

Quick Start:
    export MODEL_CHECKPOINT=checkpoints/model_best.pt
    export DEVICE=cuda
    uvicorn api.server:app --host 0.0.0.0 --port 8000

For detailed documentation, see api/README.md
"""

from api.inference import ResonantModelInference, GenerationOutput, RetrievalOutput
from api.schemas import (
    GenerateRequest,
    GenerateResponse,
    RetrievalRequest,
    RetrievalResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
    PhaseMetrics,
    MemoryStats,
)

__version__ = "1.0.0"
__author__ = "Damjan Žakelj"

__all__ = [
    # Inference
    "ResonantModelInference",
    "GenerationOutput",
    "RetrievalOutput",
    # Schemas
    "GenerateRequest",
    "GenerateResponse",
    "RetrievalRequest",
    "RetrievalResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "ErrorResponse",
    "PhaseMetrics",
    "MemoryStats",
]
