"""
FastAPI Server for Resonant Model Inference

Production-ready REST API for the world's first resonant neural model repository.
Provides endpoints for text generation, content retrieval, and model introspection.

Architecture:
    - Async FastAPI server with CORS support
    - Global model instance (singleton pattern)
    - Comprehensive error handling
    - Request validation via Pydantic schemas
    - Structured logging

Endpoints:
    - POST /v1/generate - Text generation with phase-aware sampling
    - POST /v1/retrieve - Needle-in-haystack content retrieval
    - GET /v1/health - Health check and readiness probe
    - GET /v1/model/info - Model metadata and configuration

Security Features:
    - CORS middleware (configurable origins)
    - Request validation (Pydantic)
    - Rate limiting notes (implement via reverse proxy)
    - Input sanitization
    - Error message sanitization (no internal details leaked)

Usage:
    # Development server
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

    # Production server (with Gunicorn)
    gunicorn api.server:app -w 4 -k uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:8000 --timeout 120

Environment Variables:
    MODEL_CHECKPOINT: Path to trained model checkpoint (required)
    MODEL_NAME: Base model identifier (default: EleutherAI/pythia-160m)
    DEVICE: Computation device (default: cuda)
    API_HOST: Server host (default: 0.0.0.0)
    API_PORT: Server port (default: 8000)
    CORS_ORIGINS: Allowed CORS origins (comma-separated)

Example:
    export MODEL_CHECKPOINT=checkpoints/model_best.pt
    export DEVICE=cuda
    uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.schemas import (
    GenerateRequest,
    GenerateResponse,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalMatch,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
    PhaseMetrics,
    MemoryStats
)
from api.inference import ResonantModelInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance (loaded on startup)
model: Optional[ResonantModelInference] = None
startup_time: Optional[float] = None

# API version
API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown logic.

    Startup:
        - Load environment variables
        - Initialize model from checkpoint
        - Warm up inference engine

    Shutdown:
        - Clean up GPU memory
        - Log shutdown message
    """
    global model, startup_time

    logger.info("=" * 60)
    logger.info("Resonant Model API Server - Startup")
    logger.info("=" * 60)

    # Load environment variables
    checkpoint_path = os.getenv('MODEL_CHECKPOINT', None)
    model_name = os.getenv('MODEL_NAME', 'EleutherAI/pythia-160m')
    device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

    if checkpoint_path is None:
        logger.warning("MODEL_CHECKPOINT not set - running without trained adapters")
        logger.warning("Set MODEL_CHECKPOINT environment variable to load trained model")

    # Initialize model
    try:
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Device: {device}")

        model = ResonantModelInference(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            device=device,
            dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
            max_batch_size=8
        )

        startup_time = time.time()
        logger.info("Model loaded successfully!")
        logger.info(f"Ready to serve requests on {device}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise RuntimeError(f"Model initialization failed: {e}")

    logger.info("=" * 60)

    yield  # Server is running

    # Shutdown
    logger.info("Shutting down server...")
    if model is not None:
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Resonant Model API",
    description=(
        "Production REST API for resonant neural models - "
        "the world's first phase-based alternative to attention mechanisms. "
        "Built on AFRB (Adaptive Frequency-Resonant Blocks) + "
        "PVM (Phase-Vector Memory) + PLM (Phase-Lattice Memory)."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware configuration
cors_origins = os.getenv('CORS_ORIGINS', '*').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="http_error",
            message=exc.detail,
            timestamp=datetime.utcnow().isoformat() + "Z"
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with sanitized error response."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_error",
            message="An unexpected error occurred. Please try again later.",
            timestamp=datetime.utcnow().isoformat() + "Z"
        ).dict()
    )


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing."""
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000

    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration_ms:.2f}ms"
    )

    return response


# API Endpoints

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs."""
    return {
        "message": "Resonant Model API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/v1/health"
    }


@app.get(
    "/v1/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check API server health and model readiness"
)
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.

    Returns:
        HealthResponse with server status, model readiness, and uptime

    Status Codes:
        200: Server healthy and model loaded
        503: Server degraded (model not loaded)
    """
    model_loaded = model is not None
    uptime = time.time() - startup_time if startup_time else 0.0

    response = HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        uptime_seconds=uptime,
        version=API_VERSION,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )

    status_code = status.HTTP_200_OK if model_loaded else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=status_code,
        content=response.dict()
    )


@app.get(
    "/v1/model/info",
    response_model=ModelInfoResponse,
    summary="Model Information",
    description="Get model metadata and configuration details"
)
async def model_info():
    """
    Retrieve model metadata and configuration.

    Returns detailed information about the loaded resonant model:
    - Base model architecture
    - Number of AFRB layers
    - Phase dynamics configuration
    - Memory configuration (PVM/PLM)
    - Device and precision

    Returns:
        ModelInfoResponse with comprehensive model metadata

    Raises:
        503: Model not loaded

    Example Response:
        {
            "model_name": "EleutherAI/pythia-160m",
            "num_afrb_layers": 8,
            "hidden_size": 768,
            "phase_features": {"adaptive_omega": true, ...}
        }
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    info = model.get_model_info()

    return ModelInfoResponse(**info)


@app.post(
    "/v1/generate",
    response_model=GenerateResponse,
    summary="Text Generation",
    description="Generate text continuation with phase-aware sampling"
)
async def generate_text(request: GenerateRequest):
    """
    Generate text continuation for given prompt.

    Uses resonant model with phase dynamics for creative text generation.
    Supports temperature sampling, nucleus sampling (top-p), top-k filtering,
    and repetition penalty.

    Args:
        request: GenerateRequest with prompt and sampling parameters

    Returns:
        GenerateResponse with generated text, timing, and optional metrics

    Raises:
        503: Model not loaded
        400: Invalid request parameters
        500: Generation failed

    Example Request:
        {
            "prompt": "Once upon a time",
            "max_tokens": 100,
            "temperature": 0.8,
            "top_p": 0.9,
            "return_phase_metrics": true
        }

    Example Response:
        {
            "text": " in a neural network, there lived...",
            "tokens_generated": 87,
            "finish_reason": "max_tokens",
            "timing_ms": 342.5
        }
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # Run inference
        output = model.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            stop_sequences=request.stop_sequences,
            seed=request.seed,
            return_phase_metrics=request.return_phase_metrics,
            return_memory_stats=request.return_memory_stats
        )

        # Build response
        response = GenerateResponse(
            text=output.text,
            prompt=request.prompt,
            tokens_generated=output.tokens_generated,
            finish_reason=output.finish_reason,
            timing_ms=output.timing_ms,
            model_info=model.get_model_info()
        )

        # Add optional metrics
        if output.phase_metrics:
            response.phase_metrics = PhaseMetrics(**output.phase_metrics)

        if output.memory_stats:
            response.memory_stats = MemoryStats(**output.memory_stats)

        return response

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


@app.post(
    "/v1/retrieve",
    response_model=RetrievalResponse,
    summary="Content Retrieval",
    description="Needle-in-haystack retrieval via phase-based content addressing"
)
async def retrieve_content(request: RetrievalRequest):
    """
    Content-addressable retrieval from long contexts.

    Uses phase-based similarity (PVM or PLM) to find patterns matching
    the needle query. This tests the model's long-range memory and
    content-addressing capabilities.

    Args:
        request: RetrievalRequest with context, needle, and retrieval config

    Returns:
        RetrievalResponse with ranked matches, similarities, and timing

    Raises:
        503: Model not loaded
        400: Invalid request parameters
        500: Retrieval failed

    Example Request:
        {
            "context": "...long text... The secret code is X7Z9 ...",
            "needle": "secret code",
            "retrieval_mode": "pvm",
            "top_k": 5
        }

    Example Response:
        {
            "matches": [
                {"text": "secret code is X7Z9", "similarity": 0.94},
                {"text": "code for security", "similarity": 0.67}
            ],
            "timing_ms": 45.2
        }
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # Run retrieval
        output = model.retrieve(
            context=request.context,
            needle=request.needle,
            query=request.query,
            mode=request.retrieval_mode,
            top_k=request.top_k
        )

        # Build response
        matches = [
            RetrievalMatch(
                text=text,
                position=position if request.return_positions else None,
                similarity=similarity if request.return_similarity_scores else None,
                confidence=confidence
            )
            for text, position, similarity, confidence in output.matches
        ]

        response = RetrievalResponse(
            matches=matches,
            needle=request.needle,
            retrieval_mode=request.retrieval_mode,
            timing_ms=output.timing_ms
        )

        # Add optional metrics
        if output.phase_metrics:
            response.phase_metrics = PhaseMetrics(**output.phase_metrics)

        if output.memory_stats:
            response.memory_stats = MemoryStats(**output.memory_stats)

        return response

    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}"
        )


# Development server entry point
if __name__ == "__main__":
    import uvicorn

    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))

    logger.info(f"Starting development server on {host}:{port}")

    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
