"""
Resonant Model API Client Example

Demonstrates how to interact with the Resonant Model API for:
- Text generation with phase-aware sampling
- Content retrieval (needle-in-haystack)
- Model introspection and diagnostics

This script provides a simple Python interface to the REST API.
For production use, consider using an async HTTP client (httpx, aiohttp).

Requirements:
    pip install requests

Usage:
    # Start API server first
    export MODEL_CHECKPOINT=checkpoints/model_best.pt
    uvicorn api.server:app --host 0.0.0.0 --port 8000

    # Run client
    python examples/api_client.py

Author: Damjan Žakelj
"""

import requests
import json
import sys
from typing import Optional, Dict, Any, List


class ResonantModelClient:
    """
    Python client for Resonant Model API.

    Provides high-level methods for text generation, retrieval, and
    model introspection.

    Args:
        base_url: API base URL (default: http://localhost:8000)
        timeout: Request timeout in seconds (default: 120)

    Example:
        >>> client = ResonantModelClient()
        >>> result = client.generate("Once upon a time", max_tokens=50)
        >>> print(result["text"])
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """
        Check API server health.

        Returns:
            Health status dictionary with model readiness and uptime

        Example:
            >>> client = ResonantModelClient()
            >>> health = client.health_check()
            >>> print(f"Status: {health['status']}")
        """
        response = self.session.get(
            f"{self.base_url}/v1/health",
            timeout=10  # Short timeout for health checks
        )
        response.raise_for_status()
        return response.json()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and configuration.

        Returns:
            Model information dictionary with architecture details

        Example:
            >>> client = ResonantModelClient()
            >>> info = client.get_model_info()
            >>> print(f"Model: {info['model_name']}")
            >>> print(f"AFRB layers: {info['num_afrb_layers']}")
        """
        response = self.session.get(
            f"{self.base_url}/v1/model/info",
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        seed: Optional[int] = None,
        return_phase_metrics: bool = False,
        return_memory_stats: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text continuation.

        Args:
            prompt: Input text to continue
            max_tokens: Maximum tokens to generate (1-2048)
            temperature: Sampling temperature (0.0-2.0)
            top_k: Top-K sampling (0 = disabled)
            top_p: Nucleus sampling threshold (0.0-1.0)
            repetition_penalty: Penalty for token repetition (1.0-2.0)
            stop_sequences: Stop at these strings
            seed: Random seed for reproducibility
            return_phase_metrics: Include phase coherence diagnostics
            return_memory_stats: Include PVM/PLM memory diagnostics

        Returns:
            Generation result with text, timing, and optional metrics

        Raises:
            requests.HTTPError: If request fails

        Example:
            >>> client = ResonantModelClient()
            >>> result = client.generate(
            ...     prompt="Resonant neural networks",
            ...     max_tokens=50,
            ...     temperature=0.8,
            ...     return_phase_metrics=True
            ... )
            >>> print(result["text"])
            >>> print(f"Coherence: {result['phase_metrics']['phase_coherence']}")
        """
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "return_phase_metrics": return_phase_metrics,
            "return_memory_stats": return_memory_stats,
        }

        if stop_sequences is not None:
            payload["stop_sequences"] = stop_sequences
        if seed is not None:
            payload["seed"] = seed

        response = self.session.post(
            f"{self.base_url}/v1/generate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def retrieve(
        self,
        context: str,
        needle: str,
        query: Optional[str] = None,
        retrieval_mode: str = "pvm",
        top_k: int = 5,
        return_positions: bool = True,
        return_similarity_scores: bool = True
    ) -> Dict[str, Any]:
        """
        Content-addressable retrieval (needle-in-haystack).

        Args:
            context: Long text containing the needle
            needle: Pattern to retrieve
            query: Alternative query text (defaults to needle)
            retrieval_mode: Retrieval method ("pvm", "plm", "hybrid")
            top_k: Number of top matches to return
            return_positions: Include token position indices
            return_similarity_scores: Include cosine similarity scores

        Returns:
            Retrieval result with ranked matches and timing

        Raises:
            requests.HTTPError: If request fails

        Example:
            >>> client = ResonantModelClient()
            >>> result = client.retrieve(
            ...     context="...long text... The answer is 42 ...",
            ...     needle="answer is 42",
            ...     retrieval_mode="pvm",
            ...     top_k=3
            ... )
            >>> for match in result["matches"]:
            ...     print(f"{match['text']} (similarity: {match['similarity']:.3f})")
        """
        payload = {
            "context": context,
            "needle": needle,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "return_positions": return_positions,
            "return_similarity_scores": return_similarity_scores,
        }

        if query is not None:
            payload["query"] = query

        response = self.session.post(
            f"{self.base_url}/v1/retrieve",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the HTTP session."""
        self.session.close()


def demo_generation():
    """Demonstrate text generation with various parameters."""
    print("=" * 60)
    print("Demo 1: Text Generation")
    print("=" * 60)

    client = ResonantModelClient()

    # Example 1: Simple generation
    print("\n[1] Simple generation:")
    result = client.generate(
        prompt="Once upon a time in a neural network",
        max_tokens=50,
        temperature=0.8
    )
    print(f"Prompt: {result['prompt']}")
    print(f"Generated: {result['text']}")
    print(f"Tokens: {result['tokens_generated']}")
    print(f"Timing: {result['timing_ms']:.1f}ms")

    # Example 2: With phase metrics
    print("\n[2] Generation with phase metrics:")
    result = client.generate(
        prompt="The theory of resonant neural networks",
        max_tokens=30,
        temperature=0.7,
        return_phase_metrics=True
    )
    print(f"Generated: {result['text']}")
    if result.get('phase_metrics'):
        print(f"Phase coherence: {result['phase_metrics']['phase_coherence']:.3f}")
        print(f"Gamma mean: {result['phase_metrics']['gamma_mean']:.3f}")

    # Example 3: Low temperature (deterministic)
    print("\n[3] Low temperature (greedy):")
    result = client.generate(
        prompt="Attention is all you need, but",
        max_tokens=20,
        temperature=0.1,  # Nearly deterministic
        seed=42
    )
    print(f"Generated: {result['text']}")

    # Example 4: High temperature (creative)
    print("\n[4] High temperature (creative):")
    result = client.generate(
        prompt="In the year 2050, AI will",
        max_tokens=30,
        temperature=1.5,  # More random
    )
    print(f"Generated: {result['text']}")


def demo_retrieval():
    """Demonstrate content retrieval (needle-in-haystack)."""
    print("\n" + "=" * 60)
    print("Demo 2: Content Retrieval")
    print("=" * 60)

    client = ResonantModelClient()

    # Create a context with embedded needles
    context = """
    Neural networks have revolutionized machine learning. Traditional architectures
    rely on attention mechanisms for sequence processing. However, attention has
    quadratic complexity. Recent research explores alternative mechanisms.

    The secret code is X7Z9 for accessing the system.

    Phase-based neural networks offer a promising direction. They replace attention
    with resonant frequency coupling. This achieves O(d) memory complexity instead
    of O(n²). The key insight is resonance enables long-range binding without
    explicit pairwise comparisons.

    Applications include language modeling, time series analysis, and more.
    """

    print("\n[1] Retrieve exact needle:")
    result = client.retrieve(
        context=context,
        needle="secret code is X7Z9",
        retrieval_mode="pvm",
        top_k=3
    )
    print(f"Needle: {result['needle']}")
    print(f"Top match: {result['matches'][0]['text']}")
    if result['matches'][0].get('similarity'):
        print(f"Similarity: {result['matches'][0]['similarity']:.3f}")

    print("\n[2] Retrieve semantic pattern:")
    result = client.retrieve(
        context=context,
        needle="key insight",
        retrieval_mode="pvm",
        top_k=5
    )
    print(f"Needle: {result['needle']}")
    print("Top matches:")
    for i, match in enumerate(result['matches'][:3], 1):
        print(f"  {i}. {match['text'][:50]}... "
              f"(sim: {match.get('similarity', 0.0):.3f})")


def demo_model_info():
    """Demonstrate model introspection."""
    print("\n" + "=" * 60)
    print("Demo 3: Model Information")
    print("=" * 60)

    client = ResonantModelClient()

    # Health check
    print("\n[1] Health check:")
    health = client.health_check()
    print(f"Status: {health['status']}")
    print(f"Model loaded: {health['model_loaded']}")
    print(f"Uptime: {health['uptime_seconds']:.1f}s")
    print(f"Version: {health['version']}")

    # Model info
    print("\n[2] Model configuration:")
    info = client.get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"AFRB layers: {info['num_afrb_layers']}")
    print(f"Hidden size: {info['hidden_size']}")
    print(f"Vocab size: {info['vocab_size']}")
    print(f"Max context: {info['max_context_length']}")
    print(f"Device: {info['device']}")

    print("\n[3] Phase features:")
    phase = info['phase_features']
    print(f"Adaptive omega: {phase['adaptive_omega']}")
    print(f"Learnable gamma: {phase['learnable_gamma']}")
    print(f"Omega base: {phase['omega_base']}")

    print("\n[4] Memory features:")
    memory = info['memory_features']
    print(f"PVM enabled: {memory['pvm_enabled']}")
    print(f"PLM enabled: {memory['plm_enabled']}")


def demo_error_handling():
    """Demonstrate error handling."""
    print("\n" + "=" * 60)
    print("Demo 4: Error Handling")
    print("=" * 60)

    client = ResonantModelClient()

    # Invalid parameters
    print("\n[1] Invalid max_tokens:")
    try:
        result = client.generate(
            prompt="Test",
            max_tokens=10000  # Too large
        )
    except requests.HTTPError as e:
        print(f"Error: {e.response.status_code}")
        error_data = e.response.json()
        print(f"Message: {error_data['message']}")

    # Invalid temperature
    print("\n[2] Invalid temperature:")
    try:
        result = client.generate(
            prompt="Test",
            temperature=5.0  # Too high
        )
    except requests.HTTPError as e:
        print(f"Error: {e.response.status_code}")
        error_data = e.response.json()
        print(f"Message: {error_data['message']}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Resonant Model API Client - Demo Suite")
    print("=" * 60)
    print("\nThis script demonstrates the Resonant Model API capabilities.")
    print("Make sure the API server is running:")
    print("  uvicorn api.server:app --host 0.0.0.0 --port 8000")
    print()

    try:
        # Check if server is running
        client = ResonantModelClient()
        health = client.health_check()

        if not health['model_loaded']:
            print("WARNING: Server is running but model is not loaded!")
            print("Set MODEL_CHECKPOINT environment variable and restart server.")
            sys.exit(1)

        print(f"Server status: {health['status']}")
        print(f"API version: {health['version']}")
        print()

        # Run demos
        demo_generation()
        demo_retrieval()
        demo_model_info()
        demo_error_handling()

        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)

    except requests.ConnectionError:
        print("ERROR: Cannot connect to API server!")
        print("Start the server with:")
        print("  export MODEL_CHECKPOINT=checkpoints/model_best.pt")
        print("  uvicorn api.server:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
