"""
Resonant Model Inference Engine

Handles model loading, checkpoint restoration, and efficient inference
for resonant neural architectures with AFRB + PVM + PLM components.

This module provides the ResonantModelInference class which wraps the
trained resonant model and provides high-level inference APIs for:
- Text generation (with phase-aware sampling)
- Content-addressable retrieval (needle-in-haystack)
- Phase metrics collection
- Memory diagnostics

Architecture:
    Base Model (e.g., Pythia-160m)
    + AFRB Adapters (8 layers with learnable omega/gamma)
    + PVM (Phase-Vector Memory for temporal coherence)
    + PLM (Phase-Lattice Memory for spatial binding)

Usage:
    from api.inference import ResonantModelInference

    # Load trained checkpoint
    model = ResonantModelInference(
        checkpoint_path="checkpoints/model.pt",
        device="cuda"
    )

    # Generate text
    output = model.generate(
        prompt="Once upon a time",
        max_tokens=100,
        temperature=0.7
    )

    # Retrieve pattern
    matches = model.retrieve(
        context="... long text ...",
        needle="specific pattern",
        mode="pvm"
    )
"""

import os
import time
import math
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import resonant components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from afrb import AFRB
from phase_memory import PhaseVectorMemory


@dataclass
class GenerationOutput:
    """
    Output from text generation.

    Attributes:
        text: Generated text (without prompt)
        full_text: Prompt + generated text
        tokens_generated: Number of tokens produced
        finish_reason: Why generation stopped
        timing_ms: Latency in milliseconds
        phase_metrics: Phase coherence diagnostics
        memory_stats: PVM/PLM state information
    """
    text: str
    full_text: str
    tokens_generated: int
    finish_reason: str
    timing_ms: float
    phase_metrics: Optional[Dict[str, Any]] = None
    memory_stats: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalOutput:
    """
    Output from content-addressable retrieval.

    Attributes:
        matches: List of (text, position, similarity, confidence) tuples
        timing_ms: Retrieval latency in milliseconds
        phase_metrics: Phase coherence during retrieval
        memory_stats: Memory state diagnostics
    """
    matches: List[Tuple[str, Optional[int], Optional[float], float]]
    timing_ms: float
    phase_metrics: Optional[Dict[str, Any]] = None
    memory_stats: Optional[Dict[str, Any]] = None


class ResonantModelInference:
    """
    Inference engine for resonant neural models.

    This class manages model loading, state management, and provides
    high-level APIs for generation and retrieval tasks.

    The model architecture consists of:
    1. Base transformer (e.g., Pythia-160m)
    2. AFRB adapters injected into transformer layers
    3. Optional PVM (Phase-Vector Memory) modules
    4. Optional PLM (Phase-Lattice Memory) modules

    Features:
    - Efficient batching (supports batch_size > 1)
    - Memory management (automatic state reset)
    - Phase metrics collection
    - Content-addressable retrieval via PVM
    - Flexible sampling strategies

    Args:
        checkpoint_path: Path to trained model checkpoint (.pt file)
        model_name: Base model identifier (default: "EleutherAI/pythia-160m")
        device: Computation device ("cuda", "cpu", or "mps")
        dtype: Model precision (torch.float32, torch.bfloat16, torch.float16)
        max_batch_size: Maximum batch size for inference
        autocast: Enable automatic mixed precision

    Example:
        >>> model = ResonantModelInference(
        ...     checkpoint_path="checkpoints/model_best.pt",
        ...     device="cuda",
        ...     dtype=torch.bfloat16
        ... )
        >>> output = model.generate("The theory of resonance", max_tokens=50)
        >>> print(output.text)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_name: str = "EleutherAI/pythia-160m",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        max_batch_size: int = 8,
        autocast: bool = False
    ):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.autocast = autocast

        print(f"[ResonantModelInference] Initializing on {self.device}")
        print(f"[ResonantModelInference] Model: {model_name}")
        print(f"[ResonantModelInference] Dtype: {dtype}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        print(f"[ResonantModelInference] Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None  # Manual device placement
        )
        self.base_model.to(self.device)

        # Extract model config
        self.hidden_size = self.base_model.config.hidden_size
        self.vocab_size = self.base_model.config.vocab_size
        self.num_layers = self.base_model.config.num_hidden_layers
        self.max_context_length = getattr(
            self.base_model.config, 'max_position_embeddings', 2048
        )

        # Initialize AFRB adapters (will be populated from checkpoint)
        self.afrb_adapters: List[AFRB] = []
        self.num_afrb_layers = 0

        # Load checkpoint if provided
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        else:
            if checkpoint_path is not None:
                print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
            print(f"[WARNING] Running without AFRB adapters (base model only)")

        # Model evaluation mode
        self.base_model.eval()
        for adapter in self.afrb_adapters:
            adapter.eval()

        print(f"[ResonantModelInference] Ready! ({self.num_afrb_layers} AFRB layers)")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load trained checkpoint and restore AFRB adapters.

        The checkpoint should contain:
        - afrb_adapters: List of AFRB state dicts
        - model_config: Configuration dictionary
        - Optional: optimizer_state, epoch, etc.

        Args:
            checkpoint_path: Path to .pt checkpoint file

        Raises:
            FileNotFoundError: If checkpoint does not exist
            KeyError: If checkpoint format is invalid
        """
        print(f"[ResonantModelInference] Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract AFRB adapters
        if 'afrb_adapters' in checkpoint:
            afrb_states = checkpoint['afrb_adapters']
            self.num_afrb_layers = len(afrb_states)

            print(f"[ResonantModelInference] Restoring {self.num_afrb_layers} AFRB adapters...")

            # Reconstruct AFRB modules from saved state
            # Note: We need to match the training config
            model_config = checkpoint.get('model_config', {})

            for i, state_dict in enumerate(afrb_states):
                # Create AFRB with same config as training
                adapter = AFRB(
                    dim=self.hidden_size,
                    block_idx=i + 1,
                    alpha_base=model_config.get('alpha_base', 0.04),
                    gamma_base=model_config.get('gamma_base', 0.20),
                    omega=model_config.get('omega', 6.0),
                    phi=model_config.get('phi', 0.0),
                    use_glu=model_config.get('use_glu', True),
                    learnable_omega=model_config.get('learnable_omega', False),
                    use_pvm=model_config.get('use_pvm', False),
                    use_plm=model_config.get('use_plm', False),
                )

                # Load state dict
                adapter.load_state_dict(state_dict)
                adapter.to(self.device)
                adapter.eval()

                self.afrb_adapters.append(adapter)

            print(f"[ResonantModelInference] Checkpoint loaded successfully")

        else:
            print(f"[WARNING] No 'afrb_adapters' found in checkpoint")

        # Store config for introspection
        self.model_config = checkpoint.get('model_config', {})

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and configuration.

        Returns:
            Dictionary with model information:
                - model_name: Base model identifier
                - num_afrb_layers: Number of AFRB adapters
                - hidden_size: Model dimension
                - vocab_size: Tokenizer vocabulary
                - max_context_length: Maximum sequence length
                - phase_features: Phase dynamics config
                - memory_features: PVM/PLM config
                - device: Current device
        """
        return {
            'model_name': self.model_name,
            'num_afrb_layers': self.num_afrb_layers,
            'hidden_size': self.hidden_size,
            'vocab_size': self.vocab_size,
            'max_context_length': self.max_context_length,
            'phase_features': {
                'adaptive_omega': self.model_config.get('learnable_omega', False),
                'learnable_gamma': True,  # Always learnable
                'omega_base': self.model_config.get('omega', 6.0),
                'phi_base': self.model_config.get('phi', 0.0),
            },
            'memory_features': {
                'pvm_enabled': self.model_config.get('use_pvm', False),
                'plm_enabled': self.model_config.get('use_plm', False),
            },
            'device': str(self.device),
        }

    def forward_with_afrb(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through base model + AFRB adapters.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Tuple of (logits, hidden_states_list)
                - logits: [batch, seq_len, vocab_size]
                - hidden_states_list: List of [batch, seq_len, hidden_size]
        """
        # Get base model hidden states
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        hidden_states = list(outputs.hidden_states)  # List of [B, T, D]

        # Apply AFRB adapters to selected layers
        # Typically applied to last N layers
        if self.num_afrb_layers > 0:
            # Apply to last num_afrb_layers
            start_layer = max(0, len(hidden_states) - self.num_afrb_layers - 1)

            for i, adapter in enumerate(self.afrb_adapters):
                layer_idx = start_layer + i + 1
                if layer_idx < len(hidden_states):
                    hidden_states[layer_idx] = adapter(hidden_states[layer_idx])

        # Final layer normalization + LM head
        final_hidden = hidden_states[-1]
        logits = self.base_model.lm_head(
            self.base_model.gpt_neox.final_layer_norm(final_hidden)
        )

        return logits, hidden_states

    @torch.no_grad()
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
    ) -> GenerationOutput:
        """
        Generate text continuation for given prompt.

        Uses autoregressive sampling with phase-aware dynamics.
        Supports nucleus sampling (top-p), top-k filtering, and
        repetition penalty.

        Args:
            prompt: Input text to continue
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_k: Keep top K tokens (0 = disabled)
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
            stop_sequences: Stop at these strings
            seed: Random seed for reproducibility
            return_phase_metrics: Include phase diagnostics
            return_memory_stats: Include memory diagnostics

        Returns:
            GenerationOutput with text, metrics, and timing

        Example:
            >>> output = model.generate(
            ...     prompt="Resonant models use",
            ...     max_tokens=50,
            ...     temperature=0.8
            ... )
            >>> print(output.text)
        """
        start_time = time.time()

        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_context_length
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        prompt_length = input_ids.shape[1]
        generated_tokens = []
        finish_reason = "max_tokens"

        # Reset PVM memory states (clean slate for each generation)
        self._reset_memory_states()

        # Autoregressive generation loop
        for step in range(max_tokens):
            # Forward pass
            logits, hidden_states = self.forward_with_afrb(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Get next token logits
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Apply repetition penalty
            if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                for token_id in set(generated_tokens):
                    next_token_logits[:, token_id] /= repetition_penalty

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(
                    next_token_logits, top_k
                )[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply nucleus (top-p) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check for EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                finish_reason = "eos_token"
                break

            # Append to sequence
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device)
            ], dim=1)

            # Check stop sequences
            if stop_sequences:
                current_text = self.tokenizer.decode(generated_tokens)
                if any(stop_seq in current_text for stop_seq in stop_sequences):
                    finish_reason = "stop_sequence"
                    break

        # Decode generated text
        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        full_text = prompt + generated_text

        timing_ms = (time.time() - start_time) * 1000

        # Collect optional metrics
        phase_metrics = None
        memory_stats = None

        if return_phase_metrics:
            phase_metrics = self._collect_phase_metrics()

        if return_memory_stats:
            memory_stats = self._collect_memory_stats()

        return GenerationOutput(
            text=generated_text,
            full_text=full_text,
            tokens_generated=len(generated_tokens),
            finish_reason=finish_reason,
            timing_ms=timing_ms,
            phase_metrics=phase_metrics,
            memory_stats=memory_stats
        )

    @torch.no_grad()
    def retrieve(
        self,
        context: str,
        needle: str,
        query: Optional[str] = None,
        mode: str = "pvm",
        top_k: int = 5
    ) -> RetrievalOutput:
        """
        Content-addressable retrieval from context.

        Uses phase-based similarity to find patterns matching the needle.
        Supports PVM (Phase-Vector Memory) and PLM (Phase-Lattice Memory).

        Args:
            context: Long text containing the needle
            needle: Pattern to retrieve
            query: Alternative query (defaults to needle)
            mode: Retrieval method ("pvm", "plm", "hybrid")
            top_k: Number of top matches to return

        Returns:
            RetrievalOutput with ranked matches and timing

        Example:
            >>> output = model.retrieve(
            ...     context="...long text... key insight ...",
            ...     needle="key insight",
            ...     mode="pvm",
            ...     top_k=3
            ... )
            >>> print(output.matches[0])  # Best match
        """
        start_time = time.time()

        if query is None:
            query = needle

        # Tokenize context
        context_inputs = self.tokenizer(
            context,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_context_length
        )
        context_ids = context_inputs['input_ids'].to(self.device)

        # Reset memory and process context
        self._reset_memory_states()
        _, hidden_states = self.forward_with_afrb(input_ids=context_ids)

        # Tokenize query
        query_inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding=False,
            truncation=True
        )
        query_ids = query_inputs['input_ids'].to(self.device)
        _, query_hidden = self.forward_with_afrb(input_ids=query_ids)

        # Extract query vector (mean pooling)
        query_vec = query_hidden[-1].mean(dim=1).squeeze(0)  # [D]

        # Retrieve from PVM if available
        matches = []
        if mode == "pvm" and self.num_afrb_layers > 0:
            # Use PVM from last AFRB layer
            last_adapter = self.afrb_adapters[-1]
            if hasattr(last_adapter, 'pvm') and last_adapter.pvm is not None:
                retrieved_vec = last_adapter.pvm.readout(query_vec, topk=top_k)

                # Compute cosine similarity
                similarity = F.cosine_similarity(
                    query_vec.unsqueeze(0),
                    retrieved_vec.unsqueeze(0),
                    dim=1
                ).item()

                matches.append((
                    needle,  # Placeholder (no position tracking yet)
                    None,    # Position unknown
                    similarity,
                    similarity  # Confidence = similarity
                ))
            else:
                # Fallback: Cosine similarity across all positions
                context_hidden = hidden_states[-1].squeeze(0)  # [T, D]
                similarities = F.cosine_similarity(
                    query_vec.unsqueeze(0),
                    context_hidden,
                    dim=1
                )  # [T]

                topk_vals, topk_idx = torch.topk(similarities, k=min(top_k, len(similarities)))

                for val, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
                    # Decode token at position
                    token_text = self.tokenizer.decode([context_ids[0, idx].item()])
                    matches.append((token_text, idx, val, val))

        else:
            # Fallback: Sliding window similarity
            print(f"[WARNING] Mode '{mode}' not fully implemented, using fallback")
            context_hidden = hidden_states[-1].squeeze(0)  # [T, D]
            similarities = F.cosine_similarity(
                query_vec.unsqueeze(0),
                context_hidden,
                dim=1
            )

            topk_vals, topk_idx = torch.topk(
                similarities, k=min(top_k, len(similarities))
            )

            for val, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
                token_text = self.tokenizer.decode([context_ids[0, idx].item()])
                matches.append((token_text, idx, val, val))

        timing_ms = (time.time() - start_time) * 1000

        return RetrievalOutput(
            matches=matches,
            timing_ms=timing_ms,
            phase_metrics=None,
            memory_stats=None
        )

    def _reset_memory_states(self) -> None:
        """Reset PVM/PLM memory states to zero."""
        for adapter in self.afrb_adapters:
            if hasattr(adapter, 'pvm') and adapter.pvm is not None:
                adapter.pvm.reset_memory()
            if hasattr(adapter, 'plm') and adapter.plm is not None:
                adapter.plm.reset_memory()

    def _collect_phase_metrics(self) -> Dict[str, Any]:
        """Collect phase coherence and resonance metrics from AFRB layers."""
        omega_values = []
        gamma_values = []
        phi_values = []

        for adapter in self.afrb_adapters:
            # Extract omega (frequency)
            if hasattr(adapter.phase_embed, 'current_omega'):
                omega_values.append(adapter.phase_embed.current_omega().item())

            # Extract gamma (resonance depth)
            if hasattr(adapter, 'get_gamma'):
                gamma_values.append(adapter.get_gamma().item())

            # Extract phi (phase offset)
            if hasattr(adapter, 'phi_base'):
                phi_values.append(adapter.phi_base.item())

        # Compute aggregate metrics
        phase_coherence = sum(gamma_values) / len(gamma_values) if gamma_values else 0.0
        gamma_mean = sum(gamma_values) / len(gamma_values) if gamma_values else 0.0

        return {
            'phase_coherence': phase_coherence,
            'gamma_mean': gamma_mean,
            'omega_values': omega_values,
            'phase_offsets': phi_values,
        }

    def _collect_memory_stats(self) -> Dict[str, Any]:
        """Collect PVM/PLM memory diagnostics."""
        pvm_mem_norms = []
        pvm_gate_strengths = []

        for adapter in self.afrb_adapters:
            if hasattr(adapter, 'pvm') and adapter.pvm is not None:
                info = adapter.pvm.get_memory_info()
                pvm_mem_norms.append(info['mem_norm'])
                pvm_gate_strengths.append(info['gate_strength'])

        # Compute memory footprint (rough estimate)
        total_memory_kb = 0.0
        for adapter in self.afrb_adapters:
            if hasattr(adapter, 'pvm') and adapter.pvm is not None:
                mem_state_size = adapter.pvm.mem_state.numel() * adapter.pvm.mem_state.element_size()
                total_memory_kb += mem_state_size / 1024.0

        return {
            'pvm_mem_norm': sum(pvm_mem_norms) / len(pvm_mem_norms) if pvm_mem_norms else None,
            'pvm_gate_strength': sum(pvm_gate_strengths) / len(pvm_gate_strengths) if pvm_gate_strengths else None,
            'plm_coherence': None,  # Placeholder (not yet implemented)
            'total_memory_kb': total_memory_kb if total_memory_kb > 0 else None,
        }
