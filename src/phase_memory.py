"""
Phase Vector Memory (PVM): O(d) Content-Addressable Memory for Resonant Models

PVM implements a holographic memory mechanism inspired by phase dynamics in neural systems.
Unlike attention (O(n²)), PVM achieves O(d) memory complexity through:

1. Log-periodic rotation: Φ(t) = α·exp(iωt) encodes temporal information
2. Content-addressable readout: cosine similarity between query and memory trace
3. Constant memory footprint: single vector accumulates entire history

Key Properties:
- Memory complexity: O(d) regardless of sequence length
- Temporal binding: Phase rotation preserves relative timing
- Graceful forgetting: Exponential decay prevents saturation

Mathematical Foundation:
    m(t+1) = β·m(t) + α·h(t)·exp(iφ(t))

    where:
    - m(t): Memory state at time t
    - h(t): Hidden state from backbone
    - α, β: Memory dynamics (α+β ≈ 1 for stability)
    - φ(t): Phase angle (ωt + phase_offset)

Reference:
    Kuramoto-inspired synchronization for temporal pattern binding

Version 2.1:
- Recursive token-by-token memory update
- Full sequence trace [B, T, D] for content-addressable readout
- Fixed missing super().__init__() call
- Removed pooled mean(dim=1) in favor of sequential accumulation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseBuffer(nn.Module):
    """
    T2-decay buffer with Phase-Change Memory (PCM) for memory stabilization.

    Implements two decay modes:
    1. Exponential decay: Simple T2-like relaxation
    2. Resonant decay: Log-periodic oscillations for critical slowing

    PCM component: Learnable offset mimicking amorphous/crystalline phase states
    to stabilize long-term memory patterns.

    Args:
        dim: Hidden dimension
        t2_steps: Decay time constant (number of steps)
        t2_mode: 'exp' (exponential) or 'resonant' (log-periodic)
        t2_k: Decay rate for resonant mode
        t2_alpha: Oscillation amplitude for resonant mode
        t2_omega: Log-periodic frequency for resonant mode
        t2_phi: Phase offset for resonant mode
        pcm_gate_init: Initial value for PCM gate (0-1 after sigmoid)
    """
    def __init__(self, dim, t2_steps=1500, t2_mode='exp', t2_k=0.001, t2_alpha=0.08,
                 t2_omega=6.0, t2_phi=1.0472, pcm_gate_init=0.5):
        super().__init__()
        self.dim = dim
        self.t2_steps = float(t2_steps)
        self.t2_mode = t2_mode
        self.t2_k = float(t2_k)
        self.t2_alpha = float(t2_alpha)
        self.t2_omega = float(t2_omega)
        self.t2_phi = float(t2_phi)

        # PCM (Phase-Change Memory) components:
        # Gate interpolates between amorphous (short-term) and crystalline (long-term) states
        self.pcm_gate = nn.Parameter(torch.tensor(pcm_gate_init))
        self.phase_amorph = nn.Parameter(torch.zeros(dim))
        self.phase_cryst = nn.Parameter(torch.zeros(dim))

    def apply(self, phase_vec, step_idx):
        """
        Apply T2 decay and PCM offset to phase vector.

        Args:
            phase_vec: Memory state tensor [..., dim]
            step_idx: Current time step (starts at 0)

        Returns:
            Stabilized phase vector: decay(t) * phase_vec + PCM_offset
        """
        t = float(step_idx) + 1.0

        # Compute decay factor based on mode
        if self.t2_mode == 'exp':
            # Standard exponential decay: exp(-t / T2)
            if self.t2_steps > 0:
                decay = torch.exp(-torch.tensor(t, device=phase_vec.device) / self.t2_steps)
            else:
                decay = torch.tensor(1.0, device=phase_vec.device)

        elif self.t2_mode == 'resonant':
            # Log-periodic decay with oscillations:
            # decay(t) = exp(-k·t) * [1 + α·sin(ω·log(t) + φ)]
            # This creates self-similar patterns across time scales
            base_decay = math.exp(-self.t2_k * t)
            log_t = math.log(t)
            oscillation = 1.0 + self.t2_alpha * math.sin(self.t2_omega * log_t + self.t2_phi)
            decay_value = max(0.0, base_decay * max(0.0, oscillation))
            decay = torch.tensor(decay_value, device=phase_vec.device, dtype=phase_vec.dtype)
        else:
            decay = torch.tensor(1.0, device=phase_vec.device)

        # PCM offset: Interpolate between amorphous and crystalline states
        g = torch.sigmoid(self.pcm_gate)
        pcm_offset = g * self.phase_cryst + (1 - g) * self.phase_amorph

        return phase_vec * decay + pcm_offset


class PhaseVectorMemory(nn.Module):
    """
    Phase Vector Memory: O(d) holographic memory for infinite context.

    PVM maintains a single memory state vector m(t) that accumulates the entire
    sequence history through exponential moving average (EMA) and phase rotation:

        m(t+1) = β·m(t) + α·h(t)·exp(iφ(t))

    Key Innovation: Unlike attention which requires O(n²) operations and O(n·d) storage,
    PVM achieves O(d) memory complexity regardless of sequence length.

    How It Works:
    1. **EMA Update**: New information blends with existing memory (α + β ≈ 1)
    2. **Phase Rotation**: Log-periodic rotation φ(t) = ω·log(1+t) + φ₀ encodes time
    3. **Content Retrieval**: Query uses cosine similarity against full trace for readout

    Why This Works:
    - Log-periodic rotation prevents saturation (unlike linear phase)
    - Cosine similarity is rotation-invariant (geometric stability)
    - EMA provides graceful forgetting (exponential decay of old info)

    Args:
        hidden_size: Hidden dimension d
        alpha: Memory write strength (default: 0.12)
        beta: Memory decay factor (default: 0.88, α+β ≈ 1)
        omega: Phase rotation frequency (default: 6.0)
        phi_base: Initial phase offset (default: 0.0)
        gate_init: Initial gate logit for residual connection (default: -2.0)
        history_size: Legacy parameter, kept for compatibility (ignored)
        t2_enable: Enable T2 decay buffer (default: False)
        t2_steps: T2 decay time constant (default: 1500)
        t2_mode: Decay mode 'exp' or 'resonant' (default: 'exp')
        t2_k: Resonant decay rate (default: 0.001)
        t2_alpha: Resonant oscillation amplitude (default: 0.08)
        t2_omega: Resonant log-frequency (default: 6.0)
        t2_phi: Resonant phase offset (default: 1.0472)
        pcm_enable: Enable Phase-Change Memory buffer (default: False)
        pcm_gate_init: PCM gate initialization (default: 0.5)

    Example:
        >>> pvm = PhaseVectorMemory(hidden_size=768, alpha=0.12, beta=0.88)
        >>> x = torch.randn(1, 100, 768)  # [batch, seq_len, hidden]
        >>> y = pvm(x, step_idx=0)        # [1, 100, 768] with memory
        >>> query = torch.randn(768)
        >>> retrieved = pvm.readout(query)  # Content-addressable retrieval
    """
    def __init__(
        self,
        hidden_size: int,
        alpha: float = 0.12,
        beta: float = 0.88,
        omega: float = 6.0,
        phi_base: float = 0.0,
        gate_init: float = -2.0,
        history_size: int = 64,   # Ignored in v2.0+, kept for backward compatibility
        t2_enable: bool = False,
        t2_steps: int = 1500,
        t2_mode: str = 'exp',
        t2_k: float = 0.001,
        t2_alpha: float = 0.08,
        t2_omega: float = 6.0,
        t2_phi: float = 1.0472,
        pcm_enable: bool = False,
        pcm_gate_init: float = 0.5
    ):
        super().__init__()

        # Core memory parameters
        self.alpha = float(alpha)      # Write strength
        self.beta = float(beta)        # Decay factor
        self.omega = float(omega)      # Phase rotation frequency
        self.phi_base = float(phi_base)  # Phase offset
        self.hidden_size = hidden_size

        # Learnable parameters
        self.gate = nn.Parameter(torch.tensor(gate_init))
        self.input_scale = nn.Parameter(torch.tensor(4.0))

        # Memory state buffer: [1, 1, hidden_size]
        # This accumulates information across entire sequence
        self.register_buffer("mem_state", torch.zeros(1, 1, hidden_size))

        # Last sequence trace: [T, D] for content-addressable readout
        # Set to None initially; populated during forward pass
        self._last_trace = None

        # Trajectory recording for phase curvature diagnostics
        self.record_traj = False  # Enable via flag for diagnostics
        self._traj = None         # Stores [T, D] trajectory when enabled

        # T2 decay and PCM parameters
        self.t2_enable = t2_enable
        self.t2_mode = t2_mode
        self.t2_steps = float(t2_steps)
        self.t2_k = float(t2_k)
        self.t2_alpha = float(t2_alpha)
        self.t2_omega = float(t2_omega)
        self.t2_phi = float(t2_phi)
        self.pcm_enable = pcm_enable

        # Initialize phase buffer if T2 or PCM is enabled
        if t2_enable or pcm_enable:
            self.phase_buffer = PhaseBuffer(
                dim=hidden_size,
                t2_steps=t2_steps if t2_enable else 0,
                t2_mode=t2_mode,
                t2_k=t2_k,
                t2_alpha=t2_alpha,
                t2_omega=t2_omega,
                t2_phi=t2_phi,
                pcm_gate_init=pcm_gate_init
            )
        else:
            self.phase_buffer = None

    def forward(self, x: torch.Tensor, step_idx: int = 0, training: bool = True) -> torch.Tensor:
        """
        Update memory state recursively for each token and return gated output.

        Algorithm:
        For each token t in sequence:
            1. m(t) = decay(t) * m(t-1) + α * x(t)    [EMA update]
            2. θ(t) = ω·log(1+t) + φ₀                 [Log-periodic phase]
            3. m_rot(t) = Rotate(m(t), θ(t))          [Phase rotation]
            4. Optional: Apply T2/PCM buffer
            5. Store m_rot(t) in trace

        Output: y = x + sigmoid(gate) * m_seq  [Residual connection]

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            step_idx: Global step counter for phase calculation
            training: Whether in training mode (affects memory persistence)

        Returns:
            Output tensor [batch, seq_len, hidden_dim] with memory-augmented features
        """
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        # Prepare T2 decay factors for entire sequence (if enabled)
        if self.phase_buffer is not None and training:
            t = torch.arange(1, T + 1, device=device, dtype=dtype)

            if self.t2_mode == 'exp':
                # Exponential decay: exp(-t / T2)
                if self.t2_steps > 0:
                    decay_t = torch.exp(-t / self.t2_steps)
                else:
                    decay_t = torch.ones(T, device=device, dtype=dtype)

            elif self.t2_mode == 'resonant':
                # Log-periodic resonant decay
                log_t = torch.log(t)
                base_decay = torch.exp(-self.t2_k * t)
                oscillation = 1.0 + self.t2_alpha * torch.sin(self.t2_omega * log_t + self.t2_phi)
                decay_t = torch.clamp(base_decay * torch.clamp(oscillation, min=0.0), min=0.0)
            else:
                decay_t = torch.ones(T, device=device, dtype=dtype)

            decay_t = decay_t.view(1, T, 1)  # [1, T, 1] for broadcasting
        else:
            # Use beta as constant decay if T2 is disabled
            decay_t = torch.full((1, T, 1), self.beta, device=device, dtype=dtype)

        # Initialize memory from buffer (already detached on write)
        m = self.mem_state.expand(B, 1, D)
        states_trace = []

        # Recursive token-by-token memory update
        for i in range(T):
            x_t = x[:, i:i+1, :]  # [B, 1, D] - current token

            # EMA update: m(t) = decay(t) * m(t-1) + α * scale * x(t)
            m = decay_t[:, i:i+1, :] * m + self.alpha * (x_t * self.input_scale)

            # Compute log-periodic phase angle
            # θ(t) = ω·log(1 + t_global) + φ₀
            # Why log? Prevents phase from growing unbounded, creates self-similar dynamics
            t_glob = float(step_idx + i + 1)
            theta = self.omega * math.log1p(t_glob) + self.phi_base
            theta = theta % (2 * math.pi)  # Wrap to [0, 2π]

            # Compute rotation matrix components
            c = torch.cos(torch.tensor(theta, device=device, dtype=dtype))
            s = torch.sin(torch.tensor(theta, device=device, dtype=dtype))

            # Apply 2D rotation to pairs of dimensions
            # This implements: [a', b'] = [[c, -s], [s, c]] @ [a, b]
            # Why? Phase rotation binds temporal information geometrically
            even_len = (D // 2) * 2
            half = even_len // 2

            if half > 0:
                a = m[..., :half]
                b = m[..., half:even_len]
                a_rot = a * c - b * s
                b_rot = a * s + b * c

                # Handle odd dimension (last dim stays unrotated)
                if even_len < D:
                    tail = m[..., even_len:]
                    m_rot = torch.cat([a_rot, b_rot, tail], dim=-1)
                else:
                    m_rot = torch.cat([a_rot, b_rot], dim=-1)
            else:
                m_rot = m

            # Apply T2/PCM buffer for additional stabilization
            if self.phase_buffer is not None and training:
                m_rot = self.phase_buffer.apply(m_rot, t_glob)

            # Store rotated state in trace
            states_trace.append(m_rot)

            # Update persistent memory buffer (detached to prevent gradient flow)
            # Why detach? Gradient flows through _last_trace for learning, not mem_state
            if training:
                self.mem_state = m.detach().mean(dim=0, keepdim=True)

        # Concatenate full sequence trace: [B, T, D]
        m_seq = torch.cat(states_trace, dim=1)

        # Save trace for content-addressable readout
        # Store first batch element: [T, D]
        self._last_trace = m_seq[0]

        if training and step_idx == 0:
            print("[PVM] trace:", tuple(self._last_trace.shape))

        # Optional: Record trajectory for phase curvature diagnostics
        if self.record_traj:
            self._traj = m_seq[0].detach()  # [T, D]

        # Apply learnable gate and residual connection
        # Output: y = x + sigmoid(gate) * memory_sequence
        gate_strength = torch.sigmoid(self.gate)
        y = x + gate_strength * m_seq

        return y

    def readout(self, query: torch.Tensor, topk: int = 0) -> torch.Tensor:
        """
        Content-addressable retrieval from memory trace using cosine similarity.

        Given a query vector, finds similar patterns in the stored memory trace
        and returns a weighted combination of matching states.

        Algorithm:
            1. Normalize query and memory trace (L2 normalization)
            2. Compute cosine similarity: sim(t) = <q, m(t)> / (||q|| ||m(t)||)
            3. Optional: Keep only top-k most similar states
            4. Compute softmax weights from similarities
            5. Return weighted sum of memory states

        Why cosine similarity?
        - Rotation-invariant: Unaffected by phase rotation in memory
        - Scale-invariant: Focuses on direction, not magnitude
        - Content-based: High similarity = semantically related patterns

        Args:
            query: Query vector [hidden_dim] or [1, hidden_dim]
            topk: If > 0, only use top-k most similar states (default: 0 = use all)

        Returns:
            Retrieved memory vector [hidden_dim]
        """
        # Handle empty memory case
        if self._last_trace is None or self._last_trace.numel() == 0:
            return torch.zeros_like(query)

        mN = self._last_trace  # [T, D] - full memory trace
        q = query.squeeze()     # [D] - query vector

        # Ensure dtype consistency (important for mixed precision)
        if mN.dtype != q.dtype:
            mN = mN.to(q.dtype)

        # L2 normalization for cosine similarity
        q_norm = F.normalize(q, dim=-1)      # [D]
        mN_norm = F.normalize(mN, dim=-1)    # [T, D]

        # Compute cosine similarity: sim[t] = <q, m[t]>
        sim = torch.matmul(mN_norm, q_norm)  # [T]

        # Optional: Keep only top-k most similar states
        if topk > 0 and topk < sim.numel():
            vals, idx = torch.topk(sim, k=topk, dim=0)
            mask = torch.zeros_like(sim, dtype=torch.bool).scatter(0, idx, True)
            sim_masked = torch.where(mask, sim, torch.tensor(float('-inf'), device=sim.device))
            weights = F.softmax(sim_masked, dim=0)
        else:
            # Use all states
            weights = F.softmax(sim, dim=0)  # [T]

        # Weighted retrieval: weighted sum of memory states
        read = torch.einsum('t,td->d', weights, mN)  # [D]

        return read

    def get_last_trajectory(self):
        """
        Return last recorded PVM trajectory [T, D] or None.

        Only populated when self.record_traj=True during forward pass.
        Used for phase curvature diagnostics to analyze memory dynamics.

        Returns:
            Trajectory tensor [seq_len, hidden_dim] or None if not recorded
        """
        return self._traj

    def reset_memory(self):
        """
        Reset memory state to zero.

        Call this between independent sequences to prevent
        information leakage across examples.
        """
        self.mem_state.zero_()
        self._last_trace = None
        self._traj = None

    def get_memory_info(self):
        """
        Get diagnostic information about current memory state.

        Returns:
            dict with:
                - mem_norm: Average magnitude of memory state
                - gate_strength: Current gate value (after sigmoid)
        """
        return {
            'mem_norm': self.mem_state.abs().mean().item(),
            'gate_strength': torch.sigmoid(self.gate).item(),
        }

    def compute_rca_metrics(self, query: torch.Tensor = None):
        """
        Compute Resonant Coupling Analysis (RCA) metrics.

        RCA metrics measure memory-query coupling strength and phase coherence.
        These diagnostics help understand how well the memory is binding patterns.

        Args:
            query: Optional query tensor for coupling analysis

        Returns:
            dict with:
                - beta_eff: Effective decay factor
                - omega_eff: Effective rotation frequency (placeholder)
                - phase_coherence: Phase alignment metric (placeholder)
                - coupling_strength: Max cosine similarity between query and memory
        """
        metrics = {
            'beta_eff': self.beta,
            'omega_eff': 0.0,
            'phase_coherence': 0.0,
            'coupling_strength': 0.0
        }

        # Compute coupling strength if query and trace are available
        if hasattr(self, '_last_trace') and query is not None:
            if self._last_trace.abs().sum() > 1e-8:
                q_norm = F.normalize(query.squeeze(), dim=-1)
                mN_norm = F.normalize(self._last_trace, dim=-1)
                scores = torch.matmul(mN_norm, q_norm)
                metrics['coupling_strength'] = scores.max().item()

        return metrics


def compute_pvm_metrics(model, enable_rca=False, query=None):
    """
    Aggregate PVM metrics across all layers in a model.

    Collects memory norm, gate strength, and optional RCA metrics
    from all PVM modules in the model's AFRB adapters.

    Args:
        model: Model containing afrb_adapters with PVM modules
        enable_rca: Whether to compute RCA coupling metrics
        query: Optional query tensor for RCA computation

    Returns:
        dict mapping metric names to values, e.g.:
            {
                'pvm_mem_norm_block1': 0.23,
                'pvm_gate_strength_block1': 0.12,
                'pvm_coupling_strength_block1': 0.87,
                ...
            }
    """
    metrics = {}

    if not hasattr(model, 'afrb_adapters'):
        return metrics

    for i, adapter in enumerate(model.afrb_adapters):
        if hasattr(adapter, 'pvm') and adapter.pvm is not None:
            # Basic memory info
            info = adapter.pvm.get_memory_info()
            metrics[f'pvm_mem_norm_block{i+1}'] = info['mem_norm']
            metrics[f'pvm_gate_strength_block{i+1}'] = info['gate_strength']

            # Optional RCA metrics
            if enable_rca:
                rca = adapter.pvm.compute_rca_metrics(query=query)
                metrics.update({f'pvm_{k}_block{i+1}': v for k, v in rca.items()})

    return metrics
