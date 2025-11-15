"""
Adaptive Fourier Resonance Blocks (AFRB): Phase-Based Neural Processing

AFRB replaces traditional attention mechanisms with resonant phase dynamics.
Instead of computing pairwise token similarities (O(n²)), AFRB modulates
hidden states through learned frequency responses (O(d)).

Core Mechanism:
    h_out = h_in + α · sin(ω·t + φ) · f(h_in)

    where:
    - ω: Learnable resonant frequency (adaptive)
    - φ: Phase offset (cascade-dependent)
    - α: Coupling strength (modulates resonance depth)
    - f(h): Learned transformation (MLP or similar)

Key Innovations:
1. Adaptive Omega: Frequencies learned per layer
2. Cascade Architecture: Multi-scale resonance (coarse→fine)
3. Phase Coherence: Kuramoto synchronization for binding
4. O(d) Complexity: No pairwise computations

Physical Inspiration:
    Neural oscillations (alpha/beta/gamma bands) for information routing
    Kuramoto model for collective synchronization

Components:
    - PhaseEmbedding: Sinusoidal phase modulation with adaptive frequency
    - AFRB: Full resonant block with DRC, Stillness, and memory augmentation
    - PVM: Phase-Vector Memory for temporal coherence
    - PLM: Phase-Lattice Memory for spatial coherence

References:
    - Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence
    - Buzsáki, G. (2006). Rhythms of the Brain
    - Fries, P. (2015). Rhythms for Cognition: Communication through Coherence
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from phase_lattice import PhaseLatticeMemory


class PhaseEmbedding(nn.Module):
    """
    Phase embedding with optional learnable omega (Adaptive-Omega).

    Generates sinusoidal embeddings that modulate hidden states:
        phase(t) = ω · log(1+t) + φ
        embedding = [cos(phase), sin(phase)]

    The log-time parameterization ensures temporal stability while preserving
    high-frequency information in early tokens.

    Args:
        dim: Embedding dimension
        omega: Base resonant frequency (default: 6.0, ~1Hz neural oscillation)
        learnable_omega: If True, omega is optimized via backprop
        omega_min: Lower bound for learned omega (stability constraint)
        omega_max: Upper bound for learned omega (prevents aliasing)

    Physical Interpretation:
        - omega ~ 6.0: Alpha-band oscillations (8-12 Hz), spatial attention
        - omega ~ 4.0: Theta-band (4-8 Hz), working memory
        - omega ~ 8.0: Beta-band (12-30 Hz), motor planning
    """

    def __init__(self, dim, omega=6.0, learnable_omega=False, omega_min=5.6, omega_max=6.4):
        super().__init__()
        self.dim = dim

        # Project [cos, sin] → dim-dimensional space
        # Small std ensures phase modulation starts weak (gradual learning)
        self.proj = nn.Linear(2, dim)
        nn.init.normal_(self.proj.weight, std=0.001)
        nn.init.zeros_(self.proj.bias)

        # Adaptive-Omega: learnable frequency parameter
        self.learnable_omega = learnable_omega
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)

        if learnable_omega:
            # Learnable: model optimizes omega via gradient descent
            # Allows each layer to find its optimal temporal scale
            self.omega_param = nn.Parameter(torch.tensor(float(omega)))
        else:
            # Fixed: omega is constant (classical Fourier features)
            self.register_buffer("omega_const", torch.tensor(float(omega)))

    def current_omega(self):
        """
        Get current omega value (clamped if learnable, constant otherwise).

        Clamping prevents:
            - Aliasing (omega too high → temporal ambiguity)
            - Collapse (omega → 0 → constant embedding)
        """
        if self.learnable_omega:
            # Clamp to safe range [omega_min, omega_max]
            return torch.clamp(self.omega_param, self.omega_min, self.omega_max)
        return self.omega_const

    def forward(self, x, alpha=0.04, phi=0.0):
        """
        Apply phase embedding to input.

        Args:
            x: Input tensor [B, T, D]
            alpha: Blend factor (0 = passthrough, 1 = full phase modulation)
            phi: Phase offset (shifts temporal binding window)

        Returns:
            Modulated tensor [B, T, D]

        Mechanism:
            1. Generate log-time coordinate: t_log = log(1+t)
            2. Compute phase: ω·t_log + φ
            3. Project [cos(phase), sin(phase)] to D dimensions
            4. Blend with input: (1-α)·x + α·phase_emb
        """
        B, T, D = x.shape
        omega = self.current_omega()  # Get current (possibly learned) omega

        # Match dtype with input x (e.g., bfloat16, float32)
        t = torch.arange(T, device=x.device, dtype=x.dtype)

        # Log-scale time for stability: emphasizes early tokens, compresses late
        # Prevents phase wrap-around in long sequences
        t_log = torch.log1p(t)

        # Compute phase: ω·t + φ
        phase = omega * t_log.unsqueeze(-1) + phi

        # Sinusoidal encoding: [cos, sin] preserves phase information
        cos_sin = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)  # [T, 2]

        # Project to model dimension
        phase_emb = self.proj(cos_sin).unsqueeze(0).expand(B, -1, -1)  # [B, T, D]

        # Blend with input (alpha controls resonance strength)
        return (1 - alpha) * x + alpha * phase_emb


class AFRB(nn.Module):
    """
    Attention-Free Resonant Block: Phase-based alternative to Transformer layers.

    Architecture:
        h = x + norm(mixer(x) + γ·phase(x))

    where:
        - mixer: GLU or depthwise convolution (local feature extraction)
        - phase: Sinusoidal modulation (temporal binding)
        - γ: Learnable resonance depth (controls phase influence)

    Key Features:
        1. DRC (Dynamic Resonance Control): Per-block phase offsets for cascade
        2. Stillness: EMA low-pass filtering for training stability
        3. Stagger: Progressive block activation (prevents early collapse)
        4. Adaptive-Omega: Layer-specific resonant frequencies
        5. PVM (Phase-Vector Memory): Temporal coherence across tokens
        6. PLM (Phase-Lattice Memory): Spatial coherence grid
        7. T2 (Tonal Tempering): Gradual frequency annealing
        8. PCM (Phase Coherence Modulation): Kuramoto synchronization

    Cascade Architecture:
        Layer 1: ω₁=6.0, φ₁=0.0    (coarse rhythm, low-frequency binding)
        Layer 2: ω₂=6.1, φ₂=0.1    (intermediate, phase-shifted)
        Layer 3: ω₃=6.2, φ₃=0.2    (fine-grained, high-frequency details)

        Each layer processes at a different temporal scale, creating
        multi-resolution temporal hierarchies.

    Args:
        dim: Hidden dimension
        block_idx: Layer index (used for stagger scheduling)
        alpha_base: Phase blend strength (0.04 = 4% phase modulation)
        gamma_base: Resonance depth (0.20 = 20% phase influence on output)
        omega: Base resonant frequency (6.0 ≈ alpha-band)
        phi: Base phase offset (0.0 = in-phase with time zero)
        use_glu: If True, use GLU mixer; else depthwise conv
        stillness_ema: EMA coefficient for phase filtering (0.0 = disabled)
        stillness_floor: Minimum gain after filtering (prevents collapse)
        stillness_warm: Warmup steps before stillness activates
        block_warm_delta: Per-block warmup offset (stagger control)
        block_ramp: Smooth activation duration (steps)
        phase_ramp: Phase glide duration (steps)
        phase_delta: Target phase shift (for DRC cascade)
        learnable_omega: Enable adaptive frequency learning
        omega_min/max: Bounds for learned omega
        use_pvm: Enable Phase-Vector Memory
        pvm_alpha/beta: PVM blend coefficients
        pvm_gate_init: Initial gating strength
        use_plm: Enable Phase-Lattice Memory
        plm_grid_x/y: Lattice dimensions
        plm_alpha/beta: PLM blend coefficients
        plm_omega/kappa: Lattice resonance parameters
        plm_gate_init: Initial gating strength
        t2_enable: Enable Tonal Tempering (frequency annealing)
        t2_steps: Annealing duration
        t2_mode: Annealing schedule ('exp', 'linear', 'cosine')
        t2_k: Annealing rate
        t2_alpha: Target blend strength
        t2_omega: Target frequency
        t2_phi: Target phase offset
        pcm_enable: Enable Phase Coherence Modulation (Kuramoto)
        pcm_gate_init: Initial coherence gating
    """

    def __init__(self, dim, block_idx=1, alpha_base=0.04, gamma_base=0.20, omega=6.0, phi=0.0, use_glu=True,
                 stillness_ema=0.0, stillness_floor=0.0, stillness_warm=0,
                 block_warm_delta=0, block_ramp=300, phase_ramp=300, phase_delta=0.0,
                 learnable_omega=False, omega_min=5.6, omega_max=6.4,
                 use_pvm=False, pvm_alpha=0.12, pvm_beta=0.88, pvm_gate_init=-2.0,
                 use_plm=False, plm_grid_x=4, plm_grid_y=4, plm_alpha=0.10, plm_beta=0.90,
                 plm_omega=6.0, plm_kappa=0.05, plm_gate_init=-2.0,
                 t2_enable=False, t2_steps=1500, t2_mode='exp', t2_k=0.001, t2_alpha=0.08,
                 t2_omega=6.0, t2_phi=1.0472, pcm_enable=False, pcm_gate_init=0.5):
        super().__init__()
        self.block_idx = block_idx

        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Mixer: Local feature extraction
        # GLU: Gated linear unit (learnable gating)
        # DepthwiseConv: Efficient local context (kernel=3)
        if use_glu:
            self.mixer = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GLU(dim=-1)  # Split last dimension (D), not sequence (T)
            )
        else:
            self.mixer = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)

        # DRC parameters: Control phase dynamics
        # alpha: Phase modulation strength (how much sine wave affects embedding)
        # phi_base: Base phase offset (layer-specific temporal alignment)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_base)), requires_grad=False)
        self.phi_base = nn.Parameter(torch.tensor(float(phi)), requires_grad=False)

        # Gamma: Resonance depth (controls overall phase influence on residual path)
        # Stored as logit for numerical stability: gamma = sigmoid(gamma_raw)
        # Learnable: allows model to modulate resonance strength
        g0 = float(gamma_base)
        self.gamma_raw = nn.Parameter(torch.tensor(math.log(g0 / (1.0 - g0))))

        # Phase embedding with optional Adaptive-Omega
        self.phase_embed = PhaseEmbedding(
            dim=dim,
            omega=omega,
            learnable_omega=learnable_omega,
            omega_min=omega_min,
            omega_max=omega_max
        )
        self.use_glu = use_glu

        # Stillness: EMA low-pass filter on phase term
        # Physical analogy: Membrane capacitance in neurons (dampens rapid oscillations)
        # Benefits: Stabilizes early training, prevents phase noise amplification
        self.rho = float(stillness_ema)       # EMA coefficient (ρ ∈ [0,1])
        self.floor = float(stillness_floor)   # Minimum gain (prevents over-damping)
        self.register_buffer("phase_ema", None)  # Lazy init on first forward
        self.register_buffer("global_step", torch.tensor(0))  # Step counter

        # Stagger: Per-block activation delay + smooth ramp
        # Prevents early training collapse by activating layers progressively
        # Layer 1: active from step 0
        # Layer 2: active from step block_warm_delta
        # Layer 3: active from step 2*block_warm_delta, etc.
        self.warm_i = stillness_warm + (block_idx - 1) * block_warm_delta  # Per-block warmup
        self.ramp = block_ramp          # Smooth activation ramp duration
        self.phase_ramp = phase_ramp    # Phase glide duration
        self.phase_delta = phase_delta  # Target phase offset (from DRC)

        # Phase-Vector Memory (PVM): Temporal coherence across tokens
        # Maintains exponentially-weighted history of phase states
        # Query-based readout for context-dependent retrieval
        self.use_pvm = use_pvm
        if self.use_pvm:
            from phase_memory import PhaseVectorMemory
            self.pvm = PhaseVectorMemory(
                hidden_size=dim,
                alpha=pvm_alpha,
                beta=pvm_beta,
                omega=omega,  # Use same omega as phase embedding (coherence)
                phi_base=phi,
                gate_init=pvm_gate_init,
                history_size=64,  # Sliding window for query-addressed readout
                t2_enable=t2_enable,
                t2_steps=t2_steps,
                t2_mode=t2_mode,
                t2_k=t2_k,
                t2_alpha=t2_alpha,
                t2_omega=t2_omega,
                t2_phi=t2_phi,
                pcm_enable=pcm_enable,
                pcm_gate_init=pcm_gate_init
            )
        else:
            self.pvm = None

        # Phase Lattice Memory (PLM): Spatial coherence grid
        # 2D lattice of phase oscillators (Kuramoto model)
        # Enables spatial binding via phase synchronization
        self.use_plm = use_plm
        if self.use_plm:
            self.plm = PhaseLatticeMemory(
                model_dim=dim,
                grid=(plm_grid_x, plm_grid_y),
                alpha=plm_alpha,
                beta=plm_beta,
                omega=plm_omega,
                kappa=plm_kappa,  # Coupling strength (Kuramoto parameter)
                gate_init=plm_gate_init,
                periodic=True,  # Toroidal topology (no boundaries)
                t2_enable=t2_enable,
                t2_steps=t2_steps,
                t2_mode=t2_mode,
                t2_k=t2_k,
                t2_alpha=t2_alpha,
                t2_omega=t2_omega,
                t2_phi=t2_phi,
                pcm_enable=pcm_enable,
                pcm_gate_init=pcm_gate_init
            )
        else:
            self.plm = None

    def get_gamma(self):
        """
        Compute resonance depth via sigmoid transformation.

        Returns gamma ∈ (0,1) controlling phase influence:
            - gamma → 0: Minimal resonance (attention-like, data-driven)
            - gamma → 1: Full resonance (oscillator-like, phase-driven)

        Typical values: 0.15-0.25 (balanced phase-data coupling)
        """
        return torch.sigmoid(self.gamma_raw)

    def forward(self, x):
        """
        Forward pass: Apply resonant processing.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Output tensor [B, T, D]

        Processing Pipeline:
            1. Normalize input
            2. Apply mixer (GLU or conv)
            3. Augment with PVM (if enabled)
            4. Augment with PLM (if enabled)
            5. Compute phase embedding with stagger/glide
            6. Apply stillness filtering (training only)
            7. Blend via residual connection

        Temporal Dynamics:
            - Early training: Low gamma, weak phase (mixer-dominated)
            - Mid training: Rising gamma, increasing resonance
            - Late training: Stable gamma, full phase dynamics
        """
        # Increment global step counter during training
        if self.training:
            self.global_step += 1

        step = int(self.global_step.item())  # Current global step

        residual = x
        x = self.norm1(x)

        # Mixer: Local feature extraction
        if self.use_glu:
            mixed = self.mixer(x)
        else:
            # Conv1d expects [B, D, T], so transpose
            mixed = self.mixer(x.transpose(1, 2)).transpose(1, 2)

        # Phase-Vector Memory: Temporal coherence
        # Blends current state with phase-weighted history
        if self.use_pvm:
            mixed = self.pvm(mixed, step_idx=step, training=self.training)

        # Phase Lattice Memory: Spatial coherence
        # Projects onto 2D lattice, applies Kuramoto coupling, projects back
        if self.use_plm:
            mixed = self.plm(mixed, step_idx=step, training=self.training)

        # Stagger: Smooth activation ramp (0→1) for this block
        # Prevents early collapse by delaying layer contribution
        if step < self.warm_i:
            ramp_factor = 0.0  # Block not yet active
        else:
            # Linear ramp from 0 to 1 over 'ramp' steps
            ramp_factor = min(1.0, float(step - self.warm_i) / max(1, self.ramp))

        # Stagger: Phase glide (exponential shift from phi_base → phi_base + delta_phi)
        # Creates smooth phase transition for cascade alignment
        if step >= self.warm_i and self.phase_delta != 0.0:
            # Exponential approach: phi(t) = phi0 + Δφ*(1 - exp(-t/τ))
            # Prevents phase discontinuities during warmup
            progress = float(step - self.warm_i) / max(1, self.phase_ramp)
            phi_eff = self.phi_base.item() + self.phase_delta * (1.0 - math.exp(-progress))
        else:
            phi_eff = self.phi_base.item()

        # Compute phase embedding with glided phi
        # Omega is now internal to phase_embed (adaptive or fixed)
        phase = self.phase_embed(x, alpha=self.alpha.item(), phi=phi_eff)

        # Stillness: EMA low-pass filter (training stabilizer)
        # Analogous to dropout/batch norm: smooths phase gradients during learning
        # Disabled during eval (inference uses full dynamics)
        if self.rho > 0.0 and self.training and step >= self.warm_i:
            # Lazy init or reinit on shape change (safety for batch size changes)
            if (self.phase_ema is None) or (self.phase_ema.shape != phase.shape):
                self.phase_ema = phase.detach().clone()

            # EMA update: ema_new = (1-ρ)·ema_old + ρ·phase_current
            # Low ρ (e.g., 0.1): Strong filtering, slow adaptation
            # High ρ (e.g., 0.5): Weak filtering, fast adaptation
            self.phase_ema = (1.0 - self.rho) * self.phase_ema + self.rho * phase.detach()

            # Apply filtered phase with soft gain (floor protection)
            # Gain prevents over-damping (ensures some phase signal remains)
            gain = max(self.floor, 1.0 - self.rho)  # e.g., 0.9 for ρ=0.1
            phase = gain * self.phase_ema + (1.0 - gain) * phase

        # Apply resonance with ramp factor (smooth activation)
        # Residual connection preserves gradient flow
        gamma = self.get_gamma()
        out = residual + self.norm2(mixed + (gamma * ramp_factor) * phase)

        return out
