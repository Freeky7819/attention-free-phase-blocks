"""
Ridge Regression and Query Computation for Phase-Space Alignment

This module provides utilities for aligning Phase Vector Memory (PVM) readouts with
target embedding spaces using Ridge regression (L2-regularized least squares).

Core Mathematical Framework:
----------------------------
Given PVM readouts Z ∈ R^(N×Dp) and target embeddings E ∈ R^(N×De), we solve for
the transformation matrix W ∈ R^(Dp×De) that minimizes:

    ||ZW - E||² + λ||W||²

The closed-form solution is:
    W = (Z^T Z + λI)^(-1) Z^T E

This "calibrates" PVM readouts to produce logits directly, bypassing the need for
attention mechanisms.

Query Computation:
-----------------
The query vector q ∈ R^D is computed from recent hidden states and controls what
information is retrieved from PVM. Multiple strategies are supported:
- Mean: Simple average of recent tokens (baseline)
- EMA: Exponential moving average (recent tokens weighted more)
- Phase-weighted: Weight tokens by local phase velocity (where Δhidden is largest)

Physical Intuition:
------------------
PVM curvature acts like "gravitational lensing" around important events (the needle).
Phase-weighted queries follow this curvature by focusing on tokens where phase
changes rapidly (large ||Δhidden||).
"""

import torch
import torch.nn.functional as F


def compute_query_key(hidden_seq, window=16, mode="mean", temperature=0.5):
    """
    Compute query key from hidden states for PVM retrieval.

    This function extracts a query vector from the last N tokens of hidden states,
    which determines what information is retrieved from Phase Vector Memory.

    Args:
        hidden_seq: Tensor [B, T, D] - Hidden states from transformer layers
        window: int - Number of recent tokens to use for query (default: 16)
        mode: str - Query computation strategy (default: "mean")
            - "mean": Simple average across batch and time
            - "ema": Exponential moving average (recent tokens weighted more)
            - "phase_weighted": Weight by local phase velocity (||Δhidden||)
            - "mlp_query": Non-linear transformation via small MLP
        temperature: float - Softmax temperature for phase_weighted mode (default: 0.5)
            Lower values = sharper focus on high-velocity regions

    Returns:
        query: Tensor [D] or [B, D] - Normalized query vector(s)

    Mathematical Details:
    --------------------
    For phase_weighted mode:
    1. Compute phase velocity: δ[t] = hidden[t+1] - hidden[t]
    2. Compute magnitude: speed[t] = ||δ[t]||
    3. Apply softmax weighting: w[t] = softmax(speed / temperature)
    4. Weighted average: q = Σ(w[t] * hidden[t])

    Physical Analogy:
    ----------------
    Where ||Δhidden|| is largest, phase is changing rapidly. This signals an
    "important event" (e.g., the needle). The query should focus here, following
    the curvature of phase space.
    """
    # Take last window tokens from sequence
    if hidden_seq.size(1) < window:
        H = hidden_seq[:, :, :]  # Use all available
    else:
        H = hidden_seq[:, -window:, :]  # Last N tokens

    # Compute query based on mode
    if mode in ["mean", "mean_tail"]:
        # Baseline: simple mean across batch and time
        q = H.mean(dim=(0, 1))  # [D]

    elif mode == "ema":
        # EMA weighting: recent tokens weighted more
        T = H.size(1)
        w = torch.linspace(0.1, 1.0, steps=T, device=H.device)
        w = w / w.sum()
        q = (H[0] * w.unsqueeze(-1)).sum(dim=0)  # [D] (use first batch item)

    elif mode == "phase_weighted":
        """
        Phase-weighted query: weight tokens by local phase velocity.

        Intuition: Where ||Δhidden|| is largest, phase is changing rapidly.
        This signals an "important event" - that's where query should look.

        Physical analogy: PVM curvature is like gravitational lensing -
        it curves around the needle. Query should follow this curvature.
        """
        # Handle edge case: window too small
        if H.size(1) <= 1:
            # Fall back to last token if window=1
            q = H[:, -1, :].mean(dim=0)  # [D]
        else:
            # 1. Compute local phase velocity (Δhidden) for each batch item
            delta = H[:, 1:, :] - H[:, :-1, :]  # [B, window-1, D]

            # 2. Compute magnitude of phase change (speed)
            phase_speed = torch.norm(delta, dim=-1)  # [B, window-1]

            # 3. Handle all-zero case (no phase change)
            if phase_speed.abs().sum() < 1e-8:
                # Degenerate to uniform mean
                q = H[:, :-1, :].mean(dim=(0, 1))  # [D]
            else:
                # 4. Softmax weights (temperature controls sharpness)
                # Higher phase speed = higher weight
                weights = F.softmax(phase_speed / temperature, dim=-1)  # [B, window-1]

                # 5. Weighted mean (apply to tokens BEFORE delta for alignment)
                # We weight hidden[t] by speed of change from t to t+1
                tokens_to_weight = H[:, :-1, :]  # [B, window-1, D]

                # Weighted sum per batch item
                weighted = (tokens_to_weight * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

                # Average across batch
                q = weighted.mean(dim=0)  # [D]

    elif mode == "mlp_query":
        """
        MLP query: Small MLP over tail tokens.

        Architecture: Linear(D, D/4) -> ReLU -> Linear(D/4, D)
        This learns a non-linear transformation of tail tokens.

        Note: Uses deterministic pseudo-random projection (no learnable parameters).
        Expected improvement: Better curvature peak alignment with needle position.
        """
        # Flatten batch and time dimensions
        B, T, D = H.shape
        H_flat = H.reshape(B * T, D)  # [B*T, D]

        # Generate deterministic projection matrices using fixed seed
        # This ensures consistency across training steps
        device = H.device
        dtype = H.dtype
        torch.manual_seed(42)  # Fixed seed for reproducibility
        W1 = torch.randn(D, D // 4, device=device, dtype=dtype) * 0.01
        W2 = torch.randn(D // 4, D, device=device, dtype=dtype) * 0.01

        # Forward pass
        h1 = F.relu(H_flat @ W1)  # [B*T, D/4]
        h2 = H_flat @ W2  # [B*T, D]

        # Average across batch and time
        q = h2.mean(dim=0)  # [D]

    else:
        # Unknown mode: fall back to mean
        q = H.mean(dim=(0, 1))

    # Normalize (preserve dtype to match model precision)
    return F.normalize(q, dim=-1).to(hidden_seq.dtype)


@torch.no_grad()
def ridge_fit(Z: torch.Tensor, E: torch.Tensor, l2: float = 1e-3) -> torch.Tensor:
    """
    Ridge regression: solve for W such that ZW ≈ E.

    This implements L2-regularized least squares to find the optimal linear
    transformation from PVM readout space to embedding space.

    Mathematical Formulation:
    ------------------------
    Minimize: ||ZW - E||² + λ||W||²

    Closed-form solution:
        W = (Z^T Z + λI)^(-1) Z^T E

    Where:
    - Z ∈ R^(N×Dp): Source vectors (PVM readouts)
    - E ∈ R^(N×De): Target vectors (embedding space)
    - λ: Regularization strength (prevents overfitting)
    - I: Identity matrix

    Args:
        Z: Tensor [N, Dp] - Source space vectors (e.g., PVM readouts)
        E: Tensor [N, De] - Target space vectors (e.g., needle embeddings)
        l2: float - L2 regularization factor (default: 1e-3)

    Returns:
        W: Tensor [Dp, De] - Transformation matrix such that Z @ W ≈ E

    Implementation Notes:
    --------------------
    - Computation performed in float32 for numerical stability
    - Uses torch.linalg.solve (more stable than matrix inversion)
    - Result converted back to original dtype to match model precision
    """
    # Convert to float32 for numerical stability
    original_dtype = Z.dtype
    Z_f32 = Z.to(torch.float32)
    E_f32 = E.to(torch.float32)

    Dp = Z_f32.shape[1]

    # Regularization term: λI
    I = torch.eye(Dp, device=Z_f32.device, dtype=torch.float32)

    # Compute Gram matrix and cross-correlation
    ZtZ = Z_f32.transpose(0, 1) @ Z_f32  # [Dp, Dp]
    ZtE = Z_f32.transpose(0, 1) @ E_f32  # [Dp, De]

    # Solve (Z^T Z + λI) W = Z^T E
    W_f32 = torch.linalg.solve(ZtZ + l2 * I, ZtE)

    # Convert back to original dtype
    return W_f32.to(original_dtype)


@torch.no_grad()
def collect_alignment_pairs(dataloader, model, lm_embed_table, max_pairs=512):
    """
    Collect (Z, E) pairs for ridge calibration from training data.

    This function runs the model on a dataloader to collect:
    - Z: PVM readouts (what the model currently retrieves)
    - E: Target embeddings (what we want the model to output)

    These pairs are used to calibrate the ridge regression matrix W such that
    Z @ W ≈ E, allowing PVM readouts to produce logits directly.

    Pipeline:
    --------
    1. Run model forward pass to populate PVM memory and hidden states
    2. Compute per-sample queries from hidden states (using compute_query_key)
    3. Retrieve PVM readouts using these queries
    4. Extract target embeddings from needle tokens or labels
    5. Normalize both Z and E for stability
    6. Return paired datasets for ridge fitting

    Args:
        dataloader: DataLoader - Training data iterator
        model: nn.Module - Model with AFRB adapters containing PVM
        lm_embed_table: Tensor [vocab_size, embed_dim] - Token embedding table
        max_pairs: int - Maximum number of (Z, E) pairs to collect (default: 512)

    Returns:
        tuple: (Z, E) where:
            - Z: Tensor [N, Dp] - PVM readouts (normalized)
            - E: Tensor [N, De] - Target embeddings (normalized)

    Implementation Notes:
    --------------------
    - Uses PER-SAMPLE queries (one query per batch item) for correct alignment
    - Model must be in train() mode to populate PVM memory
    - Requires model to have 'afrb_adapters' with PVM instances
    - Normalizes both Z and E before returning (improves ridge conditioning)

    Alignment Fix (2025-11-13):
    ---------------------------
    Previously used a single global query per batch, causing misalignment between
    calibration and runtime. Now uses per-sample queries to match runtime behavior.
    """
    Z_list, E_list = [], []
    n = 0

    # Switch to train mode so PVM memory can be populated
    model.train()
    model_dtype = next(model.parameters()).dtype

    print(f"[RIDGE-CALIB] Collecting alignment pairs (max={max_pairs})...")
    print("[RIDGE-CALIB] Using PER-SAMPLE queries for correct alignment")
    print("[RIDGE-CALIB] Each sample gets its own query to match runtime behavior")

    for batch in dataloader:
        if n >= max_pairs:
            break

        # Move batch to model device and dtype
        inputs = {}
        for k, v in batch.items():
            if hasattr(v, "to"):
                if v.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    inputs[k] = v.to(device=model.device, dtype=model_dtype)
                else:
                    inputs[k] = v.to(model.device)
            else:
                inputs[k] = v

        # 1. Run model forward pass (populates model._last_hidden_states)
        outputs = model(**inputs, output_hidden_states=True)

        # 2. Extract hidden states (same as in apply_readout_to_logits)
        if not hasattr(model, '_last_hidden_states'):
            print("[RIDGE-CALIB] ERROR: model._last_hidden_states not found. Skipping batch.")
            continue

        hidden = model._last_hidden_states  # [B, T, D]

        # 3. Compute per-sample queries
        B_batch = hidden.shape[0]
        queries = []
        for b in range(B_batch):
            q = compute_query_key(
                hidden[b:b+1, :, :],  # Per-sample: [1, T, D]
                window=16,
                mode="mean"
            )  # [D]
            queries.append(q)

        queries_batch = torch.stack(queries, dim=0)  # [B, D]

        # 4. Per-sample PVM readout
        readouts_per_sample = []
        if hasattr(model, 'afrb_adapters') and model.afrb_adapters is not None:
            for adapter in model.afrb_adapters:
                if hasattr(adapter, 'pvm') and adapter.pvm is not None:
                    adapter_reads = []
                    for b in range(B_batch):
                        query = F.normalize(queries_batch[b], dim=-1)  # [D]
                        pvm_read = adapter.pvm.readout(query, topk=0)  # [D]
                        adapter_reads.append(pvm_read)

                    if adapter_reads:
                        adapter_reads = torch.stack(adapter_reads, dim=0)  # [B, D]
                        readouts_per_sample.append(adapter_reads)

        if not readouts_per_sample:
            print("[RIDGE-CALIB] ERROR: No PVM memory found in adapters. Skipping batch.")
            continue

        # 5. Average across adapters, keep per-sample dimension
        z_pvm_batch = torch.stack(readouts_per_sample, dim=0).mean(dim=0)  # [B, D]

        # 6. Get target embeddings (per-sample)
        if 'needle' in inputs and isinstance(inputs['needle'], torch.Tensor):
            target_ids = inputs['needle'][:, 0]  # [B] - first token of needle
        elif 'labels' in inputs:
            # Find first non-ignore token per sample
            labels = inputs['labels']
            target_ids = []
            for b in range(B_batch):
                valid_labels = labels[b][labels[b] != -100]
                if len(valid_labels) > 0:
                    target_ids.append(valid_labels[-1])  # Last valid token
                else:
                    target_ids.append(torch.tensor(0, device=labels.device))
            target_ids = torch.stack(target_ids)  # [B]
        else:
            print("[RIDGE-CALIB] WARNING: No 'needle' or 'labels' found. Skipping batch.")
            continue

        E_batch = lm_embed_table[target_ids]  # [B, De]

        # 7. Collect pairs (per-sample)
        Z_list.append(z_pvm_batch.detach())
        E_list.append(E_batch.detach())
        n += z_pvm_batch.shape[0]

    if len(Z_list) == 0:
        raise ValueError("[RIDGE-CALIB] No pairs collected! Check dataloader and PVM logic.")

    Z = torch.cat(Z_list, dim=0)[:max_pairs]
    E = torch.cat(E_list, dim=0)[:max_pairs]

    # Normalize Z and E before ridge solve (improves conditioning)
    Z = F.normalize(Z.float(), dim=1)
    E = F.normalize(E.float(), dim=1)

    # Return to train mode
    model.train()

    print(f"[RIDGE-CALIB] Collection complete. Z={Z.shape}, E={E.shape}")

    return Z, E
