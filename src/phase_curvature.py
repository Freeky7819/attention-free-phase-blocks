"""
Phase Curvature Diagnostics Module

This module provides curvature-based trajectory analysis for Phase Vector Memory (PVM).
It computes discrete curvature along the PVM trajectory to identify where the model
makes sharp "turns" in phase space, particularly around needle positions.

Mathematical Framework:
----------------------
Given a PVM trajectory z[t] ∈ R^D over time t ∈ [0, T], we compute:
- Velocity: v[t] = z[t+1] - z[t]
- Acceleration: a[t] = v[t+1] - v[t]
- Curvature: κ[t] = ||a[t]||

High curvature indicates the model is making sharp directional changes in phase space,
which often coincides with important tokens (e.g., the needle in a haystack).

Why These Metrics Matter:
-------------------------
1. **Curvature Peak at Needle**: Measures if phase space "curves" around the needle.
   High values indicate the model creates a distinctive phase signature at the needle.

2. **Curvature Before Retrieval**: Checks if curvature builds up before the query.
   This validates that PVM encodes the needle before needing to retrieve it.

3. **Curvature Entropy**: Measures how "focused" the curvature is.
   Low entropy = sharp peak (good). High entropy = diffuse (bad).

4. **Peak Offset from Retrieval**: How far is the curvature peak from the query point?
   Small offset indicates good alignment between encoding and retrieval.

Physical Intuition:
------------------
Think of PVM trajectory as a particle moving through phase space:
- Straight lines = boring tokens (low curvature)
- Sharp turns = important events like the needle (high curvature)
- The model should "curve around" the needle, creating a memorable landmark

DIAGNOSTICS ONLY - No gradient computation or training modifications.
"""

import torch
import math


def compute_phase_curvature_metrics(pvm_traj, needle_pos, retrieval_start, topk=5, extended=False):
    """
    Compute curvature-based diagnostics for PVM along the sequence.

    This function analyzes the PVM trajectory to determine if the model creates
    distinctive phase signatures around important tokens (the needle).

    Mathematical Details:
    --------------------
    1. Velocity: v[t] = z[t+1] - z[t]  (first difference)
    2. Acceleration: a[t] = v[t+1] - v[t]  (second difference)
    3. Curvature: κ[t] = ||a[t]||  (magnitude of acceleration)
    4. Normalized curvature: κ_norm[t] = (κ[t] - min(κ)) / (max(κ) - min(κ))

    Args:
        pvm_traj: Tensor [T, D] or [B, T, D] - PVM states per timestep (no grad needed)
        needle_pos: int or list[int] - Token index where needle is located
        retrieval_start: int - Token index where query "RETURN NEEDLE" starts
        topk: int - Number of closest curvature points to consider (default: 5)
        extended: bool - If True, compute additional extended metrics (default: False)

    Returns:
        dict with scalar metrics:

        Base Metrics (always computed):
        {
            "curv_peak_at_needle": float [0,1] - Local peak / global peak
            "curv_peak_before_needle": float [0,1] - Max curvature before retrieval
            "curv_traj_len": int - Length of trajectory
        }

        Extended Metrics (if extended=True):
        {
            "curv_peak_idx": int - Index of global curvature peak
            "curv_peak_offset_from_retrieval": int - Signed distance (peak - retrieval)
            "curv_peak_at_needle_binary": int (0/1) - 1 if peak within ±4 of needle
            "curv_pre_peak_fraction": float [0,1] - Fraction of curvature before retrieval
            "curv_entropy": float [0,1] - Normalized Shannon entropy of curvature
        }

    Interpretation Guide:
    --------------------
    - curv_peak_at_needle ≈ 1.0: Perfect! Peak curvature at needle position
    - curv_peak_before_needle high: Model encodes needle before retrieval (good)
    - curv_entropy low: Sharp, focused curvature peak (good signal-to-noise)
    - curv_peak_offset_from_retrieval ≈ 0: Good alignment between encode and query
    """
    # 1. Ensure shape [T, D]
    if pvm_traj.dim() == 3:
        # Take first batch for diagnostic analysis
        pvm_traj = pvm_traj[0]

    T, D = pvm_traj.shape

    # Guard for very short trajectories (cannot compute curvature)
    if T < 3:
        return {
            "curv_peak_at_needle": 0.0,
            "curv_peak_before_needle": 0.0,
            "curv_traj_len": int(T),
            "curv_peak_offset_from_retrieval": 0.0,
            "curv_peak_at_needle_binary": 0,
            "curv_pre_peak_fraction": 0.0,
            "curv_entropy": 0.0,
            "curv_peak_idx": -1,
            "curv_peak_dist_from_needle_center": 0.0,
            "curv_peak_dist_from_retrieval": 0.0,
            "curv_soft_success": 0.0,
            "curv_soft_success_weighted": 0.0,
            "curv_margin": 0.0,
            "curv_margin_abs": 0.0,
            "curv_margin_ratio": 0.0,
            "phase_curv_regime": "undefined_short_traj",
        }

    # 2. Compute discrete curvature along trajectory
    # First difference (velocity): v[t] = z[t+1] - z[t]
    v = pvm_traj[1:] - pvm_traj[:-1]  # [T-1, D]

    # Second difference (acceleration): a[t] = v[t+1] - v[t]
    a = v[1:] - v[:-1]  # [T-2, D]

    # Curvature magnitude: κ[t] = ||a[t]||
    curv = torch.norm(a, dim=-1)  # [T-2]

    # Normalize to [0, 1] range
    curv_norm = (curv - curv.min()) / (curv.max() - curv.min() + 1e-8)

    # 3. Map needle_pos into curvature index space
    # Curvature is defined between tokens 1..T-2, so shift accordingly
    needle_idx = int(needle_pos)
    needle_idx = max(1, min(needle_idx, T-2)) - 1  # Align into [0, T-3]

    # 4. Compute base metrics
    window = 4  # Small neighborhood around needle (±4 tokens)
    start = max(0, needle_idx - window)
    end = min(curv_norm.numel(), needle_idx + window + 1)

    # Local peak: maximum curvature near needle
    local_peak = float(curv_norm[start:end].max().item())

    # Global peak: maximum curvature anywhere
    global_peak = float(curv_norm.max().item())

    # Ratio: how much of global peak is captured locally?
    curv_peak_at_needle = local_peak / (global_peak + 1e-8)

    # Did curvature start rising before the retrieval_start?
    rs_idx = max(1, min(int(retrieval_start), T-2)) - 1
    before_region = curv_norm[:rs_idx]
    curv_peak_before = float(before_region.max().item()) if before_region.numel() > 0 else 0.0

    # Base results
    results = {
        "curv_peak_at_needle": curv_peak_at_needle,
        "curv_peak_before_needle": curv_peak_before,
        "curv_traj_len": int(T),
    }

    # 5. Compute extended metrics (only when requested)
    if extended:
        # Edge case: insufficient trajectory for curvature
        if T < 3 or curv.numel() == 0:
            results.update({
                "curv_peak_idx": 0,
                "curv_peak_offset_from_retrieval": 0,
                "curv_peak_at_needle_binary": 0,
                "curv_pre_peak_fraction": 0.0,
                "curv_entropy": 0.0,
            })
        else:
            # 1. Find global peak index in curvature array
            peak_idx = int(torch.argmax(curv).item())

            # 2. Offset from retrieval: signed distance
            rs_idx_safe = max(0, rs_idx)
            offset = peak_idx - rs_idx_safe

            # 3. Binary metric: is peak within ±4 tokens of needle?
            binary = 1 if (needle_idx - 4) <= peak_idx <= (needle_idx + 4) else 0

            # 4. Pre-peak fraction: what fraction of curvature occurs before retrieval?
            curvature_total = curv.sum().item() + 1e-8
            if rs_idx_safe > 0 and rs_idx_safe <= curv.numel():
                curvature_before = curv[:rs_idx_safe].sum().item()
            else:
                curvature_before = 0.0
            pre_peak_fraction = curvature_before / curvature_total

            # 5. Shannon entropy: measures how "focused" the curvature is
            # Low entropy = sharp peak (good). High entropy = diffuse (bad).
            curv_sum = curv.sum().item() + 1e-8
            p = curv / curv_sum  # Probability distribution

            # Shannon entropy: H = -Σ p(i) log(p(i))
            entropy = 0.0
            for p_i in p:
                p_val = p_i.item()
                if p_val > 1e-10:  # Skip near-zero to avoid log(0)
                    entropy -= p_val * math.log(p_val)

            # Normalize by maximum possible entropy: log(N)
            max_entropy = math.log(len(curv)) if len(curv) > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # Add extended metrics to results
            results.update({
                "curv_peak_idx": peak_idx,
                "curv_peak_offset_from_retrieval": offset,
                "curv_peak_at_needle_binary": binary,
                "curv_pre_peak_fraction": float(pre_peak_fraction),
                "curv_entropy": float(normalized_entropy),
            })

    return results


def compute_phase_metrics(pvm_traj, needle_pos, retrieval_start, topk=5, extended=False):
    """
    Alias for compute_phase_curvature_metrics for backward compatibility.

    See compute_phase_curvature_metrics for full documentation.
    """
    return compute_phase_curvature_metrics(pvm_traj, needle_pos, retrieval_start, topk, extended)
