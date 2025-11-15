"""
Resonant Adapter Training System - Phase-Based Neural Architecture

This module implements training infrastructure for Resonant Models, a novel
neural architecture that replaces traditional attention mechanisms with
phase dynamics and resonant coupling.

=== ARCHITECTURE OVERVIEW ===

Traditional Transformers compute attention(Q, K, V) via softmax over all positions,
resulting in O(N²) complexity. Resonant Models instead use phase coupling with
resonant frequencies, achieving O(1) memory access while maintaining interpretability
through geometric phase structure and coherence metrics.

Core Innovation:
    Attention(Q,K,V) → Resonance(frequency, amplitude, phase)

=== KEY COMPONENTS ===

1. Adaptive Frequency-Resonant Blocks (AFRB)
   - Learnable resonant coupling layers that replace attention
   - Adaptive omega parameters for frequency tuning
   - Phase-based information routing

2. Phase-Vector Memory (PVM)
   - Holographic memory storage using phase encoding
   - Pattern storage and retrieval via phase coherence
   - Enables constant-time memory access

3. Phase Lattice Memory (PLM)
   - Geometric phase encoding on lattice structure
   - Spatial organization of phase patterns
   - Curvature-based similarity metrics

4. Query-Addressed Readout
   - Phase-coherence-based retrieval mechanism
   - Learned query head with phase offset correction
   - Window-based and top-k readout strategies

5. LearnedQueryHead
   - Trainable MLP for query extraction from hidden states
   - Corrects phase offsets (e.g., -33 token delay)
   - Bottleneck architecture forces feature extraction

=== TRAINING FEATURES ===

- Needle-in-haystack task: Evaluates long-range memory capabilities
- InfoNCE contrastive loss: Aligns phase and embedding spaces
- Ridge regression: KISS-principle alignment bridge (PVM→Embedding)
- Adaptive omega learning: Fine-tunes resonant frequencies
- Pointer-generator mixing: Learn to copy vs generate

=== EXPERIMENTAL TASKS ===

1. Needle Retrieval (Synthetic)
   - Test long-range memory: retrieve specific pattern from context
   - Measure: needle hit rate, phase coherence, retrieval accuracy

2. Language Modeling (Standard)
   - Evaluate on standard LM benchmarks
   - Compare perplexity with traditional attention models

=== METRICS & DIAGNOSTICS ===

Phase Metrics:
    - Phase coherence: Measures resonance alignment quality
    - Gamma saturation: Tracks amplitude stability
    - Entropy flow: Information propagation through layers

Memory Metrics:
    - PVM metrics: Pattern storage quality
    - PLM metrics: Lattice structure coherence
    - Phase curvature: Geometric similarity measures

Pointer Quality:
    - Recall: Fraction of needle tokens in top-K
    - Precision: Fraction of top-K that are needle tokens
    - F1: Harmonic mean of recall and precision

=== USAGE ===

Basic training:
    python train.py --model pythia-160m --task needle --n_afrb 8

With phase readout:
    python train.py --model pythia-160m --task needle --n_afrb 8 \
                    --readout-from pvm --readout-scale 0.1

With InfoNCE alignment:
    python train.py --model pythia-160m --task needle --n_afrb 8 \
                    --infonce-weight 0.1 --infonce-negatives 128

=== REFERENCES ===

Phase-Vector Memory: Holographic pattern storage in neural networks
Adaptive Frequency-Resonant Blocks: Learnable resonant coupling
InfoNCE: Representation learning via noise-contrastive estimation
Ridge Regression Alignment: KISS-principle phase-embedding bridge

Author: Resonant Model Development Team
License: Apache 2.0
"""
import os, sys, json, csv, argparse, math, random, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from transformers.trainer_callback import ProgressCallback
from torch.optim import AdamW
from datasets import load_dataset, Dataset
from resonant_blocks import AFRB
from utils_phase import phase_coherence, gamma_saturation, entropy_flow
from phase_memory import compute_pvm_metrics
from phase_lattice import compute_plm_metrics
from phase_curvature import compute_phase_curvature_metrics
from utils_alignment import ridge_fit, collect_alignment_pairs, compute_query_key

# Force tqdm to use single-line horizontal mode
os.environ['TQDM_POSITION'] = '0'
os.environ['TQDM_MININTERVAL'] = '1'

# Import and configure tqdm
import tqdm
tqdm.tqdm.monitor_interval = 0

# Suppress TF32 deprecation warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*TF32.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*float32_matmul_precision.*')

# Determinism + Performance (PyTorch 2.2+)
torch.set_float32_matmul_precision("high")  # Use TensorCore
torch.backends.cudnn.benchmark = False      # Deterministic
torch.backends.cudnn.deterministic = True   # Reproducible

# ============================================================
# LEARNED QUERY HEAD (Phase 10: Trainable MLP for Phase Offset Correction)
# ============================================================

class LearnedQueryHead(nn.Module):
    """
    Learns to extract optimal query from hidden states.
    Trainable MLP to correct phase offsets (e.g., -33 token delay).
    """
    def __init__(self, dim, window=16):
        super().__init__()
        self.window = window
        # Bottleneck structure to force feature extraction
        hidden_dim = dim // 4
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, hidden_seq):
        # hidden_seq: [B, T, D]
        # Take last N tokens
        if hidden_seq.size(1) < self.window:
            H = hidden_seq
        else:
            H = hidden_seq[:, -self.window:, :]

        # Mean pool over window (diffuse attention)
        pooled = H.mean(dim=1)  # [B, D]

        # Transform via MLP (Learns to rotate/shift the query)
        query = self.mlp(pooled)  # [B, D]

        # Normalize
        return torch.nn.functional.normalize(query, dim=-1)

# ============================================================
# CUSTOM DATA COLLATOR (for needle task with retrieval_start)
# ============================================================

class NeedleDataCollator:
    """
    Custom collator that preserves retrieval_start field from needle dataset.

    Default HuggingFace collators ignore custom fields, which breaks
    query-addressed readout functionality that depends on retrieval_start.
    """
    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm

    def __call__(self, examples):
        # DEBUG: Check if collator is being called
        if not hasattr(self, '_first_call_done'):
            self._first_call_done = True
            print(f"[COLLATOR-DEBUG] First call! Num examples: {len(examples)}", flush=True)
            if len(examples) > 0:
                print(f"[COLLATOR-DEBUG] First example type: {type(examples[0])}", flush=True)
                print(f"[COLLATOR-DEBUG] First example keys: {list(examples[0].keys()) if isinstance(examples[0], dict) else 'NOT A DICT'}", flush=True)
                if isinstance(examples[0], dict) and 'retrieval_start' in examples[0]:
                    print(f"[COLLATOR-DEBUG] retrieval_start present: {examples[0]['retrieval_start']}", flush=True)
                if isinstance(examples[0], dict) and 'needle' in examples[0]:
                    print(f"[COLLATOR-DEBUG] needle present: {examples[0]['needle']}", flush=True)

        # Collate input_ids and labels (standard padding)
        input_ids = [ex['input_ids'] for ex in examples]
        labels = [ex['labels'] for ex in examples]

        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(lbls) for lbls in labels],
            batch_first=True,
            padding_value=-100
        )

        batch = {
            'input_ids': input_ids,
            'labels': labels,
        }

        # ★ FIX: Preserve ALL needle fields (critical for DELTA-TOP5 diagnostics and hard-pointer)
        def grab_field(key, cast_long=True):
            """Extract field from all examples if present"""
            if all(key in ex for ex in examples):
                vals = [ex[key] for ex in examples]
                # Handle both single values and lists/tensors
                if isinstance(vals[0], (list, tuple)):
                    # Pad sequences if needed
                    max_len = max(len(v) if isinstance(v, (list, tuple)) else 1 for v in vals)
                    padded = []
                    for v in vals:
                        if isinstance(v, (list, tuple)):
                            padded.append(list(v) + [0] * (max_len - len(v)))
                        else:
                            padded.append([v] + [0] * (max_len - 1))
                    t = torch.tensor(padded)
                else:
                    t = torch.tensor(vals)
                return t.long() if cast_long else t
            return None

        # Preserve all needle-related fields
        retrieval_start = grab_field('retrieval_start')
        retrieval_end = grab_field('retrieval_end')
        needle_pos = grab_field('needle_pos')
        # Fix: explicit None check instead of 'or' (which fails with tensors)
        needle_ids = grab_field('needle')
        if needle_ids is None:
            needle_ids = grab_field('needle_token_ids')

        if retrieval_start is not None:
            batch['retrieval_start'] = retrieval_start
        if retrieval_end is not None:
            batch['retrieval_end'] = retrieval_end
        if needle_pos is not None:
            batch['needle_pos'] = needle_pos
        if needle_ids is not None:
            batch['needle'] = needle_ids  # Use 'needle' as canonical name

        return batch

# ============================================================
# READOUT HELPERS (Phase 9 - Query-addressed readout)
# ============================================================

# FORENSIC PATCH B: compute_query_key unified - now imported from utils_alignment.py (line 17)
# Duplicate definition removed to maintain single source of truth

def apply_readout_to_logits(
    model,
    lm_logits,
    last_hidden,
    input_ids=None,
    retrieval_start=None,
    readout_window=None,
    scale: float = 1.0,
    topk: int = None,
    head: str = 'shared'
):
    """
    Shared helper to apply PVM/PLM readout to LM logits.

    READOUT-PATCH B:
      - If last_hidden [B, T, D], pass it to model.read_from_pvm as a whole.
      - No more slicing to last token here; query is computed in _read_from_pvm_impl
        via compute_query_key, same as in collect_alignment_pairs (KISS).
    """
    import torch
    import torch.nn.functional as F

    # Guard: if not readout head, return original logits
    if not hasattr(model, "read_from_pvm"):
        return lm_logits

    # Early exit for non-shared modes
    if head != 'shared':
        return lm_logits

    # ------------------------------
    # STEP 1: PVM readout -> [B, D]
    # ------------------------------
    pvm_read = None

    if last_hidden is not None:
        # IMPORTANT:
        # - If [B, T, D], leave as is → _read_from_pvm_impl will
        #   call compute_query_key(last_hidden, window, mode)
        # - If [B, D], assume it is already a query (legacy mode)
        if last_hidden.dim() not in (2, 3):
            raise ValueError(f"[READOUT] Unexpected last_hidden shape: {last_hidden.shape}")

        pvm_read = model.read_from_pvm(last_hidden)
    else:
        # Fallback: use model._last_hidden_states if it exists
        if hasattr(model, "_last_hidden_states") and model._last_hidden_states is not None:
            pvm_read = model.read_from_pvm(model._last_hidden_states)
        else:
            # Nothing to read
            return lm_logits

    # Guard: without PVM result → without logit changes
    if pvm_read is None:
        return lm_logits

    # Store for InfoNCE / diagnostiko
    model._last_pvm_readout = pvm_read

    # ------------------------------------------
    # STEP 2: Mapping to SIR / vocabulary space
    # ------------------------------------------
    # pvm_read: [B, D_pvm]
    B = pvm_read.shape[0]

    # A) 'separate' mode: PVM -> SIR -> Vocab (via readout_head_proj)
    if hasattr(model, "sir_compress") and hasattr(model, "readout_head_proj"):
        z_sir = model.sir_compress(pvm_read)
        vocab_logits = model.readout_head_proj(z_sir)

    # B) 'shared' mode (KISS-Ridge): PVM -> Embedding (via pvm2emb) -> Vocab (via lm_head)
    elif hasattr(model, "lm_head"):
        # Check if we have 'pvm2emb' bridge (KISS-Ridge matrix)
        if hasattr(model, "pvm2emb"):
            # FIX: Use bridge!
            # PVM (faza) -> pvm2emb -> Embedding prostor
            # Match pvm2emb dtype for computation
            pvm2emb_dtype = next(model.pvm2emb.parameters()).dtype
            emb_pred = model.pvm2emb(pvm_read.to(pvm2emb_dtype))
            # Convert to lm_head dtype if needed
            target_dtype = next(model.lm_head.parameters()).dtype
            if emb_pred.dtype != target_dtype:
                emb_pred = emb_pred.to(target_dtype)

            # Debug confirmation (only print once per eval)
            if not hasattr(model, '_pvm2emb_used_logged'):
                print("[READOUT-BRIDGE] Using pvm2emb (KISS-Ridge) for PVM->Embedding->Vocab projection")
                model._pvm2emb_used_logged = True
        else:
            # If no bridge, use raw PVM (legacy)
            emb_pred = pvm_read.to(next(model.lm_head.parameters()).dtype)

            if not hasattr(model, '_pvm_legacy_logged'):
                print("[READOUT-WARNING] No pvm2emb bridge found, using raw PVM->Vocab (legacy mode)")
                model._pvm_legacy_logged = True

        # Embedding -> Vocab
        vocab_logits = model.lm_head(emb_pred)

    else:
        # No head, no point continuing
        return lm_logits

    # ------------------------------------
    # STEP 3: Window mask (needle window)
    # ------------------------------------
    if input_ids is not None and retrieval_start is not None and readout_window is not None:
        V = vocab_logits.shape[1]
        device = vocab_logits.device
        mask = torch.zeros((B, V), dtype=torch.bool, device=device)

        if isinstance(retrieval_start, int):
            rs = torch.full((B,), retrieval_start, device=device, dtype=torch.long)
        else:
            rs = retrieval_start.to(device)

        T = input_ids.shape[1]
        for b in range(B):
            s = int(rs[b].item())
            e = min(T, s + readout_window)
            window_ids = torch.unique(input_ids[b, s:e])
            mask[b, window_ids] = True

        neg_inf = torch.finfo(vocab_logits.dtype).min
        vocab_logits = torch.where(
            mask,
            vocab_logits,
            torch.tensor(neg_inf, device=device, dtype=vocab_logits.dtype),
        )

    # ----------------------------
    # STEP 4: Top-K mask (optional)
    # ----------------------------
    if topk is not None and topk > 0:
        V = vocab_logits.shape[1]
        k = min(topk, V)
        vals, idx = torch.topk(vocab_logits, k=k, dim=1)
        keep = torch.zeros_like(vocab_logits, dtype=torch.bool)
        keep.scatter_(1, idx, True)
        neg_inf = torch.finfo(vocab_logits.dtype).min
        vocab_logits = torch.where(
            keep,
            vocab_logits,
            torch.tensor(neg_inf, device=vocab_logits.device, dtype=vocab_logits.dtype),
        )

    # ----------------------------
    # STEP 5: Residual na lm_logits
    # ----------------------------
    vocab_logits = vocab_logits.to(lm_logits.dtype)

    delta = scale * vocab_logits
    delta = torch.nan_to_num(delta, nan=0.0, posinf=10.0, neginf=-10.0)
    delta = delta.clamp(min=-10.0, max=10.0)

    if lm_logits.dim() == 3:
        out = lm_logits.clone()
        out[:, -1, :] = out[:, -1, :] + delta
    else:
        out = lm_logits + delta

    # ----------------------------
    # STEP 6: Logging + nan guard
    # ----------------------------
    if not hasattr(model, "_readout_shared_logged"):
        model._readout_shared_logged = True
        B = vocab_logits.shape[0]
        window_active = (
            input_ids is not None and retrieval_start is not None and readout_window is not None
        )
        topk_active = topk is not None and topk > 0
        print(
            f"[READOUT-SHARED] bsz={B}, window={readout_window if window_active else None}, "
            f"topk={topk if topk_active else None}, scale={scale}"
        )
        print(f"[READOUT-SHARED] window|topk active: {window_active}|{topk_active}")

    if not torch.isfinite(out).all():
        print("[READOUT] WARNING: non-finite logits after readout, applying nan_to_num")
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

    return out



class AdaptiveOmegaTrainer(Trainer):
    """Custom Trainer with separate optimizer group for Adaptive-OMEGA parameters"""
    def __init__(self, *args, adaptive_omega_lr=1e-7, adaptive_omega_interval=10,
                 infonce_weight=0.0, infonce_negatives=128, infonce_tau=0.08,
                 pvm2emb_ridge_recalib=0, **kwargs):
        # Extract our custom parameters before passing to parent
        self.adaptive_omega_lr = adaptive_omega_lr
        self.adaptive_omega_interval = adaptive_omega_interval
        # InfoNCE parameters
        self.infonce_weight = infonce_weight
        self.infonce_negatives = infonce_negatives
        self.infonce_tau = infonce_tau
        # Ridge recalibration parameter
        self.pvm2emb_ridge_recalib = pvm2emb_ridge_recalib
        # Debug logging for InfoNCE configuration (only on first instantiation)
        if not hasattr(self, "_infonce_logged"):
            self._infonce_logged = True
            print(f"[INFONCE-CONFIG] weight={self.infonce_weight}, negatives={self.infonce_negatives}, tau={self.infonce_tau}")
        self._sanity_checked = False  # Flag for first-step device check
        self._compute_loss_first_call = True  # Flag for debug logging
        self._train_start_time = None  # Track training start time for progress ETA
        # PHASE 10D: Track pointer quality during training
        self.pointer_quality_history = []  # Stores needle_in_topk_ratio per step
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """Create optimizer with separate parameter groups for omega and copy_gate"""
        if self.optimizer is None:
            # Separate omega, copy_gate, and base params
            omega_params = []
            copy_gate_params = []
            base_params = []

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    if "omega_param" in n:
                        omega_params.append(p)
                    elif "copy_gate" in n:
                        copy_gate_params.append(p)
                    else:
                        base_params.append(p)

            # Create optimizer with separate groups
            optimizer_grouped_parameters = [
                {"params": base_params, "lr": self.args.learning_rate, "weight_decay": self.args.weight_decay}
            ]

            # Add copy_gate group with 100x higher LR
            if copy_gate_params:
                copy_gate_lr = 2e-4  # 100x higher than typical 2e-6
                optimizer_grouped_parameters.append({
                    "params": copy_gate_params,
                    "lr": copy_gate_lr,
                    "weight_decay": 0.0
                })
                print(f"[COPY-GATE] Created separate optimizer group:")
                print(f"[COPY-GATE]   copy_gate params: {len(copy_gate_params)}, LR={copy_gate_lr}")

            # Add omega group if needed
            if omega_params and self.adaptive_omega_lr > 0:
                optimizer_grouped_parameters.append({
                    "params": omega_params,
                    "lr": self.adaptive_omega_lr,
                    "weight_decay": 1e-3
                })
                print(f"[ADAPTIVE-OMEGA] Created separate optimizer group:")
                print(f"[ADAPTIVE-OMEGA]   Omega params: {len(omega_params)}, LR={self.adaptive_omega_lr}")

            print(f"[OPTIMIZER] Base params: {len(base_params)}, LR={self.args.learning_rate}")

            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )

        return self.optimizer

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to add sanity check on first batch and periodic Ridge recalibration"""
        # Sanity check: Print device of first batch (GPU utilization check)
        if not self._sanity_checked:
            self._sanity_checked = True
            input_ids = inputs.get('input_ids', None)
            if input_ids is not None:
                print(f"[SANITY] First batch device: {input_ids.device}")
                print(f"[SANITY] Batch shape: {input_ids.shape}")
                if input_ids.device.type == 'cpu':
                    print(f"[WARNING] Batch is on CPU! GPU will be idle. Check DataLoader pin_memory and device transfer.")
                else:
                    print(f"[SANITY] OK Batch on GPU ({input_ids.device}) - training will utilize GPU")
            else:
                print(f"[SANITY] No input_ids in batch (keys: {inputs.keys()})")

        # Periodic Ridge recalibration (optional, controlled by --pvm2emb-ridge-recalib)
        if self.pvm2emb_ridge_recalib > 0 and self.state.global_step > 0 and self.state.global_step % self.pvm2emb_ridge_recalib == 0 and hasattr(model, 'pvm2emb'):
            print(f"[RIDGE-RECALIB] Recalibrating Ridge mapping at step {self.state.global_step}")

            # Temporarily set model to eval for stable collection
            model.eval()

            try:
                # Collect fresh alignment pairs
                train_dataloader = self.get_train_dataloader()
                Z_recalib, E_recalib = collect_alignment_pairs(
                    train_dataloader,
                    model,
                    model.get_input_embeddings().weight,
                    max_pairs=256  # Fewer pairs for faster recalibration
                )

                # Compute new Ridge weights
                W_recalib = ridge_fit(Z_recalib, E_recalib, l2=1e-3)

                # Exponential moving average update (keep 70% old, 30% new)
                with torch.no_grad():
                    old_weight = model.pvm2emb.weight.data
                    new_weight = W_recalib.T
                    model.pvm2emb.weight.data = 0.7 * old_weight + 0.3 * new_weight

                print(f"[RIDGE-RECALIB] Ridge weights updated (EMA blend)")
            except Exception as e:
                print(f"[RIDGE-RECALIB] ERROR during recalibration: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Back to training mode
                model.train()

        # Call parent training_step with graceful handling of no-gradient case
        # This occurs in baseline configs (n_afrb=0, infonce_weight=0, readout_from=none)
        try:
            # Newer versions support num_items_in_batch parameter
            return super().training_step(model, inputs, num_items_in_batch)
        except TypeError:
            # Older versions only accept (model, inputs)
            return super().training_step(model, inputs)
        except RuntimeError as e:
            # Handle "element 0 of tensors does not require grad" error
            if "does not require grad" in str(e):
                if self.state.global_step == 0:
                    print("\n" + "="*80)
                    print("[BASELINE-MODE] No trainable parameters in computational graph")
                    print("="*80)
                    print("This is expected in baseline mode with:")
                    print(f"  - n_afrb = {getattr(self.args, 'n_afrb', 0)}")
                    print(f"  - infonce_weight = {getattr(self, 'infonce_weight', 0.0)}")
                    print(f"  - readout_from = {getattr(self.args, 'readout_from', 'none')}")
                    print("\nThe model is fully frozen (backbone frozen, no AFRB adapters, no InfoNCE).")
                    print("Loss is computed for logging purposes, but no gradients will be applied.")
                    print("Training will continue without weight updates (pure evaluation mode).")
                    print("="*80 + "\n")

                # Return a dummy loss value without backward pass
                # This allows training to continue for logging/evaluation
                return torch.tensor(0.0, device=model.device, requires_grad=False)
            else:
                # Re-raise other RuntimeErrors
                raise

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to add retrieval aux loss for needle task.

        If needle-query mode is active (retrieval_start >= 0), compute:
            L = L_lm + lambda * L_retrieval

        Where L_retrieval is cross-entropy loss on the retrieved needle tokens only.
        """
        # DEBUG: Print on first call
        if self._compute_loss_first_call:
            self._compute_loss_first_call = False
            print(f"[DEBUG] compute_loss() FIRST CALL: global_step={self.state.global_step}", flush=True)
            print(f"[DEBUG] Input keys: {list(inputs.keys())}", flush=True)
            print(f"[DEBUG] retrieval_start in inputs: {'retrieval_start' in inputs}", flush=True)
            if 'retrieval_start' in inputs:
                print(f"[DEBUG] retrieval_start value: {inputs['retrieval_start']}", flush=True)

        # DEBUG: Track compute_loss calls
        if self.state.global_step < 3:
            print(f"[COMPUTE-LOSS] Called at step={self.state.global_step}, keys={list(inputs.keys())}", flush=True)

        # SAFEGUARD: Verify retrieval_start is present for needle-query tasks
        is_needle_query_task = (
            hasattr(self.args, 'task') and self.args.task == 'needle' and
            hasattr(self.args, 'needle_query') and self.args.needle_query
        )

        if is_needle_query_task and self.state.global_step == 0:
            # DESIGN-DOC: Diagnostic print for ALIGN info
            pvm2emb_grad = any(p.requires_grad for p in model.pvm2emb.parameters()) if hasattr(model, 'pvm2emb') else False
            sir_grad = any(p.requires_grad for p in model.readout_head_proj.parameters()) if hasattr(model, 'readout_head_proj') else False
            print(f"[ALIGN] InfoNCE={self.infonce_weight}, pvm2emb.grad={pvm2emb_grad}, SIR.grad={sir_grad}")

            # Hard check on first step (loud failure if broken)
            if 'retrieval_start' not in inputs:
                error_msg = (
                    "\n" + "="*80 + "\n"
                    "CRITICAL ERROR: retrieval_start field missing from batch!\n"
                    "="*80 + "\n"
                    "This is a DATA PIPELINE BUG. Needle-query mode requires retrieval_start,\n"
                    "but the data collator did not preserve it.\n\n"
                    "Expected: NeedleDataCollator should be active\n"
                    "Check: Trainer initialization should pass data_collator=NeedleDataCollator(tokenizer)\n\n"
                    "Without retrieval_start, readout will NOT activate and needle_hit_rate will be 0!\n"
                    "="*80 + "\n"
                )
                print(error_msg, flush=True)
                raise AssertionError("retrieval_start missing from batch (needle-query mode requires it)")

        # Check if we're in needle-query mode
        retrieval_start = inputs.get('retrieval_start', None)

        # DEBUG: Check observability condition
        if self._compute_loss_first_call:
            print(f"[DEBUG] is_needle_query_task: {is_needle_query_task}", flush=True)
            if is_needle_query_task:
                print(f"[DEBUG] hasattr(self.args, 'needle_query'): {hasattr(self.args, 'needle_query')}", flush=True)
                print(f"[DEBUG] self.args.needle_query: {self.args.needle_query if hasattr(self.args, 'needle_query') else 'N/A'}", flush=True)

        # Observability: Log why use_retrieval_loss might be False
        if is_needle_query_task and self.state.global_step == 0:
            if retrieval_start is None:
                print(f"[RETRIEVAL-LOSS] ERROR DISABLED: retrieval_start is None (data collator bug?)")
            elif not isinstance(retrieval_start, torch.Tensor):
                print(f"[RETRIEVAL-LOSS] ERROR DISABLED: retrieval_start is not a tensor (type={type(retrieval_start)})")
            elif not hasattr(self.args, 'lambda_retrieval'):
                print(f"[RETRIEVAL-LOSS] ERROR DISABLED: lambda_retrieval not set")
            elif self.args.lambda_retrieval <= 0:
                print(f"[RETRIEVAL-LOSS] ERROR DISABLED: lambda_retrieval={self.args.lambda_retrieval} (must be > 0)")
            elif not (retrieval_start >= 0).any():
                print(f"[RETRIEVAL-LOSS] ERROR DISABLED: all retrieval_start values < 0 (no query segments)")
            else:
                print(f"[RETRIEVAL-LOSS] OK ENABLED: lambda={self.args.lambda_retrieval}, retrieval_start present")

        use_retrieval_loss = (
            hasattr(self.args, 'lambda_retrieval') and
            self.args.lambda_retrieval > 0 and
            retrieval_start is not None and
            isinstance(retrieval_start, torch.Tensor) and
            (retrieval_start >= 0).any()
        )

        if use_retrieval_loss:
            # Compute standard LM loss (HuggingFace default)
            # Enable hidden_states output for query-conditioned copy gate
            outputs = model(**inputs, output_hidden_states=True)
            loss_lm = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

            # E-FIX C) Force _last_hidden_states for readout observability
            # Ensures apply_readout_to_logits has access to hidden states
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                model._last_hidden_states = outputs.hidden_states[-1]

            # NEW: Apply query-addressed readout to logits
            logits_base = outputs.logits  # [batch, seq_len, vocab] - SAVE BASE BEFORE READOUT
            logits = logits_base.clone()  # Start with base logits

            # Extract last hidden state for query-conditioned copy gate
            # hidden_states is tuple: (emb, layer1, layer2, ..., layerN)
            # We want the last layer output: hidden_states[-1]  [B, T, D]
            last_hidden = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None

            if hasattr(self.args, 'readout_from') and self.args.readout_from != 'none':
                # Get parameters from inputs (design specification)
                retrieval_start = inputs.get('retrieval_start', None)
                input_ids = inputs.get('input_ids', None)

                logits = apply_readout_to_logits(
                    model,
                    logits,
                    last_hidden,
                    input_ids=input_ids,
                    retrieval_start=retrieval_start,
                    readout_window=getattr(self.args, 'readout_window', None),
                    topk=getattr(self.args, 'readout_topk', None),
                    scale=getattr(self.args, 'readout_scale', 1.0),
                    head=getattr(self.args, 'readout_head', 'shared')
                )

                # ★ POINTER-GENERATOR MIX: Query-conditioned copy gate (See et al. 2017)
                # Instead of hard-pointer, mix base_logits (generate) vs pointer_logits (copy)
                # p_copy = sigmoid(w^T q + b) where q is query hidden state
                # DISABLED: Copy-gate causes gradient explosion at step ~100
                if False and hasattr(model, 'copy_gate_w') and last_hidden is not None and 'needle' in inputs:
                    needle_ids = inputs['needle']  # [B, K] - needle token IDs
                    if isinstance(needle_ids, torch.Tensor) and needle_ids.numel() > 0:
                        # Create pointer logits: base + boost for needle IDs only
                        pointer_logits = logits_base.clone()  # Start from base (no readout noise)

                        # Boost scale: how much to lift needle IDs in copy mode
                        # (Can be controlled by --hard-pointer-scale or default 10.0)
                        boost_scale = self.args.hard_pointer_scale if hasattr(self.args, 'hard_pointer_scale') and self.args.hard_pointer_scale > 0 else 10.0
                        gain_val = torch.tensor(boost_scale, dtype=pointer_logits.dtype, device=pointer_logits.device)

                        # E-FIX B) Apply boost to FULL RETRIEVAL WINDOW, not just last position
                        # Get retrieval_start positions and apply boost across all retrieval timesteps
                        batch_size = pointer_logits.shape[0]
                        seq_len = pointer_logits.shape[1]

                        # Get retrieval mask (already computed above at line 579-584)
                        # Recreate here for boost application
                        for b in range(batch_size):
                            start_pos = retrieval_start[b].item()
                            if start_pos >= 0 and start_pos < seq_len:
                                # Get all timesteps in retrieval window
                                retrieval_timesteps = list(range(start_pos, seq_len))
                                num_timesteps = len(retrieval_timesteps)

                                if num_timesteps > 0:
                                    # Get needle_ids for this batch example
                                    needle_ids_b = needle_ids[b]  # [K] - needle token IDs for batch b

                                    # Create gain vector [K] with boost_scale value
                                    gain_b = torch.full((needle_ids_b.shape[0],), boost_scale,
                                                       dtype=pointer_logits.dtype, device=pointer_logits.device)

                                    # Apply boost: scatter_add_ at each timestep in retrieval window
                                    for t in retrieval_timesteps:
                                        pointer_logits[b, t, :].scatter_add_(
                                            dim=-1,
                                            index=needle_ids_b,  # [K] - 1D index tensor
                                            src=gain_b           # [K] - 1D source tensor
                                        )

                        # ★ Query-conditioned copy gate: p_copy = sigmoid(w^T q + b)
                        # Extract query representation (last position hidden state)
                        query_hidden = last_hidden[:, -1, :]  # [B, D]

                        # Compute logit: w^T q + b  (broadcast over batch)
                        copy_logit = torch.matmul(query_hidden, model.copy_gate_w) + model.copy_gate_b  # [B]
                        p_copy = torch.sigmoid(copy_logit)  # [B] - per-example copy probability

                        # Interpolate between logits (generate + readout) and pointer_logits (copy)
                        # Reshape p_copy for broadcasting: [B, 1, 1]
                        p_copy_bcast = p_copy.view(-1, 1, 1)
                        logits = (1 - p_copy_bcast) * logits + p_copy_bcast * pointer_logits

                        # Diagnostic logging every 50 steps
                        if self.state.global_step % 50 == 0:
                            with torch.no_grad():
                                p_copy_mean = p_copy.mean().item()
                                # Compute gate_w_norm
                                gate_w_norm = 0.0
                                for n, p in model.named_parameters():
                                    if 'copy_gate' in n and p.dim() > 0:
                                        gate_w_norm += float(p.norm().detach().cpu())
                                print(f"[COPY] step={self.state.global_step} p_copy_mean={p_copy_mean:.4f} gate_w_norm={gate_w_norm:.6f}", flush=True)

                # DIAGNOSTIC: Log readout delta top-K tokens (first 3 steps for debugging)
                # PHASE 10D: Also track pointer quality throughout training
                # BUG FIX (2025-11-13): Use larger K to avoid saturation at 5/16=0.3125
                if self.state.global_step < 3 or self.state.global_step % 50 == 0:  # Extended tracking
                    with torch.no_grad():
                        delta = logits - logits_base  # [batch, seq_len, vocab]
                        delta_last = delta[:, -1, :]  # [batch, vocab] - last token position

                        # Use K=64 for pointer quality (covers typical needle_len=16 with room for noise)
                        k_pointer = 64
                        vals_full, idxs_full = torch.topk(delta_last, k=k_pointer, dim=-1)

                        # Also compute top-5 for logging (backward compatibility)
                        vals_top5, idxs_top5 = torch.topk(delta_last, k=5, dim=-1)

                        # Check if needle IDs are in delta top-K
                        needle_ids = inputs.get('needle', None)
                        if needle_ids is not None and isinstance(needle_ids, torch.Tensor):
                            needle_ids_flat = needle_ids[0] if len(needle_ids.shape) > 1 else needle_ids
                            topk_ids = idxs_full[0].tolist()
                            top5_ids = idxs_top5[0].tolist()
                            needle_ids_list = needle_ids_flat.tolist() if isinstance(needle_ids_flat, torch.Tensor) else []
                            overlap_k = set(topk_ids) & set(needle_ids_list)
                            overlap_5 = set(top5_ids) & set(needle_ids_list)

                            # PHASE 10D: Compute pointer quality metrics (FIXED)
                            # WHY FIX: Old metric saturated at 5/16=0.3125 (k=5, needle_len=16)
                            # NEW: Recall = (needle tokens in top-K) / needle_len
                            #      Precision = (needle tokens in top-K) / K
                            #      F1 = harmonic mean of recall and precision
                            needle_recall = len(overlap_k) / max(len(needle_ids_list), 1)
                            needle_precision = len(overlap_k) / k_pointer
                            needle_f1 = (2 * needle_recall * needle_precision) / max(needle_recall + needle_precision, 1e-8)

                            self.pointer_quality_history.append({
                                'step': self.state.global_step,
                                'recall': needle_recall,
                                'precision': needle_precision,
                                'f1': needle_f1,
                                'ratio': needle_recall  # Backward compatibility (was: len(overlap) / needle_len)
                            })

                            if self.state.global_step < 3:  # Verbose logging for first steps
                                print(f"[DELTA-TOP5 step={self.state.global_step}] IDs={top5_ids}, vals={[f'{v:.3f}' for v in vals_top5[0].tolist()]}", flush=True)
                                print(f"[DELTA-TOP5] Needle IDs present: {len(overlap_5)}/{len(needle_ids_list)} (overlap={list(overlap_5)})", flush=True)
                                print(f"[POINTER-QUALITY] K={k_pointer}: recall={needle_recall:.3f}, precision={needle_precision:.3f}, F1={needle_f1:.3f}", flush=True)
                        else:
                            if self.state.global_step < 3:
                                print(f"[DELTA-TOP5 step={self.state.global_step}] needle field not found in inputs!", flush=True)

                # PVM DEBUG: Log gate strength and alpha/beta/omega values every 10 steps
                if self.state.global_step % 10 == 0:
                    with torch.no_grad():
                        gate_values = []
                        alpha_logged = False

                        # Check if model has afrb_adapters (AFRB blocks)
                        if hasattr(model, 'afrb_adapters') and model.afrb_adapters is not None:
                            for blk in model.afrb_adapters:
                                if hasattr(blk, 'pvm') and blk.pvm is not None:
                                    # Log gate strength (averaged across all blocks)
                                    if hasattr(blk.pvm, 'gate'):
                                        g = torch.sigmoid(blk.pvm.gate).mean().item()
                                        gate_values.append(g)

                                    # Log alpha/beta/omega from first block only
                                    if not alpha_logged:
                                        a = blk.pvm.alpha if hasattr(blk.pvm, 'alpha') else None
                                        b = blk.pvm.beta if hasattr(blk.pvm, 'beta') else None
                                        w = blk.pvm.omega if hasattr(blk.pvm, 'omega') else None
                                        if a is not None:
                                            print(f"[PVM-DEBUG] step={self.state.global_step} alpha={a:.4f} beta={b:.4f} omega={w:.4f}", flush=True)
                                            alpha_logged = True

                        # Print gate average
                        if gate_values:
                            avg_gate = sum(gate_values) / len(gate_values)
                            print(f"[PVM-DEBUG] step={self.state.global_step} gate_mean={avg_gate:.4f}", flush=True)

                        # Manual progress logging with time estimates (every 10 steps)
                        if self.state.max_steps > 0:
                            # Initialize start time if needed
                            if self._train_start_time is None:
                                self._train_start_time = time.time()

                            # Calculate timing
                            elapsed_sec = time.time() - self._train_start_time
                            elapsed_min = elapsed_sec / 60.0

                            progress_pct = (self.state.global_step / self.state.max_steps) * 100
                            steps_done = max(self.state.global_step, 1)
                            sec_per_step = elapsed_sec / steps_done

                            steps_remaining = self.state.max_steps - self.state.global_step
                            eta_sec = steps_remaining * sec_per_step
                            eta_min = eta_sec / 60.0

                            # Use \r to update same line (horizontal progress bar)
                            print(f"\r[PROGRESS] {self.state.global_step}/{self.state.max_steps} ({progress_pct:.1f}%) | "
                                  f"Elapsed: {elapsed_min:.1f}min | ETA: {eta_min:.1f}min | {sec_per_step:.1f}s/it" + " "*20,
                                  end='', flush=True)

                # Log readout activity (more verbose to debug)
                if self.state.global_step == 0:
                    print(f"[READOUT-INIT] Readout enabled: from={self.args.readout_from}, scale={self.args.readout_scale}, topk={self.args.readout_topk}, head={self.args.readout_head}")
                if self.state.global_step % 50 == 0:
                    print(f"[READOUT] Step {self.state.global_step}: Applied {self.args.readout_from} readout")
                    # Stats logging removed - apply_readout_to_logits no longer tracks _readout_stats_history

            # Compute retrieval aux loss (cross-entropy on needle positions only)
            labels = inputs['labels']  # [batch, seq_len]

            # Mask: only positions after retrieval_start (where needle should be generated)
            batch_size, seq_len = labels.shape
            retrieval_mask = torch.zeros_like(labels, dtype=torch.bool)

            for i in range(batch_size):
                start_pos = retrieval_start[i].item()
                if start_pos >= 0 and start_pos < seq_len:
                    retrieval_mask[i, start_pos:] = True

            # Flatten logits and labels, apply mask
            logits_flat = logits.view(-1, logits.size(-1))  # [batch*seq, vocab]
            labels_flat = labels.view(-1)  # [batch*seq]
            mask_flat = retrieval_mask.view(-1)  # [batch*seq]

            # Filter only retrieval positions (where mask=True and label != -100)
            valid_retrieval = mask_flat & (labels_flat != -100)

            if valid_retrieval.sum() > 0:
                # Compute cross-entropy only on retrieval positions
                # Extract retrieval logits and targets
                logits_valid = logits_flat[valid_retrieval]
                targets_valid = labels_flat[valid_retrieval]

                # Guard against NaN/Inf before CE computation
                logits_valid = torch.nan_to_num(
                    logits_valid,
                    nan=0.0,
                    posinf=1e4,
                    neginf=-1e4,
                )

                # Optional debug assertion - warn if non-finite logits detected
                if not torch.isfinite(logits_valid).all():
                    print(f"[RETRIEVAL-LOSS] WARNING: non-finite logits detected in retrieval window at step {self.state.global_step}")

                # Stable cross-entropy computation
                loss_retrieval = torch.nn.functional.cross_entropy(
                    logits_valid,
                    targets_valid,
                    reduction='mean'
                )

                # Weighted sum: L = L_lm + lambda * L_retrieval
                loss = loss_lm + self.args.lambda_retrieval * loss_retrieval

                # Log retrieval loss (for monitoring)
                if self.state.global_step % 50 == 0:
                    print(f"[RETRIEVAL-LOSS] Step {self.state.global_step}: "
                          f"L_lm={loss_lm.item():.4f}, L_ret={loss_retrieval.item():.4f}, "
                          f"Total={loss.item():.4f}")
            else:
                # No valid retrieval positions, fall back to LM loss
                loss_retrieval = torch.zeros([], device=logits.device)
                loss = loss_lm

            # InfoNCE Contrastive Loss (PVM->Embedding space alignment)
            # DEBUG: Check InfoNCE args on first step
            if self.state.global_step == 0:
                print(f"[INFONCE-DEBUG-ARGS] Step 0: Checking InfoNCE configuration...")
                print(f"[INFONCE-DEBUG-ARGS] self.infonce_weight = {self.infonce_weight}")
                print(f"[INFONCE-DEBUG-ARGS] self.infonce_negatives = {self.infonce_negatives}")
                print(f"[INFONCE-DEBUG-ARGS] self.infonce_tau = {self.infonce_tau}")

            # DESIGN-DOC: Gradient flow diagnostics at step 0 and unfreeze_step
            unfreeze_step = getattr(self.args, 'kiss_ridge_unfreeze_step', 0)
            if self.state.global_step in (0, unfreeze_step):
                has_sir_grad = any(p.requires_grad for p in model.sir_compress.parameters()) if hasattr(model, "sir_compress") else False
                has_pvm2emb_grad = any(p.requires_grad for p in model.pvm2emb.parameters()) if hasattr(model, "pvm2emb") else False
                print(f"[INFONCE-DEBUG-ARGS] step={self.state.global_step}: InfoNCE={self.infonce_weight}, "
                      f"sir.grad={has_sir_grad}, pvm2emb.grad={has_pvm2emb_grad}", flush=True)

            if self.infonce_weight > 0:
                # DEBUG: Log on first step
                if self.state.global_step == 0 or self.state.global_step == 1:
                    print(f"[INFONCE-DEBUG] Step {self.state.global_step}: InfoNCE ACTIVATED: weight={self.infonce_weight}")
                    print(f"[INFONCE-DEBUG] Has sir_compress: {hasattr(model, 'sir_compress')}")
                    print(f"[INFONCE-DEBUG] Has _last_pvm_readout: {hasattr(model, '_last_pvm_readout')}")
                    if hasattr(model, '_last_pvm_readout'):
                        print(f"[INFONCE-DEBUG] _last_pvm_readout shape: {model._last_pvm_readout.shape}")

                    # DESIGN-DOC: Gradient flow diagnostics
                    has_sir_grad = any(p.requires_grad for p in model.sir_compress.parameters()) if hasattr(model, "sir_compress") else False
                    has_pvm2emb_grad = any(p.requires_grad for p in model.pvm2emb.parameters()) if hasattr(model, "pvm2emb") else False
                    print(f"[INFONCE-DEBUG] Gradient flow: sir_compress={has_sir_grad}, pvm2emb={has_pvm2emb_grad}")

                # Get needle IDs for positive examples
                needle_ids = inputs.get('needle', None)
                if self.state.global_step <= 1:
                    print(f"[INFONCE-DEBUG] Step {self.state.global_step}: needle_ids: {needle_ids is not None}, type: {type(needle_ids) if needle_ids is not None else None}")
                if needle_ids is not None and isinstance(needle_ids, torch.Tensor) and needle_ids.numel() > 0:
                    # Get PVM representation from SIR layer (if available)
                    # We need to extract z_pvm from readout path
                    if hasattr(model, 'sir_compress') and hasattr(model, '_last_pvm_readout'):
                        # _last_pvm_readout is set by apply_readout_to_logits
                        # --- FIX: _last_pvm_readout is now [B, D], not [D] ---
                        z_pvm = model._last_pvm_readout
                        if z_pvm.dim() == 1:
                            # Safety fallback for compatibility
                            z_pvm = z_pvm.unsqueeze(0).expand(batch_size, -1)
                        # ----------------------------------------------------

                        # DESIGN-DOC: dtype match - convert z_pvm to SIR layer dtype before compression
                        sir_dtype = next(model.sir_compress.parameters()).dtype
                        if self.state.global_step == 0:
                            print(f"[DTYPE-FIX] z_pvm dtype BEFORE: {z_pvm.dtype}, SIR dtype: {sir_dtype}")
                        z_pvm = z_pvm.to(sir_dtype)
                        if self.state.global_step == 0:
                            print(f"[DTYPE-FIX] z_pvm dtype AFTER: {z_pvm.dtype}")

                        # Compress to SIR space
                        z_sir = model.sir_compress(z_pvm)  # [B, 512]

                        # Get positive embeddings (needle tokens)
                        # lm_head.weight shape: [V, H] where H=2048
                        # We want embeddings for needle IDs
                        # Take mean across needle tokens as positive target
                        needle_emb = model.lm_head.weight[needle_ids]  # [B, K, H]
                        e_pos = needle_emb.mean(dim=1)  # [B, H]

                        # [E-ALIGN DISABLED] Project e_pos to SIR space using e2sir (NO slicing!)
                        # [E-ALIGN DISABLED] This replaces the problematic [:512] slicing with learned alignment
                        # if hasattr(model, 'e2sir'):
                        #     e_pos_sir = model.e2sir(e_pos)  # [B, H] -> [B, 512]
                        # else:
                        #     # Fallback to old slicing (should not happen with E-ALIGN)
                        e_pos_sir = e_pos[:, :512]  # [B, 512]

                        # Sample negative examples
                        num_negatives = self.infonce_negatives
                        vocab_size = model.lm_head.weight.shape[0]
                        neg_ids = torch.randint(0, vocab_size, (batch_size, num_negatives), device=z_sir.device)
                        e_neg = model.lm_head.weight[neg_ids]  # [B, N, H]

                        # [E-ALIGN DISABLED] Project e_neg to SIR space using e2sir (NO slicing!)
                        # if hasattr(model, 'e2sir'):
                        #     # e_neg is [B, N, H], need to reshape for e2sir
                        #     B, N, H = e_neg.shape
                        #     e_neg_flat = e_neg.view(B * N, H)  # [B*N, H]
                        #     e_neg_sir_flat = model.e2sir(e_neg_flat)  # [B*N, 512]
                        #     e_neg_sir = e_neg_sir_flat.view(B, N, -1)  # [B, N, 512]
                        # else:
                        #     # Fallback to old slicing
                        e_neg_sir = e_neg[:, :, :512]  # [B, N, 512]

                        # Compute similarities (cosine similarity)
                        tau = self.infonce_tau

                        # Normalize for cosine similarity
                        z_sir_norm = torch.nn.functional.normalize(z_sir, dim=-1)  # [B, 512]
                        e_pos_sir_norm = torch.nn.functional.normalize(e_pos_sir, dim=-1)  # [B, 512]
                        e_neg_sir_norm = torch.nn.functional.normalize(e_neg_sir, dim=-1)  # [B, N, 512]

                        # Positive similarity
                        sim_pos = torch.sum(z_sir_norm * e_pos_sir_norm, dim=-1) / tau  # [B]

                        # Negative similarities
                        # z_sir_norm is [B, 512], e_neg_sir_norm is [B, N, 512]
                        # Need [B, 512, 1] for bmm with [B, N, 512]
                        sim_neg = torch.bmm(e_neg_sir_norm, z_sir_norm.unsqueeze(-1)).squeeze(-1) / tau  # [B, N]

                        # InfoNCE loss: -log(exp(sim_pos) / (exp(sim_pos) + sum(exp(sim_neg))))
                        logits_infonce = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)  # [B, 1+N]
                        labels_infonce = torch.zeros(batch_size, dtype=torch.long, device=z_sir.device)  # Positive is at index 0

                        loss_infonce = torch.nn.functional.cross_entropy(logits_infonce, labels_infonce, reduction='mean')

                        # Add to total loss
                        loss = loss + self.infonce_weight * loss_infonce

                        # Log InfoNCE loss - ENHANCED: every 10 steps
                        if self.state.global_step % 10 == 0:
                            print(f"[INFONCE-STEP-{self.state.global_step}] loss_infonce={loss_infonce.item():.4f}, "
                                  f"weight={self.infonce_weight}, sim_pos_mean={sim_pos.mean().item():.4f}")

                            # DIAGNOSTIC: Check gradient flow at loss computation
                            print(f"[INFONCE-STEP-{self.state.global_step}] sir_compress.requires_grad={model.sir_compress.weight.requires_grad}")
                            pvm2emb_grad = model.pvm2emb.weight.requires_grad if hasattr(model, 'pvm2emb') else False
                            print(f"[INFONCE-STEP-{self.state.global_step}] pvm2emb.requires_grad={pvm2emb_grad}")
                            print(f"[INFONCE-STEP-{self.state.global_step}] z_sir.requires_grad={z_sir.requires_grad}")
                            print(f"[INFONCE-STEP-{self.state.global_step}] loss_infonce.requires_grad={loss_infonce.requires_grad}")
                    else:
                        # FORENSIC PATCH C: InfoNCE fallback warning
                        if self.state.global_step == 0:
                            print(
                                "[INFONCE-WARN] InfoNCE enabled (infonce_weight>0), "
                                "but sir_compress or _last_pvm_readout is missing. "
                                "No InfoNCE gradients will be applied."
                            )

            # Phase Alignment Loss (L2 regularization on phi_base for training stability)
            if hasattr(self.args, 'phase_alignment_weight') and self.args.phase_alignment_weight > 0:
                phase_alignment_loss = 0.0
                num_adapters = 0

                # Iterate over all AFRB adapters and compute L2 loss on phi_base
                if hasattr(model, 'afrb_adapters') and model.afrb_adapters is not None:
                    for adapter in model.afrb_adapters:
                        if hasattr(adapter, 'phi_base'):
                            # L2 regularization: penalize deviation from 0
                            phase_alignment_loss += adapter.phi_base ** 2
                            num_adapters += 1

                if num_adapters > 0:
                    # Average across all adapters
                    phase_alignment_loss = phase_alignment_loss / num_adapters
                    # Add weighted phase alignment loss to total loss
                    loss = loss + self.args.phase_alignment_weight * phase_alignment_loss

                    # Log phase alignment loss (for monitoring)
                    if self.state.global_step % 50 == 0:
                        print(f"[PHASE-ALIGN] Step {self.state.global_step}: "
                              f"L_align={phase_alignment_loss.item():.6f}, "
                              f"weight={self.args.phase_alignment_weight}")

            # NIVO 1: Per-step CSV logging for 3D visualization (every 5 steps)
            if self.state.global_step % 5 == 0:
                # Collect loss components with proper checks
                local_vars = locals()
                loss_lm_val = local_vars.get('loss_lm', 0.0)
                loss_infonce_val = local_vars.get('loss_infonce', 0.0)
                loss_retrieval_val = local_vars.get('loss_retrieval', 0.0)

                # Convert to float if tensors
                if isinstance(loss_lm_val, torch.Tensor):
                    loss_lm_val = loss_lm_val.item()
                if isinstance(loss_infonce_val, torch.Tensor):
                    loss_infonce_val = loss_infonce_val.item()
                if isinstance(loss_retrieval_val, torch.Tensor):
                    loss_retrieval_val = loss_retrieval_val.item()

                self._log_step_trajectory(
                    model=model,
                    loss_total=loss.item() if isinstance(loss, torch.Tensor) else loss,
                    loss_lm=loss_lm_val,
                    loss_infonce=loss_infonce_val,
                    loss_retrieval=loss_retrieval_val
                )

            return (loss, outputs) if return_outputs else loss
        else:
            # Standard LM loss (no retrieval aux loss) - compatibility for different Transformers versions
            try:
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            except TypeError:
                return super().compute_loss(model, inputs, return_outputs)

    def _log_step_trajectory(self, model, loss_total, loss_lm, loss_infonce, loss_retrieval):
        """
        NIVO 1: Per-step CSV logging for 3D visualization of training dynamics.

        Logs detailed metrics every 5 steps to step_trajectory.csv for interactive
        visualization of loss landscape, gradient flow, and phase dynamics.

        Args:
            model: The model being trained
            loss_total: Total loss value
            loss_lm: Language modeling loss
            loss_infonce: InfoNCE contrastive loss
            loss_retrieval: Retrieval auxiliary loss
        """
        # Skip if save directory not available
        if not hasattr(self.args, 'save') or self.args.save is None:
            return

        csv_path = Path(self.args.save) / 'step_trajectory.csv'

        # Write CSV header if file doesn't exist
        file_exists = csv_path.exists()
        if not file_exists:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'step', 'timestamp',
                    'loss_lm', 'loss_infonce', 'loss_retrieval', 'loss_total',
                    'pvm_norm', 'pvm_phase_mean', 'pvm_rotation_speed',
                    'sir_norm', 'sir_cosim_to_target',
                    'grad_norm_total', 'grad_norm_pvm2emb', 'grad_norm_sir_compress',
                    'omega_current', 'alpha_current',
                    'phase_coherence', 'entropy_flow', 'gamma_saturation'
                ])
            print(f"[CSV-LOG] Created step_trajectory.csv at {csv_path}", flush=True)

        # Collect metrics with graceful degradation
        timestamp = datetime.now().isoformat()

        # 1. PVM metrics
        pvm_norm = 0.0
        pvm_phase_mean = 0.0
        pvm_rotation_speed = 0.0

        if hasattr(model, 'afrb_adapters') and model.afrb_adapters is not None:
            pvm_norms = []
            pvm_means = []
            rotation_speeds = []

            for adapter in model.afrb_adapters:
                if hasattr(adapter, 'pvm') and adapter.pvm is not None:
                    # PVM memory norm
                    if hasattr(adapter.pvm, 'memory'):
                        pvm_norms.append(torch.norm(adapter.pvm.memory).item())
                        pvm_means.append(adapter.pvm.memory.mean().item())

                    # Rotation speed (spectral norm of rotation matrix)
                    if hasattr(adapter.pvm, 'rotation_matrix'):
                        try:
                            # Spectral norm = largest singular value
                            _, s, _ = torch.svd(adapter.pvm.rotation_matrix)
                            rotation_speeds.append(s.max().item())
                        except:
                            pass

            if pvm_norms:
                pvm_norm = np.mean(pvm_norms)
            if pvm_means:
                pvm_phase_mean = np.mean(pvm_means)
            if rotation_speeds:
                pvm_rotation_speed = np.mean(rotation_speeds)

        # 2. SIR metrics
        sir_norm = 0.0
        sir_cosim_to_target = 0.0

        if hasattr(model, 'sir_compress') and hasattr(model, '_last_pvm_readout'):
            try:
                z_pvm = model._last_pvm_readout
                if z_pvm is not None and isinstance(z_pvm, torch.Tensor):
                    # Ensure z_pvm has correct dtype
                    sir_dtype = next(model.sir_compress.parameters()).dtype
                    z_pvm = z_pvm.to(sir_dtype)

                    # Compute SIR norm
                    z_sir = model.sir_compress(z_pvm)
                    sir_norm = torch.norm(z_sir).item()

                    # Compute cosine similarity to target (if available)
                    # Target = mean of lm_head embeddings (approximation)
                    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
                        # Sample a small subset for efficiency
                        sample_size = min(100, model.lm_head.weight.shape[0])
                        sample_ids = torch.randint(0, model.lm_head.weight.shape[0], (sample_size,), device=z_sir.device)
                        target_emb = model.lm_head.weight[sample_ids].mean(dim=0)[:512]  # [512]

                        # Normalize and compute cosine similarity
                        z_sir_norm_vec = torch.nn.functional.normalize(z_sir.mean(dim=0) if z_sir.dim() > 1 else z_sir, dim=-1)
                        target_norm = torch.nn.functional.normalize(target_emb, dim=-1)
                        sir_cosim_to_target = torch.dot(z_sir_norm_vec, target_norm).item()
            except Exception as e:
                # Silently fail if SIR metrics unavailable
                pass

        # 3. Gradient norms (will be 0 before backward pass)
        grad_norm_total = 0.0
        grad_norm_pvm2emb = 0.0
        grad_norm_sir_compress = 0.0

        # Total gradient norm (if available from previous step)
        if hasattr(self, '_last_grad_norm'):
            grad_norm_total = self._last_grad_norm

        # PVM2EMB gradient norm
        if hasattr(model, 'pvm2emb'):
            try:
                pvm2emb_grads = []
                for p in model.pvm2emb.parameters():
                    if p.grad is not None:
                        pvm2emb_grads.append(torch.norm(p.grad).item())
                if pvm2emb_grads:
                    grad_norm_pvm2emb = np.sqrt(sum(g**2 for g in pvm2emb_grads))
            except:
                pass

        # SIR compress gradient norm
        if hasattr(model, 'sir_compress'):
            try:
                sir_grads = []
                for p in model.sir_compress.parameters():
                    if p.grad is not None:
                        sir_grads.append(torch.norm(p.grad).item())
                if sir_grads:
                    grad_norm_sir_compress = np.sqrt(sum(g**2 for g in sir_grads))
            except:
                pass

        # 4. Hyperparameter tracking
        omega_current = 0.0
        alpha_current = getattr(self.args, 'alpha', 0.0)

        if hasattr(model, 'afrb_adapters') and model.afrb_adapters is not None:
            omegas = []
            for adapter in model.afrb_adapters:
                if hasattr(adapter, 'pvm') and adapter.pvm is not None:
                    if hasattr(adapter.pvm, 'omega'):
                        omegas.append(adapter.pvm.omega)
            if omegas:
                omega_current = np.mean(omegas)

        # 5. Phase metrics (graceful degradation if not available)
        phase_coherence_val = 0.0
        entropy_flow_val = 0.0
        gamma_saturation_val = 0.0

        # Phase metrics are expensive to compute, so we skip them for now
        # They can be computed in post-processing from saved checkpoints
        # if needed for detailed analysis

        # Write row to CSV
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.state.global_step, timestamp,
                    loss_lm, loss_infonce, loss_retrieval, loss_total,
                    pvm_norm, pvm_phase_mean, pvm_rotation_speed,
                    sir_norm, sir_cosim_to_target,
                    grad_norm_total, grad_norm_pvm2emb, grad_norm_sir_compress,
                    omega_current, alpha_current,
                    phase_coherence_val, entropy_flow_val, gamma_saturation_val
                ])
        except Exception as e:
            # Silently fail if CSV write fails (don't crash training)
            print(f"[CSV-WARN] Failed to write step_trajectory.csv: {e}", flush=True)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step to enable readout during evaluation.

        DESIGN-DOC: This ensures that PVM/PLM readout is applied during evaluation,
        not just during training. This is critical for NIAH tasks where we need
        memory retrieval to work during model.generate() calls.
        """
        with torch.no_grad():
            # Forward pass to get outputs and capture hidden states
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits

            # Apply readout during evaluation (if enabled)
            if hasattr(self.args, 'readout_from') and self.args.readout_from != 'none':
                # Get parameters from inputs (design specification)
                retrieval_start = inputs.get('retrieval_start', None)
                input_ids = inputs.get('input_ids', None)

                # Get last hidden states
                last_hidden = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None else None

                # Apply readout with new signature
                if last_hidden is not None:
                    logits = apply_readout_to_logits(
                        model,
                        logits,
                        last_hidden,
                        input_ids=input_ids,
                        retrieval_start=retrieval_start,
                        readout_window=getattr(self.args, 'readout_window', None),
                        topk=getattr(self.args, 'readout_topk', None),
                        scale=getattr(self.args, 'readout_scale', 1.0),
                        head=getattr(self.args, 'readout_head', 'shared')
                    )

            # Compute loss if labels are present
            loss = None
            if 'labels' in inputs:
                labels = inputs['labels']
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction='mean'
                )

            return (loss, logits, outputs) if not prediction_loss_only else (loss, None, None)

class OmegaIntervalCallback(TrainerCallback):
    """Callback to enable/disable omega gradients at intervals"""
    def __init__(self, adaptive_omega_interval=10):
        self.adaptive_omega_interval = adaptive_omega_interval

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        """Enable/disable omega param updates based on interval"""
        if self.adaptive_omega_interval > 0:
            # Update omega only every N steps
            omega_active = (state.global_step % self.adaptive_omega_interval == 0)

            for n, p in model.named_parameters():
                if "omega_param" in n and p.requires_grad:
                    p.requires_grad = omega_active

class UnfreezeReadoutCallback(TrainerCallback):
    """E-FIX D) Callback to unfreeze readout projection after N steps"""
    def __init__(self, freeze_steps=80):
        self.freeze_steps = freeze_steps
        self.unfrozen = False

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        """Unfreeze readout translator after freeze_steps"""
        if not self.unfrozen and state.global_step >= self.freeze_steps:
            self.unfrozen = True
            print(f"\n[E-FIX-UNFREEZE] Step {state.global_step}: Unfreezing readout translator (sir_compress + e2sir + readout_head_proj)")
            print(f"[E-FIX-UNFREEZE] PVM has stabilized, now training projection\n")

            # Unfreeze sir_compress
            if hasattr(model, 'sir_compress'):
                for param in model.sir_compress.parameters():
                    param.requires_grad = True
                print(f"[E-FIX-UNFREEZE] sir_compress: UNFROZEN")

            # [E-ALIGN DISABLED] Unfreeze e2sir
            # if hasattr(model, 'e2sir'):
            #     for param in model.e2sir.parameters():
            #         param.requires_grad = True

            # Unfreeze readout_head_proj
            if hasattr(model, 'readout_head_proj'):
                for param in model.readout_head_proj.parameters():
                    param.requires_grad = True
                print(f"[E-FIX-UNFREEZE] readout_head_proj: UNFROZEN")

class UnfreezePvm2embCallback(TrainerCallback):
    """KISS-Ridge callback to unfreeze pvm2emb after N steps for InfoNCE fine-tuning"""
    def __init__(self, unfreeze_step=100):
        self.unfreeze_step = unfreeze_step
        self.unfrozen = False
        print(f"[KISS-UNFREEZE-CALLBACK] Registered: will unfreeze pvm2emb at step {self.unfreeze_step}")

    def _do_unfreeze(self, state, model):
        """Perform the actual unfreezing operation"""
        if not self.unfrozen and state.global_step >= self.unfreeze_step:
            self.unfrozen = True
            print(f"\n[KISS-UNFREEZE] Step {state.global_step}: Unfreezing pvm2emb for InfoNCE fine-tuning", flush=True)
            print(f"[KISS-UNFREEZE] Statistical initialization complete, now learning via gradients\n", flush=True)

            if hasattr(model, 'pvm2emb'):
                # Unfreeze the entire module (sets requires_grad for all parameters)
                model.pvm2emb.requires_grad_(True)
                # Verify it worked
                actual_grad = any(p.requires_grad for p in model.pvm2emb.parameters())
                print(f"[KISS-UNFREEZE] pvm2emb: UNFROZEN (requires_grad={actual_grad})", flush=True)

                if not actual_grad:
                    print(f"[KISS-UNFREEZE] WARNING: requires_grad_(True) did not take effect! Manually setting...", flush=True)
                    for p in model.pvm2emb.parameters():
                        p.requires_grad = True
                    actual_grad_retry = any(p.requires_grad for p in model.pvm2emb.parameters())
                    print(f"[KISS-UNFREEZE] Retry result: requires_grad={actual_grad_retry}", flush=True)
            else:
                print(f"[KISS-UNFREEZE] WARNING: model has no pvm2emb attribute!", flush=True)

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        """Triggered at the beginning of each training step"""
        self._do_unfreeze(state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Triggered at the end of each training step (fallback hook)"""
        self._do_unfreeze(state, model)

class GradNormTrackerCallback(TrainerCallback):
    """Callback to track gradient norm for CSV logging"""
    def __init__(self, trainer):
        self.trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Capture grad_norm from logs and store in trainer"""
        if logs is not None and 'grad_norm' in logs:
            # Store in trainer for CSV logging
            self.trainer._last_grad_norm = logs['grad_norm']

class AttentionBypass(nn.Module):
    """Bypass layer that replaces attention with identity forward (for pure PVM mode)"""
    def __init__(self, hidden_size, num_return_values=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_return_values = num_return_values  # Store how many values to return

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        """Identity forward: return input unchanged (no attention computation)"""
        # Return correct number of values based on what was detected during initialization
        if self.num_return_values == 3:
            # Newer Transformers: (hidden_states, attn_weights, present_key_value)
            return (hidden_states, None, None)
        else:
            # Older Transformers: (hidden_states, attn_weights)
            return (hidden_states, None)

def disable_attention_layers(model):
    """
    Replace all attention layers with bypass (identity forward) for pure PVM mode.

    This allows testing PVM as the sole mechanism for long-range dependencies,
    without quadratic O(n²) attention computation.

    Args:
        model: Transformer model with .layers attribute

    Returns:
        model: Modified model with attention layers replaced
    """
    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        print("[WARNING] Model structure unexpected, cannot disable attention")
        return model

    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size

    # Auto-detect number of return values by inspecting the decoder layer source code
    num_return_values = 2  # Default to 2 for older Transformers
    if num_layers > 0:
        try:
            # Inspect the decoder layer's forward method source code to see how it unpacks self_attn
            import inspect
            layer_forward_source = inspect.getsource(model.model.layers[0].forward)

            # Check if the calling code unpacks 3 values (newer) or 2 values (older)
            if 'hidden_states, self_attn_weights, present_key_value = self.self_attn' in layer_forward_source:
                num_return_values = 3
                print(f"[DISABLE-ATTN] Detected decoder layer expects 3 return values (newer Transformers)")
            elif 'hidden_states, self_attn_weights = self.self_attn' in layer_forward_source:
                num_return_values = 2
                print(f"[DISABLE-ATTN] Detected decoder layer expects 2 return values (older Transformers)")
            else:
                # Fallback: count commas in unpacking statement
                import re
                match = re.search(r'(\w+(?:\s*,\s*\w+)*)\s*=\s*self\.self_attn', layer_forward_source)
                if match:
                    unpacked_vars = match.group(1).split(',')
                    num_return_values = len(unpacked_vars)
                    print(f"[DISABLE-ATTN] Detected decoder layer expects {num_return_values} return values (regex fallback)")
                else:
                    print(f"[DISABLE-ATTN] Could not detect from source, using default (2 values)")
        except Exception as e:
            print(f"[DISABLE-ATTN] Could not inspect source code, using default (2 values): {e}")
            num_return_values = 2

    print(f"[DISABLE-ATTN] Replacing attention in {num_layers} transformer layers with bypass...")

    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, 'self_attn'):
            # Replace attention with bypass using detected signature
            layer.self_attn = AttentionBypass(hidden_size, num_return_values=num_return_values)

            # Zero out input/output layernorm for attention (optional safety)
            # This prevents any learned norms from affecting the bypass
            if hasattr(layer, 'input_layernorm'):
                for param in layer.input_layernorm.parameters():
                    param.requires_grad = False

    print(f"[DISABLE-ATTN] OK Attention disabled in all {num_layers} layers")
    print(f"[DISABLE-ATTN] Model now relies purely on:")
    print(f"[DISABLE-ATTN]   - MLP layers (local processing)")
    print(f"[DISABLE-ATTN]   - AFRB resonance (phase dynamics)")
    print(f"[DISABLE-ATTN]   - PVM memory (O(d) long-range context)")

    return model

def inject_afrb_adapters(model, n_afrb, alpha, omega, gamma, device,
                         cascade_kappa=0.0, cascade_lambda=0.0, phase_delta=0.0, omega_delta=0.0,
                         stillness_ema=0.0, stillness_floor=0.0, stillness_warm=0,
                         block_warm_delta=0, block_ramp=300, phase_ramp=300,
                         learnable_omega=False, omega_min=5.6, omega_max=6.4,
                         use_pvm=False, pvm_alpha=0.12, pvm_beta=0.88, pvm_gate_init=-2.0,
                         use_plm=False, plm_grid_x=4, plm_grid_y=4, plm_alpha=0.10, plm_beta=0.90,
                         plm_omega=6.0, plm_kappa=0.05, plm_gate_init=-2.0,
                         t2_enable=False, t2_steps=1500, t2_mode='exp', t2_k=0.001, t2_alpha=0.08,
                         t2_omega=6.0, t2_phi=1.0472, pcm_enable=False, pcm_gate_init=0.5,
                         needle_query_mode='mean_tail', readout_window=16):
    """
    Inject Adaptive Frequency-Resonant Blocks (AFRB) into transformer model.

    This function adds resonant coupling layers after the embedding layer,
    creating a parallel pathway for phase-based information processing.

    Architecture:
        Input → Embedding → [AFRB₁ → AFRB₂ → ... → AFRBₙ] → Transformer Layers → Output

    Each AFRB contains:
        - Phase-Vector Memory (PVM): Holographic pattern storage
        - Phase Lattice Memory (PLM): Geometric phase encoding
        - Dynamic Resonance Calibration (DRC): Adaptive coupling strength
        - Stillness detection: Activity-based gating

    Args:
        model: Base transformer model to augment
        n_afrb: Number of AFRB blocks to inject
        alpha: Resonance coupling strength (0.0-1.0)
        omega: Resonant frequency (typically ~6.0 for GPT-2 scale)
        gamma: Amplitude saturation factor
        device: Device for AFRB parameters

        Cascade parameters (exponential decay across blocks):
            cascade_kappa: Gamma decay rate
            cascade_lambda: Alpha decay rate

        Staggered initialization:
            phase_delta: Phase offset between blocks
            omega_delta: Frequency offset between blocks

        Dynamic warmup:
            block_warm_delta: Delay between block activations (steps)
            block_ramp: Warmup duration per block (steps)
            phase_ramp: Phase warmup duration (steps)

        Adaptive omega learning:
            learnable_omega: Enable trainable frequency tuning
            omega_min, omega_max: Frequency search bounds

        Phase-Vector Memory (PVM):
            use_pvm: Enable holographic memory storage
            pvm_alpha: Write strength (how much to store)
            pvm_beta: Read strength (how much to retrieve)
            pvm_gate_init: Initial gate bias (negative = mostly closed)

        Phase Lattice Memory (PLM):
            use_plm: Enable geometric phase encoding
            plm_grid_x, plm_grid_y: Lattice dimensions
            plm_alpha, plm_beta: Write/read strengths
            plm_omega: Lattice resonant frequency
            plm_kappa: Cross-coupling strength
            plm_gate_init: Initial gate bias

        Curvature-based memory (T2/PCM):
            t2_enable: Enable phase curvature tracking
            pcm_enable: Enable phase curvature memory

        Query extraction:
            needle_query_mode: How to extract query ('mean_tail', 'learned')
            readout_window: Context window for query extraction

    Returns:
        model: Augmented model with AFRB adapters injected
    """
    # Use embedding dimension, not hidden_size!
    embed_dim = model.model.embed_tokens.embedding_dim

    # Create adapter lane with cascade + stagger + adaptive-omega parameters
    adapters = nn.ModuleList()
    for i in range(1, n_afrb + 1):
        # Exponential decay for alpha and gamma
        alpha_i = alpha * math.exp(-cascade_lambda * (i - 1))
        gamma_i = gamma * math.exp(-cascade_kappa * (i - 1))
        # Linear offset for phase and omega
        phi_i = (i - 1) * phase_delta
        omega_i = omega + (i - 1) * omega_delta

        adapter = AFRB(
            dim=embed_dim,
            block_idx=i,
            alpha_base=alpha_i,
            gamma_base=gamma_i,
            omega=omega_i,
            phi=phi_i,
            stillness_ema=stillness_ema,
            stillness_floor=stillness_floor,
            stillness_warm=stillness_warm,
            block_warm_delta=block_warm_delta,
            block_ramp=block_ramp,
            phase_ramp=phase_ramp,
            phase_delta=phase_delta,
            learnable_omega=learnable_omega,
            omega_min=omega_min,
            omega_max=omega_max,
            use_pvm=use_pvm,
            pvm_alpha=pvm_alpha,
            pvm_beta=pvm_beta,
            pvm_gate_init=pvm_gate_init,
            use_plm=use_plm,
            plm_grid_x=plm_grid_x,
            plm_grid_y=plm_grid_y,
            plm_alpha=plm_alpha,
            plm_beta=plm_beta,
            plm_omega=plm_omega,
            plm_kappa=plm_kappa,
            plm_gate_init=plm_gate_init,
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
        adapters.append(adapter)

    adapters = adapters.to(device=device, dtype=next(model.parameters()).dtype)

    # Wrap forward (adapters now manage their own timing)
    orig_forward = model.model.forward
    def afrb_forward(input_ids=None, attention_mask=None, **kwargs):
        # Get embeddings
        if input_ids is not None:
            hidden_states = model.model.embed_tokens(input_ids)
        else:
            hidden_states = kwargs.get('inputs_embeds')

        # Apply AFRB adapters sequentially (no explicit step passing - internal tracking)
        for adapter in adapters:
            hidden_states = adapter(hidden_states)

        # NEW: Store hidden states for readout (used in compute_loss)
        model._last_hidden_states = hidden_states.detach()

        # Continue with rest of model (fix kwargs conflict)
        kwargs.pop('inputs_embeds', None)  # Remove if exists
        return orig_forward(inputs_embeds=hidden_states, attention_mask=attention_mask, **kwargs)

    model.model.forward = afrb_forward
    model.afrb_adapters = adapters
    # Store config for metrics logging
    model._stillness_ema = stillness_ema
    model._stillness_warm = stillness_warm
    model._stillness_floor = stillness_floor
    model._block_warm_delta = block_warm_delta
    model._block_ramp = block_ramp
    model._phase_ramp = phase_ramp
    # Store query config for needle task query construction
    model._needle_query_mode = needle_query_mode
    model._readout_window = readout_window

    # ============================================================
    # CLAUDE2.md FIX: Add batched read_from_pvm() method
    # ============================================================
    def _read_from_pvm_impl(self, last_hidden: torch.Tensor, topk: int = 0) -> torch.Tensor:
        """
        Batched PVM readout - retrieve information from PVM memory for each batch item.

        --- FIX: KISS-Readout-Alignment (2025-11-13) ---
        Per-sample queries instead of shared query across batch.
        Aligns with updated calibration logic for correct needle retrieval.

        Args:
            last_hidden: [B, T, D] or [B, D] - hidden states
            topk: int - top-k retrieval (0 = all entries)

        Returns:
            pvm_read: [B, D] - batched PVM readouts (one per sample)
        """
        import torch
        import torch.nn.functional as F
        from utils_alignment import compute_query_key

        # 1) Handle input shapes and compute per-sample queries
        if last_hidden.dim() == 3:
            B, T, D = last_hidden.shape

            # CHECK FOR LEARNED HEAD FIRST
            if hasattr(self, 'learned_query_head') and self.learned_query_head is not None:
                # Use the trainable MLP!
                # Ensure dtype alignment to prevent crashes
                head_dtype = next(self.learned_query_head.parameters()).dtype
                last_hidden = last_hidden.to(head_dtype)
                # learned_query_head expects [B, T, D] and returns [B, D]
                query_vectors = self.learned_query_head(last_hidden)
                hidden = query_vectors  # [B, D]

                # Log first use
                if not hasattr(self, '_learned_query_logged'):
                    print("[READOUT] Using LEARNED Query Head (trainable MLP)")
                    self._learned_query_logged = True
            else:
                # Fallback to static logic (legacy/KISS calibration)
                window = getattr(self, '_readout_window', 16)
                query_mode = "mean"  # Fixed for KISS alignment consistency

                # Compute per-sample queries (each sample gets its own query)
                queries = []
                for b in range(B):
                    q = compute_query_key(
                        last_hidden[b:b+1, :, :],  # Per-sample: [1, T, D]
                        window=window,
                        mode=query_mode
                    )  # [D]
                    queries.append(q)

                hidden = torch.stack(queries, dim=0)  # [B, D]

            B, D = hidden.shape

        elif last_hidden.dim() == 2:
            # Edge case: already [B, D] - assume these are pre-computed queries
            B, D = last_hidden.shape
            hidden = last_hidden
        else:
            raise ValueError(f"Expected [B,T,D] or [B,D], got {last_hidden.shape}")

        # 2) No adapters edge case
        if not hasattr(self, 'afrb_adapters') or len(self.afrb_adapters) == 0:
            return torch.zeros(B, D, dtype=hidden.dtype, device=hidden.device)

        # 3) Collect per-sample readouts from all adapters
        batch_readouts = []

        for adapter in self.afrb_adapters:
            if not hasattr(adapter, 'pvm') or adapter.pvm is None:
                continue

            # Per-sample PVM readout
            adapter_reads = []
            for b in range(B):
                query = F.normalize(hidden[b], dim=-1)  # [D] - per-sample query
                pvm_read = adapter.pvm.readout(query, topk=topk or 0)  # [D]
                adapter_reads.append(pvm_read)

            if adapter_reads:
                adapter_reads = torch.stack(adapter_reads, dim=0)  # [B, D]
                batch_readouts.append(adapter_reads)

        # 4) No PVM memory edge case
        if not batch_readouts:
            return torch.zeros(B, D, dtype=hidden.dtype, device=hidden.device)

        # 5) Average across adapters (keep per-sample dimension)
        if len(batch_readouts) == 1:
            result = batch_readouts[0]
        else:
            result = torch.stack(batch_readouts, dim=0).mean(dim=0)  # [B, D]

        return result.to(hidden.dtype)

    # Bind method to model
    import types
    model.read_from_pvm = types.MethodType(_read_from_pvm_impl, model)

    return model

def generate_needle_dataset(tokenizer, needle_len=16, ctx_chunks=64, ctx_chunk_len=512,
                            num_train=1000, num_val=100, seed=42, use_query=False):
    """
    Generate synthetic needle-in-haystack dataset.

    Task: Find a random "needle" (sequence of tokens) hidden in long context.

    Format (classic):
        Input:  [context_before] [NEEDLE] [context_after]
        Target: Predict next tokens (causal LM)

    Format (with query - retrieval mode):
        Input:  [context_before] [NEEDLE] [context_after] <SEP> QUERY: RETURN NEEDLE ->
        Target: [NEEDLE] (exact 16-token retrieval)

    Args:
        tokenizer: Tokenizer for generating token IDs
        needle_len: Length of needle sequence (tokens)
        ctx_chunks: Number of context chunks
        ctx_chunk_len: Length of each chunk
        num_train/num_val: Dataset sizes
        seed: Random seed
        use_query: If True, append query segment and target is needle retrieval

    Returns:
        train_ds, val_ds: Datasets with 'input_ids', 'labels', 'needle_pos', 'needle', 'retrieval_start'
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    vocab_size = tokenizer.vocab_size
    total_len = ctx_chunks * ctx_chunk_len

    # Query tokens (use simple numeric IDs to avoid tokenizer dependency)
    # Format: <SEP> QUERY: RETURN NEEDLE ->
    sep_token = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 2  # Fallback to ID 2
    query_tokens = [sep_token] + [100, 101, 102, 103, 104]  # "QUERY: RETURN NEEDLE ->" (placeholder IDs)
    query_len = len(query_tokens)

    def generate_example():
        # Generate random needle (avoid special tokens)
        needle = np_rng.randint(100, vocab_size - 100, size=needle_len).tolist()

        # Random position for needle (not at edges)
        min_pos = ctx_chunk_len
        max_pos = total_len - ctx_chunk_len - needle_len

        if max_pos <= min_pos:
            # Fallback: just ensure needle fits in sequence
            min_pos = 0
            max_pos = total_len - needle_len
            if max_pos <= min_pos:
                raise ValueError(
                    f"Sequence too short for needle: total_len={total_len}, needle_len={needle_len}. "
                    f"Need at least {needle_len + 1} tokens."
                )

        needle_pos = rng.randint(min_pos, max_pos)

        # Generate random context (haystack)
        context_before = np_rng.randint(100, vocab_size - 100, size=needle_pos).tolist()
        context_after_len = total_len - needle_pos - needle_len
        context_after = np_rng.randint(100, vocab_size - 100, size=context_after_len).tolist()

        if use_query:
            # RETRIEVAL MODE: Model sees context+query, must generate needle
            # Format: [context_before] + [needle] + [context_after] + <SEP> QUERY: RETURN NEEDLE -> [NEEDLE]
            context_ids = context_before + needle + context_after  # needle embedded in context

            # Input: context + query + needle (model must generate this)
            input_ids = context_ids + query_tokens + needle

            # Labels: -100 (ignore) for context+query, real needle for generation target
            labels = [-100] * (len(context_ids) + len(query_tokens)) + needle
            retrieval_start = len(context_ids) + len(query_tokens)  # Position where needle generation begins
        else:
            # CLASSIC MODE: Causal LM (predict next token)
            input_ids = context_before + needle + context_after
            labels = input_ids.copy()
            retrieval_start = -1  # No retrieval target

        return {
            'input_ids': input_ids,
            'labels': labels,
            'needle_pos': needle_pos,
            'needle': needle,
            'retrieval_start': retrieval_start,  # Where to expect retrieved needle
            'attention_mask': [1] * len(input_ids)
        }

    # Generate train/val sets
    train_examples = [generate_example() for _ in range(num_train)]
    val_examples = [generate_example() for _ in range(num_val)]

    # Convert to HF Dataset format
    from datasets import Dataset
    train_ds = Dataset.from_list(train_examples)
    val_ds = Dataset.from_list(val_examples)

    mode_str = "Retrieval (with query)" if use_query else "Causal LM (classic)"
    print(f"[NEEDLE] Generated dataset ({mode_str}):")
    print(f"[NEEDLE]   Train: {len(train_ds)} examples")
    print(f"[NEEDLE]   Val:   {len(val_ds)} examples")
    print(f"[NEEDLE]   Context length: {total_len} tokens ({ctx_chunks} chunks × {ctx_chunk_len})")
    print(f"[NEEDLE]   Needle length: {needle_len} tokens")
    if use_query:
        print(f"[NEEDLE]   Query segment: <SEP> QUERY: RETURN NEEDLE -> (retrieval mode)")
        print(f"[NEEDLE]   Total sequence length: {total_len + query_len + needle_len} tokens (context + query + target)")
        print(f"[NEEDLE]   Target: Exact {needle_len}-token needle generation")
    else:
        print(f"[NEEDLE]   Task: Predict next token (needle embedded in long context)")

    return train_ds, val_ds

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-afrb', type=int, default=1)
    p.add_argument('--alpha', type=float, default=0.04)
    p.add_argument('--gamma', type=float, default=0.20)
    p.add_argument('--omega', type=float, default=6.0)
    p.add_argument('--lr', type=float, default=2e-6)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--seq', type=int, default=256)
    p.add_argument('--bs', type=int, default=8)
    p.add_argument('--ga', type=int, default=4, help='gradient accumulation')
    p.add_argument('--seed', type=int, default=41)
    p.add_argument('--save', required=True)
    p.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    p.add_argument('--eval-only', action='store_true', help='Eval only, no training')
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--grad-clip', type=float, default=1.0)
    # DRC (Dynamic Resonance Cascade) parameters
    p.add_argument('--cascade-kappa', type=float, default=0.0, help='Gamma decay: gamma_i = gamma_0 * exp(-kappa*(i-1))')
    p.add_argument('--cascade-lambda', type=float, default=0.0, help='Alpha decay: alpha_i = alpha_0 * exp(-lambda*(i-1))')
    p.add_argument('--phase-delta', type=float, default=0.0, help='Phase offset: phi_i = (i-1)*delta')
    p.add_argument('--omega-delta', type=float, default=0.0, help='Omega offset: omega_i = omega_0 + (i-1)*delta')
    # Stillness (EMA low-pass filter on phase term)
    p.add_argument('--stillness-ema', type=float, default=0.0, help='EMA coefficient rho for per-block low-pass on phase term (0=OFF)')
    p.add_argument('--stillness-warm', type=int, default=0, help='Warmup steps before enabling stillness (EMA)')
    p.add_argument('--stillness-floor', type=float, default=0.0, help='Minimum mixing gain after EMA (0..1)')
    # Stagger (per-block activation delay + smooth ramp)
    p.add_argument('--block-warm-delta', type=int, default=0, help='Per-block warmup offset (steps between successive block activations, 0=OFF)')
    p.add_argument('--block-ramp', type=int, default=300, help='Ramp duration for smooth activation of each block after warmup')
    p.add_argument('--phase-ramp', type=int, default=300, help='Ramp duration for phase glide (smooth φ offset growth)')
    # Adaptive-OMEGA (learnable frequency per block)
    p.add_argument('--adaptive-omega', action='store_true', help='Enable learnable omega (model optimizes frequency)')
    p.add_argument('--adaptive-omega-lr', type=float, default=1e-7, help='Learning rate for omega parameters')
    p.add_argument('--adaptive-omega-interval', type=int, default=10, help='Update omega every N steps (0=every step)')
    p.add_argument('--omega-min', type=float, default=5.6, help='Minimum omega value (clamped)')
    p.add_argument('--omega-max', type=float, default=6.4, help='Maximum omega value (clamped)')
    # Phase-Vector Memory (PVM)
    p.add_argument('--enable-pvm', action='store_true', help='Enable Phase-Vector Memory (O(d) persistent memory)')
    p.add_argument('--pvm-alpha', type=float, default=0.12, help='PVM write coefficient (how much new info to absorb)')
    p.add_argument('--pvm-beta', type=float, default=0.88, help='PVM retention coefficient (how much old memory to keep)')
    p.add_argument('--pvm-gate-init', type=float, default=-2.0, help='PVM gate initial logit (sigmoid -> mixing strength)')
    p.add_argument('--pvm2emb-ridge-recalib', type=int, default=0, help='If > 0, perform ridge re-calibration every N steps (default: 0 = disabled)')
    # Phase Lattice Memory (PLM)
    p.add_argument('--enable-plm', action='store_true', help='Enable Phase Lattice Memory (2D grid with spatial interference)')
    p.add_argument('--plm-grid-x', type=int, default=4, help='PLM grid width (Gx)')
    p.add_argument('--plm-grid-y', type=int, default=4, help='PLM grid height (Gy)')
    p.add_argument('--plm-alpha', type=float, default=0.10, help='PLM write coefficient')
    p.add_argument('--plm-beta', type=float, default=0.90, help='PLM retention coefficient')
    p.add_argument('--plm-omega', type=float, default=6.0, help='PLM rotation frequency')
    p.add_argument('--plm-kappa', type=float, default=0.05, help='PLM Laplacian coupling strength')
    p.add_argument('--plm-gate-init', type=float, default=-2.0, help='PLM gate initial logit')
    # Attention control (Phase 7b - Pure PVM)
    p.add_argument('--disable-attn', action='store_true', help='Disable attention layers (replace with bypass for pure PVM mode)')
    # Synthetic tasks (Phase 7c - Memory tests)
    p.add_argument('--task', type=str, default='language', choices=['language', 'needle', 'copy'], help='Task type: language (wikitext), needle (retrieval), copy (sequence copy)')
    p.add_argument('--needle-len', type=int, default=16, help='Needle length (tokens) for needle-in-haystack task')
    p.add_argument('--ctx-chunks', type=int, default=64, help='Number of context chunks (total_len = chunks * chunk_len)')
    p.add_argument('--ctx-chunk-len', type=int, default=512, help='Length of each context chunk')
    # Needle improvements (Phase 8 - Query-based retrieval)
    p.add_argument('--needle-query', action='store_true', help='Use query segment for explicit retrieval (vs causal LM)')
    p.add_argument('--lambda-retrieval', type=float, default=0.5, help='Retrieval loss weight: L = L_lm + λ * L_retrieval')
    p.add_argument('--phase-alignment-weight', type=float, default=0.0, help='Phase alignment L2 regularization weight (0=disabled, typical: 0.001-0.01)')
    # InfoNCE contrastive loss (PVM->Embedding alignment)
    p.add_argument('--infonce-weight', type=float, default=0.0, help='InfoNCE contrastive loss weight (0=disabled, E1 uses 0.3)')
    p.add_argument('--infonce-negatives', type=int, default=128, help='Number of negative samples for InfoNCE (default: 128)')
    p.add_argument('--infonce-tau', type=float, default=0.08, help='InfoNCE temperature parameter (default: 0.08, REVERTED from 0.07)')
    p.add_argument('--sir-dim', type=int, default=512, help='SIR (Shared Intermediate Representation) dimension (default: 512, MUST be 512 for current InfoNCE)')
    p.add_argument('--mem-acc-exact', action='store_true', help='Compute exact match accuracy for needle retrieval')
    p.add_argument('--eval-soft-topk', type=int, default=5, help='Top-k for soft needle evaluation (default: 5)')
    p.add_argument('--curriculum-learning', action='store_true', help='Gradually increase context length (512->1024->2048->4096->8192)')
    p.add_argument('--curriculum-steps-per-stage', type=int, default=500, help='Steps per curriculum stage (default: 500)')
    # Query-addressed readout (Phase 9 - Content-based retrieval)
    p.add_argument('--readout-from', type=str, default='both', choices=['pvm', 'plm', 'both', 'none'], help='Readout source: pvm (temporal), plm (spatial), both, or none')
    p.add_argument('--readout-topk', type=int, default=0, help='Top-k retrieval (0=use all, >0=top-k softmax)')
    p.add_argument('--readout-scale', type=float, default=1.0, help='Readout residual scale factor')
    p.add_argument('--readout-window', type=int, default=16, help='Query key window size (last N tokens)')
    p.add_argument('--needle-query-mode', type=str, default='mean_tail', choices=['mean_tail', 'mlp_query', 'phase_weighted', 'learned'], help='Query construction mode for needle task. Use "learned" for trainable MLP to correct phase offset. (mean_tail = povprečje zadnjih tokenov, mlp_query = mali MLP nad tail, phase_weighted = fazno uteženo povprečje, learned = trainable MLP for phase correction).')
    p.add_argument('--readout-head', type=str, default='shared', choices=['shared', 'separate'], help='shared=use LM head (tied), separate=dedicated readout projection')
    p.add_argument('--log-readout-stats', action='store_true', help='Log readout statistics (cos-sim, entropy, pointer-norm)')
    p.add_argument('--enable-rca', action='store_true', help='Enable RCA (Resonant Coupling Analysis) metrics for PVM/PLM diagnostics (β, ω estimation)')
    p.add_argument('--hard-pointer-scale', type=float, default=0.0, help='DEBUG: Replace logits with pointer*scale (0=disabled, >0=diagnostic mode)')
    # E-FIX arguments
    p.add_argument('--freeze-readout-steps', type=int, default=0, help='E-FIX: Freeze readout projection (sir_compress + readout_head_proj) for first N steps to stabilize PVM (default: 0=no freeze)')

    # KISS Ridge Alignment arguments (CLAUDE2.md - Phase-space alignment)
    p.add_argument('--kiss-ridge-calib', action='store_true', help='KISS: Enable Ridge regression pre-calibration for pvm2emb alignment (CLAUDE2.md)')
    p.add_argument('--kiss-ridge-max-pairs', type=int, default=512, help='KISS Ridge: Maximum (Z,E) pairs to collect for calibration (default: 512)')
    p.add_argument('--kiss-ridge-l2', type=float, default=1e-3, help='KISS Ridge: L2 regularization factor λ (default: 1e-3)')
    p.add_argument('--kiss-ridge-unfreeze-step', type=int, default=0, help='KISS Ridge: Step to unfreeze pvm2emb for fine-tune (0=stay frozen, >0=unfreeze at N)')

    # T2/PCM buffers for phase memory
    p.add_argument('--t2-enable', action='store_true', help='Enable T2-decay buffer for phase memory')
    p.add_argument('--t2-steps', type=int, default=1500, help='T2 decay time constant (in training steps) - only used if t2-mode=exp')
    p.add_argument('--t2-mode', type=str, default='exp', choices=['exp', 'resonant'], help='T2 decay mode: exp=monotonic exp(-t/T2), resonant=EIW breathing decay')
    p.add_argument('--t2-k', type=float, default=0.001, help='Resonant T2: base decay rate (exp(-k*t) term)')
    p.add_argument('--t2-alpha', type=float, default=0.08, help='Resonant T2: oscillation amplitude')
    p.add_argument('--t2-omega', type=float, default=6.0, help='Resonant T2: log-frequency (ISM/EIW principle)')
    p.add_argument('--t2-phi', type=float, default=1.0472, help='Resonant T2: phase offset (~π/3)')
    p.add_argument('--pcm-enable', action='store_true', help='Enable PCM gate (amorphous/crystalline phase switching)')
    p.add_argument('--pcm-gate-init', type=float, default=0.5, help='Initial PCM gate value (sigmoid input)')

    # Phase alignment (CMB principle from ISM/EIW)
    p.add_argument('--phase-align', action='store_true', help='Enable learnable phase alignment (CMB principle)')
    p.add_argument('--phase-align-l2', type=float, default=1e-4, help='L2 regularization strength for phase alignment parameter')
    # Memory optimization
    p.add_argument('--gradient-checkpointing', action='store_true', help='Enable gradient checkpointing (saves memory, slower training)')
    p.add_argument('--skip-phase-metrics', action='store_true', help='Skip phase coherence metrics during eval (saves memory, use if OOM)')
    # DataLoader optimization (GPU utilization)
    p.add_argument('--num-workers', type=int, default=2, help='DataLoader workers (parallel prefetch, 0=single-threaded)')
    p.add_argument('--prefetch', '--prefetch-factor', type=int, default=2, dest='prefetch_factor', help='Batches to prefetch per worker')
    p.add_argument('--pin-memory', action='store_true', help='Pin memory for faster GPU transfer (default: True if CUDA available)')
    # Phase curvature diagnostics (CLAUDE2.md Section 2.2)
    p.add_argument('--phase-curvature-metrics', action='store_true', help='Enable PVM phase-curvature diagnostics during needle eval (Phase 2).')
    args = p.parse_args()

    # Auto-enable pin_memory if CUDA available and not explicitly disabled
    if not args.pin_memory and torch.cuda.is_available():
        args.pin_memory = True  # Default to True on GPU systems

    # KISS-RIDGE: Auto-enable unfreezing if calibration is used and unfreeze_step not explicitly set
    # WHY: KISS Ridge initializes pvm2emb statistically at step 0-1, then should fine-tune via InfoNCE
    # Default behavior: unfreeze at step 100 if InfoNCE is active and KISS calibration is used
    if args.kiss_ridge_calib and args.infonce_weight > 0 and args.kiss_ridge_unfreeze_step == 0:
        args.kiss_ridge_unfreeze_step = 100  # Smart default: unfreeze after 100 steps
        print(f"[KISS-RIDGE-AUTO] Auto-enabled pvm2emb unfreezing at step 100 (InfoNCE + KISS calibration active)")
        print(f"[KISS-RIDGE-AUTO] Override with --kiss-ridge-unfreeze-step N to change unfreeze point")

    # Eval-only mode if steps=0
    if args.steps == 0:
        args.eval_only = True

    # Clear GPU memory before starting (safety for long-context runs)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[MEMORY] Cleared GPU cache before start")

    # Set all seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Sanity checks
    print(f"[SETUP] Device: {device}")
    if torch.cuda.is_available():
        print(f"[SETUP] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[SETUP] BF16: {torch.cuda.is_bf16_supported()}")
    else:
        print("[SETUP] BF16: False (CPU mode)")
    print(f"[SETUP] Mode: {'EVAL-ONLY' if args.eval_only else 'TRAINING'}")

    # Load model
    baseline_path = Path('reports/baseline_metrics.json')
    with open(baseline_path) as f:
        baseline = json.load(f)

    model = AutoModelForCausalLM.from_pretrained(
        baseline['model'],
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map=device,
        use_cache=False  # CRITICAL: disable KV cache for gradient checkpointing
    )
    tokenizer = AutoTokenizer.from_pretrained(baseline['model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Enable gradient checkpointing if requested (memory optimization)
    if args.gradient_checkpointing:
        print(f"[MEMORY] Enabling gradient checkpointing (trade-off: saves memory, ~20% slower)")
        model.gradient_checkpointing_enable()
        # Also disable dropout for determinism (optional but recommended)
        model.config.use_cache = False

    # Inject AFRB with DRC + Stillness + Stagger + Adaptive-OMEGA + PVM + PLM
    model = inject_afrb_adapters(
        model, args.n_afrb, args.alpha, args.omega, args.gamma, device,
        cascade_kappa=args.cascade_kappa,
        cascade_lambda=args.cascade_lambda,
        phase_delta=args.phase_delta,
        omega_delta=args.omega_delta,
        stillness_ema=args.stillness_ema,
        stillness_floor=args.stillness_floor,
        stillness_warm=args.stillness_warm,
        block_warm_delta=args.block_warm_delta,
        block_ramp=args.block_ramp,
        phase_ramp=args.phase_ramp,
        learnable_omega=args.adaptive_omega,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        use_pvm=args.enable_pvm,
        pvm_alpha=args.pvm_alpha,
        pvm_beta=args.pvm_beta,
        pvm_gate_init=args.pvm_gate_init,
        use_plm=args.enable_plm,
        plm_grid_x=args.plm_grid_x,
        plm_grid_y=args.plm_grid_y,
        plm_alpha=args.plm_alpha,
        plm_beta=args.plm_beta,
        plm_omega=args.plm_omega,
        plm_kappa=args.plm_kappa,
        plm_gate_init=args.plm_gate_init,
        t2_enable=args.t2_enable,
        t2_steps=args.t2_steps,
        t2_mode=args.t2_mode,
        t2_k=args.t2_k,
        t2_alpha=args.t2_alpha,
        t2_omega=args.t2_omega,
        t2_phi=args.t2_phi,
        pcm_enable=args.pcm_enable,
        pcm_gate_init=args.pcm_gate_init,
        needle_query_mode=getattr(args, 'needle_query_mode', 'mean_tail'),
        readout_window=args.readout_window
    )

    # KISS-Ridge for ALL modes (design specification)
    # pvm2emb: PVM -> Embedding FROZEN projection (2048 -> 2048)
    # This provides direct statistical alignment from PVM space to vocabulary space
    # NO learning - initialized via Ridge regression and kept frozen!
    import torch.nn as nn
    vocab_size = model.config.vocab_size
    hidden_size = model.config.hidden_size
    # Use same dtype as model to avoid dtype mismatch errors
    model_dtype = next(model.parameters()).dtype

    model.pvm2emb = nn.Linear(hidden_size, hidden_size, bias=False).to(device=device, dtype=model_dtype)
    model.pvm2emb.requires_grad_(False)  # FROZEN!
    print(f"[KISS] pvm2emb FROZEN created for ALL modes (shared/separate)")
    print(f"[KISS] PVM({hidden_size}) -> Emb({hidden_size}) FROZEN projection")
    print(f"[KISS] This layer will be initialized via Ridge regression (step 0-1)")
    print(f"[KISS] NO training - pure statistical alignment!")

    # SIR-COMPRESS: Always create for InfoNCE (CLAUDE2.md SIR-COMPRESS fix)
    # This enables InfoNCE contrastive learning even in 'shared' mode
    sir_dim = getattr(args, 'sir_dim', 512)  # Default: 512 (matches InfoNCE e_pos slicing)

    # FORENSIC PATCH A: sir_dim guard
    if sir_dim != 512:
        raise ValueError(
            f"Current InfoNCE implementation assumes sir_dim=512, got sir_dim={sir_dim}. "
            "Either set --sir-dim 512 or re-enable the e2sir alignment mapping."
        )

    model.sir_compress = nn.Linear(hidden_size, sir_dim, bias=False).to(device=device, dtype=model_dtype)
    # Xavier init for stable training
    nn.init.xavier_uniform_(model.sir_compress.weight)
    # Explicitly mark as trainable (before freeze policy)
    model.sir_compress.weight.requires_grad = True
    print(f"[SIR-COMPRESS] Created for InfoNCE: PVM({hidden_size}) -> SIR({sir_dim})")
    print(f"[SIR-COMPRESS] Trainable params: {sum(p.numel() for p in model.sir_compress.parameters()):,}")

    # --- LEARNED QUERY HEAD (Phase 10 Fix) ---
    if getattr(args, 'needle_query_mode', '') == 'learned':
        print(f"[SETUP] Initializing LEARNED Query Head (Trainable MLP)")
        model.learned_query_head = LearnedQueryHead(
            dim=hidden_size,
            window=args.readout_window
        ).to(device=device, dtype=model_dtype)
        # Ensure it trains!
        for p in model.learned_query_head.parameters():
            p.requires_grad = True

        print(f"[SETUP] Learned Query Head: {sum(p.numel() for p in model.learned_query_head.parameters())} params")
    else:
        # Ensure the attribute exists (set to None if not using learned mode)
        model.learned_query_head = None

    # Create separate readout head if requested
    if args.readout_head == 'separate':
        # LEVEL 2 TRANSLATOR: SIR (Shared Intermediate Representation)
        # PVM (2048D) -> SIR (512D) -> Vocab (32000D)
        sir_dim_separate = 512

        # Step 1: PVM -> SIR compression with normalization and nonlinearity
        # DESIGN-DOC: Override sir_compress with more complex version for 'separate' mode
        model.sir_compress = nn.Sequential(
            nn.Linear(hidden_size, sir_dim_separate, bias=False),
            nn.LayerNorm(sir_dim_separate),
            nn.GELU()
        ).to(device=device, dtype=model_dtype)
        print(f"[SIR-COMPRESS] Overridden for 'separate' mode with LayerNorm+GELU: {hidden_size} -> {sir_dim_separate}")

        # [E-ALIGN DISABLED] Embedding -> SIR learnable projection (NO slicing!)
        # [E-ALIGN DISABLED] This replaces [:512] slicing with proper learned alignment
        # model.e2sir = nn.Linear(hidden_size, sir_dim, bias=False).to(device=device, dtype=model_dtype)
        # print(f"[E-ALIGN] Created e2sir: Embedding({hidden_size}) -> SIR({sir_dim}) learnable projection")

        # Step 2: SIR -> Vocab projection (NO weight-tie - let it learn via InfoNCE!)
        model.readout_head_proj = nn.Linear(sir_dim_separate, vocab_size, bias=False).to(device=device, dtype=model_dtype)
        # Random initialization - InfoNCE will align this to embedding space

        print(f"[READOUT-HEAD] Created SIR translator: {hidden_size} -> {sir_dim_separate} -> {vocab_size} (dtype={model_dtype})")
        print(f"[TRANSLATOR] Level 3: PVM->SIR->Vocab (NO weight-tie, learns via InfoNCE)")
        print(f"[TRANSLATOR] SIR = Shared Intermediate Representation (common language!)")

    # Phase 7b: Disable attention (Pure PVM mode)
    if args.disable_attn:
        model = disable_attention_layers(model)

    # ============================================================================
    # PHASE 2: FREEZE-BACKBONE STRATEGY (INTENTIONAL)
    # ============================================================================
    # The transformer backbone (all pretrained LM layers) is fully frozen.
    # Only AFRB adapters + PVM-related readout layers are trainable.
    # This is intentional: AFBR/PVM resonance learns on top of mostly-random LM,
    # establishing phase-space representations before any language fine-tuning.
    # ============================================================================

    # Freeze backbone, train only adapters
    for param in model.parameters():
        param.requires_grad = False
    for param in model.afrb_adapters.parameters():
        param.requires_grad = True
    # Also train separate readout head if it exists
    if hasattr(model, 'readout_head_proj'):
        for param in model.readout_head_proj.parameters():
            param.requires_grad = True
    # Train SIR compression layer if it exists
    if hasattr(model, 'sir_compress'):
        for param in model.sir_compress.parameters():
            param.requires_grad = True
    # [E-ALIGN DISABLED] Train e2sir embedding projection
    # if hasattr(model, 'e2sir'):
    #     for param in model.e2sir.parameters():
    #         param.requires_grad = True

    # E-FIX D) Freeze readout translator initially if requested
    if args.freeze_readout_steps > 0:
        print(f"[E-FIX-FREEZE] Freezing readout translator (sir_compress + e2sir + readout_head_proj) for first {args.freeze_readout_steps} steps")
        print(f"[E-FIX-FREEZE] This allows PVM to stabilize before training projection")
        if hasattr(model, 'sir_compress'):
            for param in model.sir_compress.parameters():
                param.requires_grad = False
            print(f"[E-FIX-FREEZE] sir_compress: FROZEN (will unfreeze at step {args.freeze_readout_steps})")
        # [E-ALIGN DISABLED]
        # if hasattr(model, 'e2sir'):
        #     for param in model.e2sir.parameters():
        #         param.requires_grad = False
        if hasattr(model, 'readout_head_proj'):
            for param in model.readout_head_proj.parameters():
                param.requires_grad = False
            print(f"[E-FIX-FREEZE] readout_head_proj: FROZEN (will unfreeze at step {args.freeze_readout_steps})")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SETUP] Trainable: {trainable:,} params")
    print(f"[PHASE2] Backbone frozen, AFBR/PVM adapters trainable: {trainable:,} params")

    # DIAGNOSTIC: Check sir_compress and pvm2emb trainability at setup
    if hasattr(model, 'sir_compress'):
        sir_trainable = sum(p.numel() for p in model.sir_compress.parameters() if p.requires_grad)
        sir_total = sum(p.numel() for p in model.sir_compress.parameters())
        sir_grad_status = any(p.requires_grad for p in model.sir_compress.parameters())
        print(f"[SETUP-DEBUG] sir_compress: {sir_trainable}/{sir_total} params trainable (requires_grad={sir_grad_status})")
    if hasattr(model, 'pvm2emb'):
        pvm2emb_trainable = sum(p.numel() for p in model.pvm2emb.parameters() if p.requires_grad)
        pvm2emb_total = sum(p.numel() for p in model.pvm2emb.parameters())
        pvm2emb_grad_status = any(p.requires_grad for p in model.pvm2emb.parameters())
        print(f"[SETUP-DEBUG] pvm2emb: {pvm2emb_trainable}/{pvm2emb_total} params trainable (requires_grad={pvm2emb_grad_status})")
    if hasattr(model, 'readout_head_proj'):
        rh_trainable = sum(p.numel() for p in model.readout_head_proj.parameters() if p.requires_grad)
        rh_total = sum(p.numel() for p in model.readout_head_proj.parameters())
        rh_grad_status = any(p.requires_grad for p in model.readout_head_proj.parameters())
        print(f"[SETUP-DEBUG] readout_head_proj: {rh_trainable}/{rh_total} params trainable (requires_grad={rh_grad_status})")

    # Print DRC profile if using cascade
    if args.n_afrb > 1 and (args.cascade_kappa != 0.0 or args.cascade_lambda != 0.0 or args.phase_delta != 0.0 or args.omega_delta != 0.0):
        print(f"[DRC] Dynamic Resonance Cascade enabled:")
        print(f"[DRC]   kappa={args.cascade_kappa:.3f}, lambda={args.cascade_lambda:.3f}, phase_delta={args.phase_delta:.3f}, omega_delta={args.omega_delta:.3f}")
        for i, adapter in enumerate(model.afrb_adapters, 1):
            print(f"[DRC]   Block {i}: alpha={adapter.alpha.item():.4f}, gamma_init={torch.sigmoid(adapter.gamma_raw).item():.4f}, phi_base={adapter.phi_base.item():.4f}, omega={adapter.phase_embed.current_omega().item():.2f}")

    # Print Stillness info if enabled
    if args.stillness_ema > 0.0:
        print(f"[STILLNESS] EMA low-pass filter enabled:")
        print(f"[STILLNESS]   ema_coef={args.stillness_ema:.3f}, warmup={args.stillness_warm} steps, floor={args.stillness_floor:.3f}")

    # Print Stagger info if enabled
    if args.block_warm_delta > 0:
        print(f"[STAGGER] Per-block activation timing:")
        print(f"[STAGGER]   warm_delta={args.block_warm_delta} steps, ramp={args.block_ramp} steps, phase_ramp={args.phase_ramp} steps")
        for i, adapter in enumerate(model.afrb_adapters, 1):
            warm_i = adapter.warm_i
            print(f"[STAGGER]   Block {i}: activates at step={warm_i}, fully active at step={warm_i + args.block_ramp}")

    # Print Adaptive-OMEGA info if enabled
    if args.adaptive_omega:
        print(f"[ADAPTIVE-OMEGA] Learnable frequency enabled:")
        print(f"[ADAPTIVE-OMEGA]   lr={args.adaptive_omega_lr}, interval={args.adaptive_omega_interval} steps")
        print(f"[ADAPTIVE-OMEGA]   bounds=[{args.omega_min:.2f}, {args.omega_max:.2f}]")
        print(f"[ADAPTIVE-OMEGA]   Initial omega values:")
        for i, adapter in enumerate(model.afrb_adapters, 1):
            omega_init = adapter.phase_embed.current_omega().item()
            print(f"[ADAPTIVE-OMEGA]     Block {i}: omega={omega_init:.4f}")

    # Print PVM info if enabled
    if args.enable_pvm:
        print(f"[PVM] Phase-Vector Memory enabled:")
        print(f"[PVM]   alpha={args.pvm_alpha:.3f} (write), beta={args.pvm_beta:.3f} (retention)")
        print(f"[PVM]   gate_init={args.pvm_gate_init:.2f} (sigmoid->{torch.sigmoid(torch.tensor(args.pvm_gate_init)):.3f})")
        print(f"[PVM]   Memory: O(d) persistent state per block")
        print(f"[PVM]   Blocks with PVM: {sum(1 for a in model.afrb_adapters if hasattr(a, 'pvm') and a.pvm is not None)}/{len(model.afrb_adapters)}")
        # T2/PCM info for PVM
        if args.t2_enable or args.pcm_enable:
            features = []
            if args.t2_enable:
                features.append(f"T2-decay (steps={args.t2_steps})")
            if args.pcm_enable:
                features.append(f"PCM (gate_init={args.pcm_gate_init})")
            print(f"[PVM]   PhaseBuffer: {', '.join(features)}")
        # Query configuration for needle task
        needle_query_mode = getattr(args, 'needle_query_mode', 'mean_tail')
        print(f"[QUERY-CONFIG] needle_query_mode={needle_query_mode}, window={args.readout_window}")
        print(f"[QUERY-CONFIG] *** PER-SAMPLE QUERIES ACTIVE (KISS-Alignment Fix 2025-11-13) ***")
        if needle_query_mode == 'phase_weighted':
            print(f"[QUERY-CONFIG] Phase-weighted query: weights tokens by local phase velocity (Delta-hidden)")
        elif needle_query_mode == 'mlp_query':
            print(f"[QUERY-CONFIG] MLP query: small MLP over tail tokens")
            print(f"[QUERY-CONFIG] Expected improvement: curvature peak offset closer to needle")

    if args.enable_plm:
        print(f"[PLM] Phase Lattice Memory enabled:")
        print(f"[PLM]   Grid: {args.plm_grid_x}×{args.plm_grid_y} ({args.plm_grid_x * args.plm_grid_y} cells)")
        print(f"[PLM]   alpha={args.plm_alpha:.3f} (write), beta={args.plm_beta:.3f} (retention)")
        print(f"[PLM]   omega={args.plm_omega:.2f}, kappa={args.plm_kappa:.3f} (Laplacian coupling)")
        print(f"[PLM]   gate_init={args.plm_gate_init:.2f} (sigmoid->{torch.sigmoid(torch.tensor(args.plm_gate_init)):.3f})")
        print(f"[PLM]   Memory: O(Gx·Gy·Dcell) 2D lattice per block")
        print(f"[PLM]   Blocks with PLM: {sum(1 for a in model.afrb_adapters if hasattr(a, 'plm') and a.plm is not None)}/{len(model.afrb_adapters)}")
        # T2/PCM info for PLM
        if args.t2_enable or args.pcm_enable:
            features = []
            if args.t2_enable:
                features.append(f"T2-decay (steps={args.t2_steps})")
            if args.pcm_enable:
                features.append(f"PCM (gate_init={args.pcm_gate_init})")
            print(f"[PLM]   PhaseBuffer: {', '.join(features)}")

    # Enable PVM trajectory recording for phase curvature diagnostics (CLAUDE2.md Section 2.2)
    if args.phase_curvature_metrics and args.enable_pvm:
        print("[PHASE-CURV] Enabling PVM trajectory recording for eval")
        pvm_count = 0
        for adapter in model.afrb_adapters:
            if hasattr(adapter, 'pvm') and adapter.pvm is not None:
                adapter.pvm.record_traj = True
                pvm_count += 1
        print(f"[PHASE-CURV] Trajectory recording enabled for {pvm_count} PVM modules")
    elif args.phase_curvature_metrics and not args.enable_pvm:
        print("[PHASE-CURV] WARNING: --phase-curvature-metrics requires --enable-pvm to be set")

    # Load data (task-dependent)
    if args.task == 'needle':
        # Synthetic needle-in-haystack dataset
        print(f"[SETUP] Task: Needle-in-haystack (memory retrieval)")

        # Override seq to match full context length for needle task
        needle_ctx_len = args.ctx_chunks * args.ctx_chunk_len
        if args.seq < needle_ctx_len:
            print(f"[NEEDLE] WARNING: --seq {args.seq} < context {needle_ctx_len}, overriding to {needle_ctx_len}")
            args.seq = needle_ctx_len

        train_ds, val_ds = generate_needle_dataset(
            tokenizer,
            needle_len=args.needle_len,
            ctx_chunks=args.ctx_chunks,
            ctx_chunk_len=args.ctx_chunk_len,
            num_train=1000,
            num_val=100,
            seed=args.seed,
            use_query=args.needle_query  # Enable query-based retrieval
        )
        # Log needle query mode after dataset generation
        if args.needle_query:
            print(f"[NEEDLE] Query mode: {getattr(args, 'needle_query_mode', 'mean_tail')}")
        # No tokenization needed (already tokenized)
        def tokenize_fn(ex):
            return ex
    elif args.task == 'copy':
        # TODO: Copy task (sequence reproduction)
        raise NotImplementedError("Copy task not yet implemented")
    else:
        # Language modeling (wikitext)
        print(f"[SETUP] Task: Language modeling (wikitext)")
        data_dir = Path(__file__).parent.parent / 'data'
        train_ds = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=str(data_dir), split='train')
        val_ds = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=str(data_dir), split='validation')

        def tokenize_fn(ex):
            tok = tokenizer(ex['text'], truncation=True, max_length=args.seq, padding='max_length')
            tok['labels'] = tok['input_ids'].copy()
            return tok

        train_ds = train_ds.filter(lambda x: x['text'] and len(x['text'].strip()) > 0)
        train_ds = train_ds.select(range(min(5000, len(train_ds))))
        val_ds = val_ds.filter(lambda x: x['text'] and len(x['text'].strip()) > 0)
        val_ds = val_ds.select(range(min(500, len(val_ds))))

    # Process datasets (tokenize/map)
    if args.task != 'needle':
        # For language task: tokenize and remove text columns
        train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
        val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names)
    else:
        # For needle task: already tokenized, keep all columns (including needle_pos for eval)
        pass

    # Training args (ROBUSTNI defaults)
    training_args = TrainingArguments(
        output_dir=str(save_dir),
        num_train_epochs=1,
        max_steps=args.steps if not args.eval_only else 1,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.ga,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.grad_clip,
        logging_steps=20,  # Log metrics every 20 steps
        logging_first_step=True,  # Log first step
        save_steps=500,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=args.gradient_checkpointing,  # Memory optimization
        dataloader_num_workers=args.num_workers,  # Parallel prefetch
        dataloader_pin_memory=args.pin_memory,    # Fast GPU transfer
        dataloader_prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        dataloader_persistent_workers=(args.num_workers > 0),  # Keep workers alive
        remove_unused_columns=False,  # CRITICAL: Preserve custom fields like retrieval_start
        report_to='none',
        disable_tqdm=True,  # FORCE disable Trainer's tqdm
    )

    # CRITICAL: Add custom arguments to TrainingArguments object
    # (TrainingArguments only contains standard HF parameters, so we manually attach custom ones)
    # These are needed in AdaptiveOmegaTrainer.compute_loss() for readout activation
    training_args.task = args.task
    training_args.needle_query = args.needle_query
    training_args.lambda_retrieval = args.lambda_retrieval
    training_args.readout_from = args.readout_from
    training_args.readout_scale = args.readout_scale
    training_args.readout_topk = args.readout_topk
    training_args.readout_window = args.readout_window
    training_args.readout_head = args.readout_head
    training_args.log_readout_stats = args.log_readout_stats
    training_args.hard_pointer_scale = args.hard_pointer_scale
    training_args.phase_alignment_weight = args.phase_alignment_weight

    # Sanity check: Print GPU device info
    print(f"[DATALOADER] Workers={args.num_workers}, Pin={args.pin_memory}, Prefetch={args.prefetch_factor if args.num_workers > 0 else 'N/A'}")
    print(f"[DATALOADER] Expected behavior: CPU workers prefetch batches -> GPU receives pinned tensors -> non-blocking transfer")

    # Use custom data collator for needle task (to preserve retrieval_start field)
    data_collator = None
    if args.task == 'needle' and args.needle_query:
        data_collator = NeedleDataCollator(tokenizer)
        print(f"[COLLATOR] Using NeedleDataCollator (preserves retrieval_start for readout)")

    # CRITICAL: Initialize copy_gate parameters BEFORE Trainer creation
    # (Trainer.__init__ calls create_optimizer() which scans model.named_parameters())
    if args.readout_head == 'separate':
        import torch.nn as nn
        hidden_size = model.config.hidden_size
        # Pointer-Generator: Query-conditioned copy gate (See et al. 2017)
        # p_copy = sigmoid(w^T q + b) where q is query hidden state
        # This allows model to learn: "when I see retrieval query -> use copy mode"
        model.copy_gate_w = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=torch.float32))
        model.copy_gate_b = nn.Parameter(torch.zeros(1, device=device, dtype=torch.float32))
        model.copy_gate_w.requires_grad = True
        model.copy_gate_b.requires_grad = True
        print(f"[COPY-GATE] Initialized query-conditioned copy gate: w[{hidden_size}], b[1]")

    # Use custom Trainer (AdaptiveOmegaTrainer has compute_loss override for needle tasks)
    if args.adaptive_omega:
        trainer = AdaptiveOmegaTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            adaptive_omega_lr=args.adaptive_omega_lr,
            adaptive_omega_interval=args.adaptive_omega_interval,
            infonce_weight=args.infonce_weight,
            infonce_negatives=args.infonce_negatives,
            infonce_tau=args.infonce_tau,
            pvm2emb_ridge_recalib=args.pvm2emb_ridge_recalib,
            callbacks=(
                ([OmegaIntervalCallback(args.adaptive_omega_interval)] if args.adaptive_omega_interval > 0 else []) +
                ([UnfreezeReadoutCallback(args.freeze_readout_steps)] if args.freeze_readout_steps > 0 else []) +
                ([UnfreezePvm2embCallback(args.kiss_ridge_unfreeze_step)] if args.kiss_ridge_unfreeze_step > 0 else [])
            )
        )
    else:
        # FIX: Use AdaptiveOmegaTrainer even without adaptive_omega (for compute_loss override)
        trainer = AdaptiveOmegaTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            adaptive_omega_lr=0.0,  # Disabled
            adaptive_omega_interval=0,  # Disabled
            infonce_weight=args.infonce_weight,
            infonce_negatives=args.infonce_negatives,
            infonce_tau=args.infonce_tau,
            pvm2emb_ridge_recalib=args.pvm2emb_ridge_recalib,
            callbacks=(
                ([UnfreezeReadoutCallback(args.freeze_readout_steps)] if args.freeze_readout_steps > 0 else []) +
                ([UnfreezePvm2embCallback(args.kiss_ridge_unfreeze_step)] if args.kiss_ridge_unfreeze_step > 0 else [])
            )
        )

    # Completely disable tqdm - no progress bars, only manual logging
    trainer.remove_callback(ProgressCallback)
    print("[PROGRESS] Disabled all tqdm progress bars - using manual progress logging")

    # Add gradient norm tracker for CSV logging
    trainer.add_callback(GradNormTrackerCallback(trainer))
    trainer._last_grad_norm = 0.0  # Initialize
    print("[CSV-LOG] Added GradNormTrackerCallback for step_trajectory.csv logging")

    # SANITY CHECK: Verify copy_gate parameters are in optimizer
    if hasattr(model, 'copy_gate_w'):
        # Ensure optimizer is created
        trainer.create_optimizer()
        print(f"[SANITY] Optimizer param groups: {len(trainer.optimizer.param_groups)}")
        for i, pg in enumerate(trainer.optimizer.param_groups):
            print(f"[SANITY]   Group {i}: {len(pg['params'])} params, LR={pg['lr']}")
            # Check if copy_gate is included
            has_copy_gate_w = any(p is model.copy_gate_w for p in pg['params'])
            has_copy_gate_b = any(p is model.copy_gate_b for p in pg['params'])
            if has_copy_gate_w or has_copy_gate_b:
                print(f"[SANITY]   Group {i}: OK - copy_gate_w={has_copy_gate_w}, copy_gate_b={has_copy_gate_b}")

    # [E-ALIGN DISABLED] RIDGE KICKSTART: Initialize e2sir with Procrustes/Ridge alignment
    # [E-ALIGN DISABLED] This provides optimal initial alignment between Embedding space and SIR space
    if False and hasattr(model, 'e2sir') and hasattr(model, 'sir_compress') and args.task == 'needle':
        print(f"\n[RIDGE-KICKSTART] Initializing e2sir with Ridge regression...")
        print(f"[RIDGE-KICKSTART] Collecting calibration data from 2 batches...")

        # Collect calibration data: pairs of (needle_embedding, z_pvm_sir)
        Z_list = []  # PVM→SIR projections [N, 512]
        E_list = []  # Needle embeddings [N, 2048]

        model.eval()  # Eval mode for calibration

        # Simplified approach: Use needle embeddings directly without forward passes
        # We collect 256 calibration pairs from randomly sampled needle tokens
        with torch.no_grad():
            # Sample 256 random tokens from vocabulary as "calibration needles"
            calibration_size = 256
            vocab_size = model.lm_head.weight.shape[0]

            # Sample random token IDs
            calibration_ids = torch.randint(0, vocab_size, (calibration_size,), device=device)

            # Get embeddings for calibration tokens
            e_calib = model.lm_head.weight[calibration_ids]  # [256, 2048]

            # For initial alignment, we use a simple projection through sir_compress
            # (assuming PVM would contain embeddings projected through some transform)
            # Since we don't have actual PVM readouts yet, we use embedding[:512] as "target SIR"
            # This gives e2sir a reasonable starting point
            z_calib = e_calib[:, :512]  # [256, 512] - first 512 dims as target

            E_list = [e for e in e_calib.cpu()]
            Z_list = [z for z in z_calib.cpu()]

            print(f"[RIDGE-KICKSTART] Collected {len(E_list)} calibration pairs")

        if len(Z_list) > 0 and len(E_list) > 0:
            # Stack into matrices
            Z = torch.stack(Z_list)  # [N, 512] - SIR space (source)
            E = torch.stack(E_list)  # [N, 2048] - Embedding space (target)

            print(f"[RIDGE-KICKSTART] Collected {Z.shape[0]} calibration pairs")
            print(f"[RIDGE-KICKSTART] Z (SIR): {Z.shape}, E (Embedding): {E.shape}")

            # Ridge regression: W = (Z^T Z + λI)^{-1} Z^T E
            # We want: e2sir(E) ≈ Z, so we solve: E W ≈ Z
            # Ridge: W = (E^T E + λI)^{-1} E^T Z

            lambda_ridge = 1e-3  # Regularization

            # IMPORTANT: Convert to float32 for Ridge solve (linalg.solve requires matching dtypes)
            E_f32 = E.float()  # [N, 2048] float32
            Z_f32 = Z.float()  # [N, 512] float32
            I = torch.eye(E_f32.shape[1], dtype=torch.float32)  # [2048, 2048] float32

            # Solve: W = (E^T E + λI)^{-1} E^T Z
            EtE = E_f32.T @ E_f32  # [2048, 2048]
            EtE_reg = EtE + lambda_ridge * I  # [2048, 2048]
            EtZ = E_f32.T @ Z_f32  # [2048, 512]

            # Solve linear system
            try:
                W = torch.linalg.solve(EtE_reg, EtZ)  # [2048, 512]

                # Copy to e2sir.weight (which is [512, 2048] - output × input)
                # So we need W.T
                with torch.no_grad():
                    model.e2sir.weight.copy_(W.T.to(device))

                print(f"[RIDGE-KICKSTART] OK Initialized e2sir.weight with Ridge solution")
                print(f"[RIDGE-KICKSTART] Weight shape: {model.e2sir.weight.shape}")

                # Compute alignment quality
                E_test = E_f32[:10]  # Test on first 10 samples
                Z_test = Z_f32[:10]
                with torch.no_grad():
                    Z_pred = model.e2sir(E_test.to(device)).cpu()  # [10, 512]
                    mse = ((Z_pred - Z_test)**2).mean().item()
                    cos_sim = torch.nn.functional.cosine_similarity(Z_pred, Z_test, dim=1).mean().item()

                print(f"[RIDGE-KICKSTART] Alignment quality: MSE={mse:.6f}, CosSim={cos_sim:.4f}")

            except Exception as e:
                print(f"[RIDGE-KICKSTART] WARNING Ridge solve failed: {e}")
                print(f"[RIDGE-KICKSTART] Keeping random initialization")
        else:
            print(f"[RIDGE-KICKSTART] WARNING No calibration data collected, skipping")

        model.train()  # Back to train mode
        print(f"[RIDGE-KICKSTART] Done!\n")

    # [KISS RIDGE] OLD BROKEN BLOCK DELETED - will be replaced after Trainer creation

    # Train (skip if eval-only)
    if not args.eval_only:
        # CURRICULUM LEARNING: Gradually increase context length
        if args.curriculum_learning and args.task == 'needle':
            print(f"[CURRICULUM] Starting curriculum learning (context: 512->8192)")

            # Define curriculum stages (chunk_len progression)
            # Total context = ctx_chunks * ctx_chunk_len, so we vary ctx_chunk_len
            curriculum_stages = [32, 64, 128, 256, 512]  # For 128 chunks: 4k->8k->16k->32k->64k
            steps_per_stage = args.curriculum_steps_per_stage

            total_steps_done = 0
            for stage_idx, chunk_len in enumerate(curriculum_stages):
                ctx_len = args.ctx_chunks * chunk_len
                print(f"\n[CURRICULUM] ===== Stage {stage_idx+1}/{len(curriculum_stages)} =====")
                print(f"[CURRICULUM] Context length: {ctx_len} tokens ({args.ctx_chunks} chunks × {chunk_len})")
                print(f"[CURRICULUM] Training for {steps_per_stage} steps")

                # Regenerate dataset with new context length
                stage_train_ds, stage_val_ds = generate_needle_dataset(
                    tokenizer,
                    needle_len=args.needle_len,
                    ctx_chunks=args.ctx_chunks,
                    ctx_chunk_len=chunk_len,  # Progressive chunk length
                    num_train=1000,
                    num_val=100,
                    seed=args.seed + stage_idx,  # Different seed per stage
                    use_query=args.needle_query
                )

                # Update trainer dataset
                trainer.train_dataset = stage_train_ds
                trainer.eval_dataset = stage_val_ds

                # Update training args for this stage
                stage_training_args = TrainingArguments(
                    output_dir=args.save,
                    max_steps=steps_per_stage,  # Train for stage duration
                    per_device_train_batch_size=args.bs,
                    gradient_accumulation_steps=args.ga,
                    learning_rate=args.lr,
                    warmup_steps=args.warmup_steps if stage_idx == 0 else 0,  # Warmup only first stage
                    logging_steps=20,  # Log metrics every 20 steps
                    logging_first_step=True,  # Log first step
                    save_steps=9999999,  # Disable auto-save (we do manual)
                    eval_strategy='no',
                    bf16=device.type == 'cuda',
                    dataloader_num_workers=args.num_workers,
                    dataloader_prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                    dataloader_pin_memory=args.pin_memory,
                    gradient_checkpointing=args.gradient_checkpointing,
                    max_grad_norm=args.grad_clip,
                    seed=args.seed,
                    report_to='none',  # Disable wandb/tensorboard
                    disable_tqdm=True,  # FORCE disable Trainer's tqdm
                )
                trainer.args = stage_training_args

                # Train for this stage
                stage_result = trainer.train()
                total_steps_done += steps_per_stage

                # Save checkpoint after each stage
                stage_checkpoint_path = os.path.join(args.save, f"checkpoint_curriculum_stage{stage_idx+1}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'stage': stage_idx + 1,
                    'ctx_len': ctx_len,
                    'total_steps': total_steps_done,
                }, stage_checkpoint_path)
                print(f"[CURRICULUM] Stage {stage_idx+1} complete, saved: {stage_checkpoint_path}")

                # Clear GPU cache between stages
                torch.cuda.empty_cache()

            print(f"\n[CURRICULUM] All stages complete! Total steps: {total_steps_done}")
            result = stage_result  # Last stage result

            # Final checkpoint
            checkpoint_path = os.path.join(args.save, "checkpoint_post_training.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': total_steps_done,
                'curriculum_complete': True
            }, checkpoint_path)
            print(f"[CHECKPOINT] Saved final curriculum checkpoint: {checkpoint_path}")

        else:
            # NORMAL TRAINING (no curriculum)
            print(f"[TRAIN] Starting training for {args.steps} steps...")
            print(f"[MAIN] Using trainer={type(trainer).__name__}, max_steps={trainer.args.max_steps}", flush=True)

            # ====================================================================
            # BUG #7 FIX: PVM WARMUP (Cold Start Prevention)
            # Before calibration, we must run a few batches to populate PVM
            # memory with non-zero values.
            # ====================================================================
            if args.kiss_ridge_calib and args.task == 'needle':
                print(f"[PVM-WARMUP] Starting PVM warmup (10 batches)...")
                warmup_batches = 0
                model.train()  # Ensure model is in training mode

                try:
                    # Use trainer's dataloader
                    temp_dataloader = trainer.get_train_dataloader()

                    with torch.no_grad():  # No gradients needed for warmup
                        for batch in temp_dataloader:
                            if warmup_batches >= 10:  # Warm up with 10 batches
                                break

                            # Prepare batch for model
                            inputs = {}
                            for k, v in batch.items():
                                if hasattr(v, "to"):
                                    if v.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                                        inputs[k] = v.to(device=model.device, dtype=next(model.parameters()).dtype)
                                    else:
                                        inputs[k] = v.to(model.device)

                            # Run model (this will call PVM.forward and populate memory)
                            model(**inputs, output_hidden_states=True)
                            warmup_batches += 1

                    print(f"[PVM-WARMUP] Warmup complete. PVM history is now populated.")

                    # Clean up dataloader so calibration can reuse it
                    del temp_dataloader

                except Exception as e:
                    print(f"[PVM-WARMUP] ERROR during warmup: {e}")
                    print(f"[PVM-WARMUP] Proceeding with cold memory (calibration will likely fail)")

            # KISS RIDGE ALIGNMENT: Initialize pvm2emb with statistical space calibration (CLAUDE2.md)
            if args.kiss_ridge_calib and hasattr(model, 'pvm2emb') and args.task == 'needle':
                print(f"\n{'='*80}")
                print(f"[KISS-RIDGE] Ridge Alignment Calibration (design specificationification)")
                print(f"[KISS-RIDGE] Collecting {args.kiss_ridge_max_pairs} calibration pairs from existing dataloader")
                print(f"[KISS-RIDGE] lambda (L2 reg) = {args.kiss_ridge_l2}")
                print(f"{'='*80}\n")

                try:
                    # Get the training dataloader from Trainer
                    train_dataloader = trainer.get_train_dataloader()

                    # Collect (Z, E) pairs using existing dataloader
                    Z, E = collect_alignment_pairs(
                        dataloader=train_dataloader,
                        model=model,
                        lm_embed_table=model.get_input_embeddings().weight,
                        max_pairs=args.kiss_ridge_max_pairs
                    )

                    print(f"[KISS-RIDGE] Collected pairs: Z={tuple(Z.shape)}, E={tuple(E.shape)}")
                    print(f"[KISS-RIDGE] Solving Ridge regression: W = (Z^T Z + lambda*I)^(-1) Z^T E")

                    # Solve Ridge regression
                    W = ridge_fit(Z, E, l2=args.kiss_ridge_l2)

                    # Initialize pvm2emb.weight with W^T (PyTorch Linear uses transposed weights)
                    # BUG #10 FIX: Dtype alignment for Ridge init
                    with torch.no_grad():
                        target_dtype = model.pvm2emb.weight.dtype
                        model.pvm2emb.weight.copy_(W.T.to(target_dtype))

                    print(f"[KISS-RIDGE] SUCCESS! pvm2emb initialized with Ridge solution")
                    print(f"[KISS-RIDGE] W shape: {W.shape}, dtype: {W.dtype}")

                    # Diagnostic: singular values (BUG #6 FIX: σ → sigma for Windows CP1250)
                    U, S, Vh = torch.linalg.svd(W.float())
                    sigma_max = S.max().item()
                    if sigma_max > 0:
                        sigma_min = S.min().item()
                        ratio = sigma_min / sigma_max
                        print(f"[KISS-RIDGE] SVD: sigma_min={sigma_min:.6f}, sigma_max={sigma_max:.6f}, ratio={ratio:.6f}")
                    else:
                        print(f"[KISS-RIDGE] OPOZORILO: Matrika W je nic (sigma_max = 0). Kalibracija ni uspela.")

                    # Keep layer frozen (or will be unfrozen later if configured)
                    model.pvm2emb.requires_grad_(False)
                    if args.kiss_ridge_unfreeze_step > 0:
                        print(f"[KISS-RIDGE] Layer FROZEN. Will unfreeze at step {args.kiss_ridge_unfreeze_step}")
                    else:
                        print(f"[KISS-RIDGE] Layer remains FROZEN (no fine-tune)")

                    print(f"[KISS-RIDGE] Calibration complete!\n")

                    # DESIGN-DOC: SIR poravnava - readout_head_proj should already be trainable
                    # when InfoNCE is enabled (no need to unfreeze, it's never frozen)

                except Exception as e:
                    print(f"[KISS-RIDGE] ERROR during calibration: {e}")
                    print(f"[KISS-RIDGE] Keeping random initialization")
                    import traceback
                    traceback.print_exc()

            result = trainer.train()
            print(f"[TRAIN] Training complete")

            # CRITICAL: Save checkpoint immediately after training (before eval)
            # This prevents data loss if eval OOMs
            checkpoint_path = os.path.join(args.save, "checkpoint_post_training.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': args.steps,
            }, checkpoint_path)
            print(f"[CHECKPOINT] Saved post-training checkpoint: {checkpoint_path}")
    else:
        print(f"[EVAL] Skipping training (eval-only mode)")
        result = None

    # Clear GPU cache before eval (reduce OOM risk)
    torch.cuda.empty_cache()
    print(f"[MEMORY] Cleared GPU cache before eval")

    # Eval metrics
    print(f"[EVAL] Running evaluation...")
    eval_result = trainer.evaluate()
    ce = float(eval_result['eval_loss'])  # Cross-entropy (nats)
    ppl = math.exp(ce)  # True PPL
    eval_tokens = eval_result.get('eval_samples', len(val_ds)) * args.seq

    # Needle-specific evaluation (if needle task)
    needle_hit_rate = None
    if args.task == 'needle':
        print(f"[NEEDLE-EVAL] Computing needle retrieval accuracy...")
        from datasets import Dataset

        # Reload original val dataset (with needle_pos metadata)
        _, val_ds_raw = generate_needle_dataset(
            tokenizer,
            needle_len=args.needle_len,
            ctx_chunks=args.ctx_chunks,
            ctx_chunk_len=args.ctx_chunk_len,
            num_train=1000,
            num_val=100,
            seed=args.seed,
            use_query=args.needle_query  # Match training mode
        )

        model.eval()
        correct = 0
        correct_topk = 0  # For soft check: top-k hit
        total = 0

        # PHASE 10D: Track partial needle matching
        # BUG FIX (2025-11-13): Now checks ALL needle tokens (not just first 5)
        total_correct_tokens = 0  # Sum of correct tokens across all samples
        total_needle_tokens = 0  # Total number of needle tokens checked
        partial_success_samples = 0  # Samples with >=50% correct tokens

        # Soft eval parameters
        eval_topk = getattr(args, 'eval_soft_topk', 5)  # Default: check top-5

        with torch.no_grad():
            for i, example in enumerate(val_ds_raw):
                if i >= 50:  # Sample 50 examples for speed
                    break

                # Transfer to GPU with non_blocking for faster throughput
                input_ids = torch.tensor([example['input_ids']]).to(device, non_blocking=True)
                needle_pos = example['needle_pos']
                needle = example['needle']

                # Sanity check on first eval batch
                if i == 0:
                    print(f"[NEEDLE-EVAL] First eval batch device: {input_ids.device}, shape: {input_ids.shape}")

                # Forward pass
                outputs = model(input_ids)
                logits = outputs.logits  # [1, seq_len, vocab_size]

                # PHASE 10D: Track correct tokens per sample
                sample_correct_tokens = 0
                sample_total_tokens = 0

                # BUG FIX (2025-11-13): Check ALL needle tokens (not just first 5)
                # WHY: User wants full per-token accuracy for all 16 needle tokens
                for offset in range(args.needle_len):  # Check ALL needle tokens
                    pos = needle_pos + offset
                    if pos < len(input_ids[0]) - 1 and offset < len(needle):
                        # Exact match (argmax)
                        pred_token = torch.argmax(logits[0, pos]).item()
                        true_token = needle[offset]
                        if pred_token == true_token:
                            correct += 1
                            sample_correct_tokens += 1  # Count correct tokens

                        # Top-k soft match (check if true token in top-k)
                        topk_preds = torch.topk(logits[0, pos], k=min(eval_topk, logits.size(-1)), dim=-1).indices.tolist()
                        if true_token in topk_preds:
                            correct_topk += 1

                        total += 1
                        sample_total_tokens += 1
                        total_needle_tokens += 1

                # PHASE 10D: Accumulate correct tokens
                total_correct_tokens += sample_correct_tokens

                # BUG FIX (2025-11-13): Track partial success (>=50% correct)
                # WHY: Binary all-or-nothing metric is too harsh for long needles
                if sample_total_tokens > 0 and sample_correct_tokens >= (sample_total_tokens / 2):
                    partial_success_samples += 1

        needle_hit_rate = correct / max(total, 1)
        needle_hit_rate_topk = correct_topk / max(total, 1)

        # PHASE 10D: Compute average correct tokens per needle (FIXED to use actual needle_len)
        num_samples = min(len(val_ds_raw), 50)
        avg_correct_tokens_per_needle = total_correct_tokens / max(num_samples, 1)
        avg_correct_ratio = avg_correct_tokens_per_needle / args.needle_len

        # BUG FIX (2025-11-13): Partial success rate (>=50% tokens correct)
        partial_success_rate = partial_success_samples / max(num_samples, 1)

        print(f"[NEEDLE-EVAL] Exact hit rate: {needle_hit_rate:.3f} ({correct}/{total})")
        print(f"[NEEDLE-EVAL] Top-{eval_topk} hit rate: {needle_hit_rate_topk:.3f} ({correct_topk}/{total})")
        print(f"[NEEDLE-EVAL] Avg correct tokens per needle: {avg_correct_tokens_per_needle:.2f}/{args.needle_len} ({avg_correct_ratio:.3f})")
        print(f"[NEEDLE-EVAL] Partial success rate (>=50%): {partial_success_rate:.3f} ({partial_success_samples}/{num_samples})")
        print(f"[NEEDLE-EVAL] Context length: {args.ctx_chunks * args.ctx_chunk_len} tokens")

        # Phase curvature metrics (CLAUDE2.md Section 2.3)
        if args.phase_curvature_metrics and args.enable_pvm:
            print(f"[PHASE-CURV] Computing phase curvature metrics...")
            # Log needle query mode
            needle_query_mode = getattr(args, 'needle_query_mode', 'mean_tail')
            print(f"[PHASE-CURV] needle_query_mode={needle_query_mode}")
            # Try to access PVM from first AFRB adapter
            pvm_found = False
            phase_curv_metrics = None
            for adapter in model.afrb_adapters:
                if hasattr(adapter, 'pvm') and adapter.pvm is not None:
                    traj = adapter.pvm.get_last_trajectory()
                    if traj is not None:
                        # Get needle_pos and retrieval_start from last evaluated example
                        # Use values from the last example in the eval loop
                        last_needle_pos = int(example['needle_pos'])
                        last_retrieval_start = int(example.get('retrieval_start', -1))

                        # Compute curvature metrics
                        phase_curv_metrics = compute_phase_curvature_metrics(
                            traj, last_needle_pos, last_retrieval_start, extended=True
                        )

                        print(f"[PHASE-CURV] curv_peak_at_needle = {phase_curv_metrics['curv_peak_at_needle']:.6f}")
                        print(f"[PHASE-CURV] curv_peak_before_needle = {phase_curv_metrics['curv_peak_before_needle']:.6f}")
                        print(f"[PHASE-CURV] curv_traj_len = {phase_curv_metrics['curv_traj_len']}")
                        print(f"[PHASE-CURV] curv_peak_offset_from_retrieval = {phase_curv_metrics['curv_peak_offset_from_retrieval']}")
                        print(f"[PHASE-CURV] curv_peak_at_needle_binary = {phase_curv_metrics['curv_peak_at_needle_binary']}")
                        print(f"[PHASE-CURV] curv_pre_peak_fraction = {phase_curv_metrics['curv_pre_peak_fraction']:.4f}")
                        print(f"[PHASE-CURV] curv_entropy = {phase_curv_metrics['curv_entropy']:.4f}")
                        print(f"[PHASE-CURV] curv_peak_idx = {phase_curv_metrics['curv_peak_idx']}")

                        # Distance metrics (Agent 19A - CLAUDE2.md Section 1)
                        needle_len = args.needle_len if hasattr(args, 'needle_len') else 16
                        needle_center = last_needle_pos + needle_len / 2.0

                        peak_dist_from_needle_center = float(phase_curv_metrics['curv_peak_idx'] - needle_center)
                        peak_dist_from_retrieval = float(phase_curv_metrics['curv_peak_idx'] - last_retrieval_start)

                        phase_curv_metrics['peak_dist_from_needle_center'] = peak_dist_from_needle_center
                        phase_curv_metrics['peak_dist_from_retrieval'] = peak_dist_from_retrieval

                        print(f"[PHASE-CURV] curv_peak_dist_from_needle_center = {peak_dist_from_needle_center:.1f}")
                        print(f"[PHASE-CURV] curv_peak_dist_from_retrieval = {peak_dist_from_retrieval:.1f}")

                        # Soft success metrics (Agent 19B - CLAUDE2.md Section 2 + Enhancements)
                        # Basic soft success: normalized curvature at needle (0-1 scale)
                        curv_soft_success = float(phase_curv_metrics['curv_peak_at_needle'])
                        # curv_soft_success: How close needle curvature is to global peak (0-1)

                        # Distance-weighted soft success: penalize peaks far from needle
                        dist = abs(phase_curv_metrics['peak_dist_from_needle_center'])
                        decay_factor = math.exp(-dist / 10.0)  # Exponential decay with 10-token characteristic length
                        curv_soft_success_weighted = curv_soft_success * decay_factor
                        # curv_soft_success_weighted: Soft success * distance decay (penalizes far peaks)

                        # Margin: needle advantage over background (can be negative if background stronger)
                        curv_margin = float(phase_curv_metrics['curv_peak_at_needle'] - phase_curv_metrics['curv_peak_before_needle'])
                        # curv_margin: Needle advantage over background (can be negative)

                        # Absolute margin: always-positive separation strength
                        curv_margin_abs = abs(curv_margin)
                        # curv_margin_abs: Absolute separation strength (always positive)

                        # Ratio margin: multiplicative comparison (2.5 = needle 2.5x stronger than background)
                        curv_margin_ratio = float(phase_curv_metrics['curv_peak_at_needle'] / (phase_curv_metrics['curv_peak_before_needle'] + 1e-8))
                        # curv_margin_ratio: Multiplicative advantage (2.0 = needle 2x stronger)

                        # Add all soft success metrics to dict
                        phase_curv_metrics['curv_soft_success'] = curv_soft_success
                        phase_curv_metrics['curv_soft_success_weighted'] = curv_soft_success_weighted
                        phase_curv_metrics['curv_margin'] = curv_margin
                        phase_curv_metrics['curv_margin_abs'] = curv_margin_abs
                        phase_curv_metrics['curv_margin_ratio'] = curv_margin_ratio

                        # Print soft success metrics
                        print(f"[PHASE-CURV] curv_soft_success = {curv_soft_success:.4f}")
                        print(f"[PHASE-CURV] curv_soft_success_weighted = {curv_soft_success_weighted:.4f}")
                        print(f"[PHASE-CURV] curv_margin = {curv_margin:.4f}")
                        print(f"[PHASE-CURV] curv_margin_abs = {curv_margin_abs:.4f}")
                        print(f"[PHASE-CURV] curv_margin_ratio = {curv_margin_ratio:.2f}")

                        # Phase regime classification (Agent 19C - CLAUDE2.md Section 3 + Enhancement)
                        # Phase regime categories:
                        #   on_needle: |dist| <= 4 (peak at needle, ±4 token window)
                        #   prebuild: dist < -4 (peak before needle, proactive encoding)
                        #   post_needle: 4 < dist <= 50 (peak after needle, moderate delay)
                        #   far_field: dist > 50 (peak very far, spatial misalignment)
                        if abs(peak_dist_from_needle_center) <= 4:
                            phase_regime = "on_needle"
                        elif peak_dist_from_needle_center < -4:
                            phase_regime = "prebuild"
                        elif peak_dist_from_needle_center > 50:
                            phase_regime = "far_field"
                        else:
                            phase_regime = "post_needle"

                        phase_curv_metrics['phase_curv_regime'] = phase_regime
                        print(f"[PHASE-CURV] phase_curv_regime = {phase_regime}")
                        print(f"[PHASE-CURV] curv_peak_offset = {peak_dist_from_needle_center:.1f} (target: closer to 0)")

                        pvm_found = True
                        break
                    else:
                        print(f"[PHASE-CURV] WARNING: PVM found but trajectory is None (no forward pass recorded)")

            if not pvm_found:
                print(f"[PHASE-CURV] WARNING: No PVM with valid trajectory found")
        elif args.phase_curvature_metrics and not args.enable_pvm:
            print(f"[PHASE-CURV] WARNING: --phase-curvature-metrics requires --enable-pvm")

    # Phase metrics (sample batch) - use small sample to avoid OOM
    if args.skip_phase_metrics:
        print(f"[EVAL] Skipping phase metrics (--skip-phase-metrics flag)")
        coh = 0.0
        gammas = [adapter.get_gamma().item() for adapter in model.afrb_adapters]
        gamma_sat = gamma_saturation(torch.tensor(gammas))
        ent = 0.0
    else:
        model.eval()
        with torch.no_grad():
            # Use only 4 samples to minimize memory usage
            sample = val_ds.select(range(min(4, len(val_ds))))
            sample_ids = torch.tensor([s['input_ids'] for s in sample], device=device)

            try:
                outputs = model(sample_ids, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]

                coh = phase_coherence(hidden)
                gammas = [adapter.get_gamma().item() for adapter in model.afrb_adapters]
                gamma_sat = gamma_saturation(torch.tensor(gammas))
                ent = entropy_flow(hidden)
            except torch.cuda.OutOfMemoryError:
                print(f"[WARNING] OOM during phase metrics, using fallback values")
                coh = 0.0
                gammas = [adapter.get_gamma().item() for adapter in model.afrb_adapters]
                gamma_sat = gamma_saturation(torch.tensor(gammas))
                ent = 0.0
                torch.cuda.empty_cache()

    print(f"[EVAL] CE={ce:.3f}, PPL={ppl:.0f}, Coh={coh:.2f}, g-sat={gamma_sat:.2f}, eval_tokens={eval_tokens}")

    # Save metrics
    metrics = {
        'eval_ce': ce,
        'eval_ppl': ppl,
        'eval_tokens': eval_tokens,
        'ppl': ppl,  # Keep for backward compat
        'phase_coherence': coh,
        'gamma_saturation': gamma_sat,
        'gamma_values': gammas,
        'entropy_flow': ent,
        'seed': args.seed,
        'n_afrb': args.n_afrb,
        'alpha': args.alpha,
        'gamma': args.gamma,
        'omega': args.omega,
        'lr': args.lr,
        'steps': args.steps,
        'trainable_params': trainable,
        'timestamp': datetime.now().isoformat(),
        'vram_gb': torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        # DRC parameters
        'cascade_kappa': args.cascade_kappa,
        'cascade_lambda': args.cascade_lambda,
        'phase_delta': args.phase_delta,
        'omega_delta': args.omega_delta,
        # Stillness parameters
        'stillness_ema': args.stillness_ema,
        'stillness_warm': args.stillness_warm,
        'stillness_floor': args.stillness_floor,
        # Stagger parameters
        'block_warm_delta': args.block_warm_delta,
        'block_ramp': args.block_ramp,
        'phase_ramp': args.phase_ramp,
        # Adaptive-OMEGA parameters
        'adaptive_omega': args.adaptive_omega,
        'adaptive_omega_lr': args.adaptive_omega_lr if args.adaptive_omega else 0.0,
        'adaptive_omega_interval': args.adaptive_omega_interval if args.adaptive_omega else 0,
        'omega_min': args.omega_min,
        'omega_max': args.omega_max,
        # Omega profile (learned or fixed)
        'omega_profile': [float(adapter.phase_embed.current_omega().item()) for adapter in model.afrb_adapters],
        # Per-block profile (for debugging full stack)
        'afrb_profile': [
            {
                'block_idx': adapter.block_idx,
                'alpha': float(adapter.alpha.item()),
                'gamma': float(torch.sigmoid(adapter.gamma_raw).item()),
                'phi_base': float(adapter.phi_base.item()),
                'phi_target': float(adapter.phi_base.item() + adapter.phase_delta),
                'omega': float(adapter.phase_embed.current_omega().item()),
                'omega_learnable': adapter.phase_embed.learnable_omega,
                'stillness_rho': float(adapter.rho),
                'stillness_floor': float(adapter.floor),
                'warm_i': adapter.warm_i,
                'ramp': adapter.ramp
            }
            for adapter in model.afrb_adapters
        ]
    }

    # Add PVM metrics (if enabled)
    if args.enable_pvm:
        pvm_metrics = compute_pvm_metrics(model, enable_rca=args.enable_rca)
        metrics.update(pvm_metrics)
        metrics['pvm_enabled'] = True
        metrics['pvm_alpha'] = args.pvm_alpha
        metrics['pvm_beta'] = args.pvm_beta
        metrics['pvm_gate_init'] = args.pvm_gate_init
        metrics['rca_enabled'] = args.enable_rca
    else:
        metrics['pvm_enabled'] = False
        metrics['rca_enabled'] = False

    if args.enable_plm:
        plm_metrics = compute_plm_metrics(model)
        metrics.update(plm_metrics)
        metrics['plm_enabled'] = True
        metrics['plm_grid_x'] = args.plm_grid_x
        metrics['plm_grid_y'] = args.plm_grid_y
        metrics['plm_alpha'] = args.plm_alpha
        metrics['plm_beta'] = args.plm_beta
        metrics['plm_omega'] = args.plm_omega
        metrics['plm_kappa'] = args.plm_kappa
        metrics['plm_gate_init'] = args.plm_gate_init
    else:
        metrics['plm_enabled'] = False

    # Track T2/PCM parameters (PhaseBuffer)
    metrics['t2_enable'] = args.t2_enable
    metrics['t2_steps'] = args.t2_steps if args.t2_enable else 0
    metrics['t2_mode'] = args.t2_mode if args.t2_enable else 'exp'
    metrics['t2_k'] = args.t2_k if (args.t2_enable and args.t2_mode == 'resonant') else 0
    metrics['t2_alpha'] = args.t2_alpha if (args.t2_enable and args.t2_mode == 'resonant') else 0
    metrics['t2_omega'] = args.t2_omega if (args.t2_enable and args.t2_mode == 'resonant') else 0
    metrics['t2_phi'] = args.t2_phi if (args.t2_enable and args.t2_mode == 'resonant') else 0
    metrics['pcm_enable'] = args.pcm_enable
    metrics['pcm_gate_init'] = args.pcm_gate_init if args.pcm_enable else 0
    # Compute final T2 decay value for diagnostics
    if args.t2_enable and args.t2_steps > 0 and args.t2_mode == 'exp':
        final_decay = math.exp(-args.steps / args.t2_steps)
        metrics['t2_decay_final'] = final_decay
        metrics['t2_decay_fraction'] = args.steps / args.t2_steps
    elif args.t2_enable and args.t2_mode == 'resonant':
        # For resonant mode, record decay value at final step
        t = float(args.steps) + 1.0
        base_decay = math.exp(-args.t2_k * t)
        log_t = math.log(t)
        oscillation = 1.0 + args.t2_alpha * math.sin(args.t2_omega * log_t + args.t2_phi)
        final_decay = max(0.0, base_decay * max(0.0, oscillation))
        metrics['t2_decay_final'] = final_decay

    # Track attention status (Phase 7b)
    metrics['attention_disabled'] = args.disable_attn

    # Track memory optimization (Phase 7c)
    metrics['gradient_checkpointing'] = args.gradient_checkpointing

    # Track needle task metrics (Phase 7c)
    if args.task == 'needle':
        metrics['task'] = 'needle'
        metrics['needle_len'] = args.needle_len
        metrics['context_len'] = args.ctx_chunks * args.ctx_chunk_len
        metrics['needle_hit_rate'] = needle_hit_rate if needle_hit_rate is not None else 0.0
        # Soft eval metrics (top-k hit)
        if 'needle_hit_rate_topk' in locals():
            metrics['needle_hit_rate_topk'] = needle_hit_rate_topk
            metrics['eval_soft_topk'] = getattr(args, 'eval_soft_topk', 5)
        # PHASE 10D: Track partial needle matching metrics
        # BUG FIX (2025-11-13): Now includes per-token accuracy and partial success rate
        if 'avg_correct_tokens_per_needle' in locals():
            metrics['avg_correct_tokens_per_needle'] = avg_correct_tokens_per_needle
            metrics['avg_correct_ratio'] = avg_correct_ratio
        if 'partial_success_rate' in locals():
            metrics['needle_partial_success_rate'] = partial_success_rate
        # PHASE 10D: Track pointer quality from training history
        # BUG FIX (2025-11-13): Now includes recall, precision, and F1 (was: saturated at 5/16)
        if hasattr(trainer, 'pointer_quality_history') and len(trainer.pointer_quality_history) > 0:
            # Extract all metrics (backward compatible with old 'ratio' field)
            recalls = [entry.get('recall', entry.get('ratio', 0.0)) for entry in trainer.pointer_quality_history]
            precisions = [entry.get('precision', 0.0) for entry in trainer.pointer_quality_history]
            f1s = [entry.get('f1', 0.0) for entry in trainer.pointer_quality_history]

            # Primary metric is now F1 (harmonic mean of recall and precision)
            metrics['needle_pointer_quality'] = float(np.mean(f1s)) if f1s else 0.0
            metrics['needle_pointer_quality_max'] = float(np.max(f1s)) if f1s else 0.0
            metrics['needle_pointer_quality_final'] = f1s[-1] if f1s else 0.0

            # Also track recall and precision separately
            metrics['needle_pointer_recall'] = float(np.mean(recalls)) if recalls else 0.0
            metrics['needle_pointer_recall_max'] = float(np.max(recalls)) if recalls else 0.0
            metrics['needle_pointer_precision'] = float(np.mean(precisions)) if precisions else 0.0
            metrics['needle_pointer_precision_max'] = float(np.max(precisions)) if precisions else 0.0
        # Phase curvature metrics (CLAUDE2.md Section 2.3)
        if 'phase_curv_metrics' in locals() and phase_curv_metrics is not None:
            metrics['phase_curv_peak_at_needle'] = float(phase_curv_metrics['curv_peak_at_needle'])
            metrics['phase_curv_peak_before_needle'] = float(phase_curv_metrics['curv_peak_before_needle'])
            metrics['phase_curv_traj_len'] = int(phase_curv_metrics['curv_traj_len'])
            # Extended metrics (Agent 18C)
            if 'curv_peak_offset_from_retrieval' in phase_curv_metrics:
                metrics['phase_curv_peak_offset_from_retrieval'] = int(phase_curv_metrics['curv_peak_offset_from_retrieval'])
                metrics['phase_curv_peak_at_needle_binary'] = int(phase_curv_metrics['curv_peak_at_needle_binary'])
                metrics['phase_curv_pre_peak_fraction'] = float(phase_curv_metrics['curv_pre_peak_fraction'])
                metrics['phase_curv_entropy'] = float(phase_curv_metrics['curv_entropy'])
                metrics['phase_curv_peak_idx'] = int(phase_curv_metrics['curv_peak_idx'])

            # Distance metrics (Agent 19A)
            if 'peak_dist_from_needle_center' in phase_curv_metrics:
                metrics['phase_curv_peak_dist_from_needle_center'] = float(phase_curv_metrics['peak_dist_from_needle_center'])
                metrics['phase_curv_peak_dist_from_retrieval'] = float(phase_curv_metrics['peak_dist_from_retrieval'])

            # Soft success metrics (Agent 19B - CLAUDE2.md Section 2 + Enhancements)
            if 'curv_soft_success' in phase_curv_metrics:
                metrics['phase_curv_soft_success'] = float(phase_curv_metrics['curv_soft_success'])
                metrics['phase_curv_soft_success_weighted'] = float(phase_curv_metrics['curv_soft_success_weighted'])
                metrics['phase_curv_margin'] = float(phase_curv_metrics['curv_margin'])
                metrics['phase_curv_margin_abs'] = float(phase_curv_metrics['curv_margin_abs'])
                metrics['phase_curv_margin_ratio'] = float(phase_curv_metrics['curv_margin_ratio'])

            # Phase regime classification (Agent 19C - CLAUDE2.md Section 3 + Enhancement)
            if 'phase_curv_regime' in phase_curv_metrics:
                metrics['phase_curv_regime'] = str(phase_curv_metrics['phase_curv_regime'])
    else:
        metrics['task'] = args.task

    # Track readout parameters (Phase 9)
    if hasattr(args, 'readout_from'):
        metrics['readout_from'] = args.readout_from
        metrics['readout_topk'] = args.readout_topk
        metrics['readout_scale'] = args.readout_scale
        metrics['readout_window'] = args.readout_window
        metrics['readout_head'] = args.readout_head if hasattr(args, 'readout_head') else 'shared'
        metrics['phase_alignment_weight'] = args.phase_alignment_weight

        # Compute average readout stats if logged
        if hasattr(model, '_readout_stats_history') and len(model._readout_stats_history) > 0:
            stats_keys = ['pointer_norm', 'pvm_scores_max', 'pvm_scores_mean', 'pvm_scores_entropy',
                         'plm_scores_max', 'plm_scores_mean', 'plm_scores_entropy']
            readout_stats = {}
            for key in stats_keys:
                values = [s.get(key, 0) for s in model._readout_stats_history if key in s]
                if values:
                    readout_stats[f'readout_{key}_avg'] = float(np.mean(values))
                    readout_stats[f'readout_{key}_max'] = float(np.max(values))
            metrics['readout_stats'] = readout_stats

    # NOTE: HuggingFace Trainer logs grad_norm internally via callback.
    # Previously had: metrics['grad_norm'] = result.training_loss (WRONG - mixed up loss with grad_norm!)
    # Removed to avoid confusion - actual grad_norm is logged by Trainer's default callbacks.

    # Save command for reproducibility
    metrics['command_args'] = sys.argv  # List of arguments
    metrics['command_full'] = ' '.join(sys.argv)  # Full command string

    with open(save_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # PHASE 10D: Save progress CSV for easy tracking
    csv_path = save_dir / 'progress_log.csv'
    csv_exists = csv_path.exists()

    # Create CSV with headers if it doesn't exist
    # BUG FIX (2025-11-13): Added recall/precision columns for pointer quality + partial success
    if not csv_exists:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'step', 'ppl', 'ce_loss',
                'needle_hit_rate', 'needle_hit_rate_topk', 'needle_partial_success',
                'avg_correct_tokens', 'avg_correct_ratio',
                'pointer_f1', 'pointer_f1_max',
                'pointer_recall', 'pointer_precision',
                'phase_coherence', 'entropy_flow', 'gamma_saturation'
            ])

    # Append current metrics
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            args.steps if not args.eval_only else 0,
            f"{ppl:.2f}",
            f"{ce:.4f}",
            f"{metrics.get('needle_hit_rate', 0.0):.4f}",
            f"{metrics.get('needle_hit_rate_topk', 0.0):.4f}",
            f"{metrics.get('needle_partial_success_rate', 0.0):.4f}",
            f"{metrics.get('avg_correct_tokens_per_needle', 0.0):.2f}",
            f"{metrics.get('avg_correct_ratio', 0.0):.4f}",
            f"{metrics.get('needle_pointer_quality', 0.0):.4f}",
            f"{metrics.get('needle_pointer_quality_max', 0.0):.4f}",
            f"{metrics.get('needle_pointer_recall', 0.0):.4f}",
            f"{metrics.get('needle_pointer_precision', 0.0):.4f}",
            f"{coh:.4f}",
            f"{ent:.4f}",
            f"{gamma_sat:.4f}"
        ])

    print(f"[CSV] Progress logged to {csv_path}")

    # Save model (skip if eval-only)
    if not args.eval_only:
        model.save_pretrained(save_dir / 'best.pt')
        print(f"[SAVE] Model saved to {save_dir / 'best.pt'}")

    # Update README
    readme = Path('reports/README.md')
    readme.parent.mkdir(exist_ok=True)
    with open(readme, 'a', encoding='utf-8', newline='') as f:
        mode_str = "EVAL" if args.eval_only else "TRAIN"
        f.write(f"\n[{datetime.now():%Y-%m-%d %H:%M}] [{mode_str}] AF-RB n={args.n_afrb} a={args.alpha:.2f} g={args.gamma:.2f} w={args.omega:.1f}: CE={ce:.2f} PPL={ppl:.0f} coh={coh:.2f}\n")

    # PHASE 10D: Print comprehensive needle metrics summary
    # BUG FIX (2025-11-13): Now shows per-token accuracy, partial success, F1/recall/precision
    if args.task == 'needle':
        print(f"\n{'='*80}")
        print(f"[NEEDLE-METRICS-SUMMARY]")
        print(f"{'='*80}")
        print(f"Exact hit rate (100%):             {metrics.get('needle_hit_rate', 0.0):.4f}")
        print(f"Top-{metrics.get('eval_soft_topk', 5)} hit rate:                    {metrics.get('needle_hit_rate_topk', 0.0):.4f}")
        print(f"Partial success rate (>=50%):      {metrics.get('needle_partial_success_rate', 0.0):.4f}")
        print(f"")
        print(f"Per-Token Accuracy (all {args.needle_len} tokens):")
        print(f"  Avg correct tokens:              {metrics.get('avg_correct_tokens_per_needle', 0.0):.2f}/{args.needle_len}")
        print(f"  Avg correct ratio:               {metrics.get('avg_correct_ratio', 0.0):.4f}")
        print(f"")
        print(f"Pointer Quality (K=64, during training):")
        print(f"  F1 (avg):                        {metrics.get('needle_pointer_quality', 0.0):.4f}")
        print(f"  F1 (max):                        {metrics.get('needle_pointer_quality_max', 0.0):.4f}")
        print(f"  F1 (final):                      {metrics.get('needle_pointer_quality_final', 0.0):.4f}")
        print(f"  Recall (avg):                    {metrics.get('needle_pointer_recall', 0.0):.4f}")
        print(f"  Precision (avg):                 {metrics.get('needle_pointer_precision', 0.0):.4f}")
        print(f"{'='*80}\n")

    print(f"[OK] Saved: {save_dir}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
