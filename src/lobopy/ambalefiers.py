# lobopy - A Python module
# Copyright (C) 2026 OzelTam
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
ambalefiers.py  –  Applied-function factories for Patient.ambale()
===================================================================

Every factory returns an **applied_function** with the signature::

    (stored: torch.Tensor, live: torch.Tensor) -> torch.Tensor

where:
  - ``stored`` is the pre-computed activation tensor for this layer
    from an earlier ``analyse()`` call  (shape: seq_len_s, hidden_dim)
  - ``live``   is the current forward-pass hidden-state tensor at the
    same layer                           (shape: seq_len_l, hidden_dim)

The return value replaces the layer's output during inference.

Recommended pipeline
--------------------
1. Compute a *path* (difference vector) with an aggregator::

       happy_path = difference_aggregator()(happy.activations, neutral.activations)

2. Normalise it so ``factor`` is scale-invariant across layers::

       normed = normalize_path(happy_path)

3. Select only the K most contrastive middle layers to avoid breaking coherence::

       steered = top_k_layers(normed, k=4, layer_range=(0.15, 0.75))

4. Ambale and generate::

       with model.ambale(steered, safe_scale_activation(factor=20.0)):
           out = model.llm.generate(...)
"""

import torch
from typing import Callable, Dict, Literal, Optional, Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Activations = Dict[int, torch.Tensor]
AppliedFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# Applied-function factories
# ---------------------------------------------------------------------------


def scale_activation(factor: float = 1.0) -> AppliedFunction:
    """
    Adds ``factor * stored`` to the live activations.

    A positive factor amplifies the stored direction in the live stream.
    A negative factor suppresses it.  ``factor=0`` leaves live unchanged.

    Args:
        factor: Scalar multiplier applied to ``stored`` before addition.

    Returns:
        AppliedFunction: ``live + factor * stored``
    """
    def apply(stored: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
        s = _align_to_live(stored, live)
        return live + factor * s.to(live.dtype).to(live.device)
    apply.factory_name = "scale_activation"
    apply.kwargs = {"factor": factor}
    return apply


def dampen_activation(factor: float = 1.0) -> AppliedFunction:
    """
    Subtracts ``factor * stored`` from the live activations.

    Convenience alias for ``scale_activation(-factor)``.

    Args:
        factor: How strongly to subtract the stored direction (non-negative).

    Returns:
        AppliedFunction: ``live - factor * stored``
    """
    def apply(stored: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
        s = _align_to_live(stored, live)
        return live - factor * s.to(live.dtype).to(live.device)
    apply.factory_name = "dampen_activation"
    apply.kwargs = {"factor": factor}
    return apply


def replace_activation() -> AppliedFunction:
    """
    Replaces the live activations entirely with the stored tensor.

    Returns:
        AppliedFunction: ``stored`` (aligned to live's seq length)
    """
    def apply(stored: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
        s = _align_to_live(stored, live)
        return s.to(live.dtype).to(live.device)
    apply.factory_name = "replace_activation"
    apply.kwargs = {}
    return apply


def project_out_activation(eps: float = 1e-8) -> AppliedFunction:
    """
    Removes the component of ``live`` that lies along the ``stored`` direction.

    For each token position, computes the unit vector of ``stored`` and
    subtracts the projection of ``live`` onto it.

    .. warning::
        If ``stored`` is near-zero (e.g. from ``common_ground_aggregator`` on
        opposite-polarity paths), the unit vector is computed from noise
        dominated by ``eps``, projecting out a random direction.
        Always check path magnitudes before using this.

    Args:
        eps: Small value to avoid division by zero.

    Returns:
        AppliedFunction: ``live - (live·û)û``  where ``û = stored / ||stored||``
    """
    def apply(stored: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
        s = _align_to_live(stored, live).to(live.dtype).to(live.device)
        norm = s.norm(dim=-1, keepdim=True).clamp(min=eps)
        unit = s / norm
        projection = (live * unit).sum(dim=-1, keepdim=True) * unit
        return live - projection
    apply.factory_name = "project_out_activation"
    apply.kwargs = {"eps": eps}
    return apply


def project_in_activation(eps: float = 1e-8) -> AppliedFunction:
    """
    Keeps only the component of ``live`` that lies along the ``stored`` direction.

    Args:
        eps: Small value to avoid division by zero.

    Returns:
        AppliedFunction: ``(live·û)û``  where ``û = stored / ||stored||``
    """
    def apply(stored: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
        s = _align_to_live(stored, live).to(live.dtype).to(live.device)
        norm = s.norm(dim=-1, keepdim=True).clamp(min=eps)
        unit = s / norm
        projection = (live * unit).sum(dim=-1, keepdim=True) * unit
        return projection
    apply.factory_name = "project_in_activation"
    apply.kwargs = {"eps": eps}
    return apply


def clamp_activation(min_val: Optional[float] = None, max_val: Optional[float] = None) -> AppliedFunction:
    """
    Clamps live activations element-wise using the stored tensor as reference bounds.

    Args:
        min_val: Optional lower bound multiplier on ``|stored|``.
        max_val: Optional upper bound multiplier on ``|stored|``.

    Returns:
        AppliedFunction: element-wise clamped ``live``
    """
    if min_val is None and max_val is None:
        raise ValueError("At least one of min_val or max_val must be provided.")

    def apply(stored: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
        s = _align_to_live(stored, live).to(live.dtype).to(live.device)
        mag = s.abs()
        lo = (-mag * min_val) if min_val is not None else None
        hi = (mag * max_val) if max_val is not None else None
        return live.clamp(min=lo, max=hi)
    apply.factory_name = "clamp_activation"
    apply.kwargs = {"min_val": min_val, "max_val": max_val}
    return apply


def blend_activation(alpha: float = 0.5) -> AppliedFunction:
    """
    Linearly blends the live activations with the stored activations.

    ``alpha=1.0`` → fully live (no effect), ``alpha=0.0`` → fully stored.

    .. warning::
        ``blend_activation`` injects the stored *token sequence* directly into
        the forward pass positionally.  When the stored vector comes from
        training prompts that differ in length or subject from the generation
        prompt, this mixes in training-data distributions (code, Q&A lists, etc.).
        Prefer ``safe_scale_activation`` for cleaner steering.

    Args:
        alpha: Weight of the live activation in the blend [0, 1].

    Returns:
        AppliedFunction: ``alpha * live + (1 - alpha) * stored``
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    def apply(stored: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
        s = _align_to_live(stored, live).to(live.dtype).to(live.device)
        return alpha * live + (1.0 - alpha) * s
    apply.factory_name = "blend_activation"
    apply.kwargs = {"alpha": alpha}
    return apply


def safe_scale_activation(factor: float = 1.0, clamp_sigma: float = 3.0) -> AppliedFunction:
    """
    Like ``scale_activation`` but clamps the steering delta to
    ±``clamp_sigma`` standard deviations of the live tensor.

    Use this instead of raw ``scale_activation`` to prevent blowing up the
    residual stream. Combine with ``normalize_path`` so that ``factor`` is
    scale-invariant across layers.

    Args:
        factor:      Multiplier on the stored direction.
        clamp_sigma: Max allowed delta in live std-dev units. Default 3.0.

    Returns:
        AppliedFunction: ``live + clamp(factor * stored, ±σ * clamp_sigma)``
    """
    def apply(stored: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
        s = _align_to_live(stored, live).to(live.dtype).to(live.device)
        delta = factor * s
        bound = live.std() * clamp_sigma
        delta = delta.clamp(-bound, bound)
        return live + delta
    apply.factory_name = "safe_scale_activation"
    apply.kwargs = {"factor": factor, "clamp_sigma": clamp_sigma}
    return apply


# ---------------------------------------------------------------------------
# Path utilities  (call these BEFORE model.ambale)
# ---------------------------------------------------------------------------


def normalize_path(
    activations: Activations,
    eps: float = 1e-8,
) -> Activations:
    """
    L2-normalises each layer's steering vector to unit norm per token position.

    **Call this before ``top_k_layers`` and ``model.ambale``.**

    Without normalisation ``factor`` in ``safe_scale_activation`` means
    different things in different layers (raw magnitudes vary up to 10× across
    depth).  After normalisation ``factor`` is directly interpretable as
    "how many units of hidden-space to inject", making tuning predictable.

    Args:
        activations: Raw ``{layer_idx: tensor}`` dict.
        eps:         Guard against division by zero.

    Returns:
        New dict with each tensor L2-normalised along the hidden dimension.
    """
    result: Activations = {}
    for layer_idx, tensor in activations.items():
        norm = tensor.norm(dim=-1, keepdim=True).clamp(min=eps)  # (seq, 1)
        result[layer_idx] = tensor / norm
    return result


def top_k_layers(
    activations: Activations,
    k: int,
    metric: Literal["mean_abs", "max_abs", "l2"] = "mean_abs",
    layer_range: Optional[Tuple[float, float]] = None,
) -> Activations:
    """
    Filters an activation dict to the K layers with the highest contrast signal.

    Call this on a **normalised** path tensor before ``model.ambale()``.
    Steering only K focused layers keeps the rest of the residual stream
    untouched and dramatically reduces gibberish / repetition loops.

    Args:
        activations:  ``{layer_idx: tensor}`` — typically a normalised path.
        k:            Number of layers to keep.
        metric:       Scoring function per layer:

                      * ``"mean_abs"`` — mean absolute value (default, robust)
                      * ``"max_abs"``  — max absolute value (spikiest layers)
                      * ``"l2"``       — Frobenius norm

        layer_range:  Optional ``(lo_pct, hi_pct)`` in [0, 1] to restrict
                      selection to a band of network depth before scoring.
                      Example: ``(0.15, 0.75)`` skips the first 15% and last
                      25% of layers — avoids logit-sensitive final layers that
                      cause the most coherence damage.  Percentages are
                      relative to ``max(keys) + 1``.

    Returns:
        New ``{layer_idx: tensor}`` dict with at most ``k`` entries.

    Example::

        happy_path  = difference_aggregator()(happy.activations, neutral.activations)
        normed      = normalize_path(happy_path)
        top_normed  = top_k_layers(normed, k=4, layer_range=(0.15, 0.75))
        model.ambale(top_normed, safe_scale_activation(factor=20.0))
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    if not activations:
        return {}

    candidates = activations

    if layer_range is not None:
        lo_pct, hi_pct = layer_range
        if not (0.0 <= lo_pct < hi_pct <= 1.0):
            raise ValueError(
                f"layer_range must satisfy 0 ≤ lo < hi ≤ 1, got {layer_range}"
            )
        total = max(activations.keys()) + 1
        lo_idx = int(lo_pct * total)
        hi_idx = int(hi_pct * total)
        candidates = {ki: v for ki, v in activations.items() if lo_idx <= ki < hi_idx}
        if not candidates:
            raise ValueError(
                f"layer_range={layer_range} excluded all {len(activations)} available "
                f"layers (indices {sorted(activations.keys())}). "
                f"Widen the range or reduce the restriction."
            )

    def score(tensor: torch.Tensor) -> float:
        if metric == "mean_abs":
            return tensor.abs().mean().item()
        elif metric == "max_abs":
            return tensor.abs().max().item()
        elif metric == "l2":
            return tensor.norm().item()
        else:
            raise ValueError(
                f"Unknown metric '{metric}'. Choose 'mean_abs', 'max_abs', or 'l2'."
            )

    scored = sorted(candidates.items(), key=lambda kv: score(kv[1]), reverse=True)
    return dict(scored[:k])


def path_stats(activations: Activations) -> None:
    """
    Prints a per-layer diagnostic summary (index, mean-abs, L2 norm) for
    an activation/path dict.  Call this to inspect path quality before ambale.

    Args:
        activations: Any ``{layer_idx: tensor}`` dict.
    """
    print(f"{'Layer':>6}  {'mean|x|':>10}  {'L2 norm':>10}")
    print("-" * 32)
    for idx in sorted(activations.keys()):
        t = activations[idx]
        print(f"{idx:>6}  {t.abs().mean().item():>10.4f}  {t.norm().item():>10.4f}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _align_to_live(stored: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """
    Aligns ``stored`` sequence length to match ``live`` by trimming or
    zero-padding on the LEFT (matching the left-padding convention used
    throughout the project).
    """
    seq_s = stored.shape[0]
    seq_l = live.shape[0]

    if seq_s == seq_l:
        return stored

    if seq_s > seq_l:
        return stored[seq_s - seq_l:]

    pad = torch.zeros(seq_l - seq_s, stored.shape[1], dtype=stored.dtype, device=stored.device)
    return torch.cat([pad, stored], dim=0)
