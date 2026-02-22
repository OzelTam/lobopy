import torch
from typing import Dict, Callable, Optional, Set


# Type alias for clarity
Activations = Dict[int, torch.Tensor]
Aggregator = Callable[[Activations, Activations], Activations]


def mean_aggregator() -> Aggregator:
    """
    Returns an aggregator that computes the element-wise mean of two activation dicts.

    This is the simplest way to combine activations from two prompts: for each layer
    that appears in both dicts, it averages the tensors token-by-token.
    If the sequence lengths differ, the shorter tensor is padded with zeros on the left
    (matching the left-padding convention used by the tokenizer).

    Use this when you want a balanced blend of two activation patterns.

    Returns:
        Aggregator: fn(a, b) -> {layer: (a[layer] + b[layer]) / 2}
    """
    def aggregate(a: Activations, b: Activations) -> Activations:
        result: Activations = {}
        common_layers = set(a.keys()) & set(b.keys())
        for layer in common_layers:
            ta, tb = _align_tensors(a[layer], b[layer])
            result[layer] = (ta + tb) / 2.0
        return result
    return aggregate


def weighted_mean_aggregator(alpha: float = 0.5) -> Aggregator:
    """
    Returns an aggregator that blends two activation dicts with a configurable weight.

    Instead of a simple 50/50 average, this lets you control how much each activation
    contributes. `alpha` is the weight given to the *first* dict `a`, and `(1 - alpha)`
    is given to `b`.

    Args:
        alpha (float): Weight for `a` in [0, 1]. Default 0.5 (equal blend).
                       alpha=1.0 → only `a` survives; alpha=0.0 → only `b` survives.

    Returns:
        Aggregator: fn(a, b) -> {layer: alpha * a[layer] + (1 - alpha) * b[layer]}
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    def aggregate(a: Activations, b: Activations) -> Activations:
        result: Activations = {}
        common_layers = set(a.keys()) & set(b.keys())
        for layer in common_layers:
            ta, tb = _align_tensors(a[layer], b[layer])
            result[layer] = alpha * ta + (1.0 - alpha) * tb
        return result
    return aggregate


def normalized_overlay_aggregator(eps: float = 1e-8) -> Aggregator:
    """
    Returns an aggregator that computes a direction-preserving normalized overlay.

    Each activation tensor is L2-normalized along the hidden dimension before averaging,
    then the result is re-normalized. This highlights *which directions* in activation
    space are common to both prompts, discarding magnitude differences.

    Useful when you care about the shape/direction of the representation rather than
    raw magnitude — for example, comparing concept directions across different phrasings.

    Args:
        eps (float): Small epsilon added to norms to avoid division by zero.

    Returns:
        Aggregator: fn(a, b) -> normalize(normalize(a[layer]) + normalize(b[layer]))
    """
    def aggregate(a: Activations, b: Activations) -> Activations:
        result: Activations = {}
        common_layers = set(a.keys()) & set(b.keys())
        for layer in common_layers:
            ta, tb = _align_tensors(a[layer], b[layer])
            na = ta / (ta.norm(dim=-1, keepdim=True) + eps)
            nb = tb / (tb.norm(dim=-1, keepdim=True) + eps)
            combined = na + nb
            result[layer] = combined / (combined.norm(dim=-1, keepdim=True) + eps)
        return result
    return aggregate


def common_ground_aggregator(threshold: float = 0.0) -> Aggregator:
    """
    Returns an aggregator that retains only the activation components both prompts agree on.

    For each layer and each hidden dimension, the output is the element-wise minimum of the
    absolute values, with the sign taken from `a`. Components where `a` and `b` have
    opposite signs are zeroed out (they "disagree"). This surfaces the shared subspace
    of two activation patterns.

    Think of it as an activation intersection: what is both prompts collectively "certain"
    about in the same direction.

    Args:
        threshold (float): Minimum absolute value a component must have in *both* tensors
                            to be kept. Default 0.0 (keep everything that agrees in sign).

    Returns:
        Aggregator: fn(a, b) -> element-wise signed minimum where signs match, else 0
    """
    def aggregate(a: Activations, b: Activations) -> Activations:
        result: Activations = {}
        common_layers = set(a.keys()) & set(b.keys())
        for layer in common_layers:
            ta, tb = _align_tensors(a[layer], b[layer])
            same_sign = (ta * tb) > 0                          # [seq, hidden] boolean mask
            magnitude = torch.minimum(ta.abs(), tb.abs())      # smaller magnitude wins
            above_threshold = (ta.abs() >= threshold) & (tb.abs() >= threshold)
            result[layer] = torch.where(same_sign & above_threshold, magnitude * ta.sign(), torch.zeros_like(ta))
        return result
    return aggregate


def dampened_accumulator(damping: float = 0.9) -> Aggregator:
    """
    Returns an aggregator that accumulates activations with exponential dampening.

    Each time this aggregator is called it treats `a` as the running accumulation and `b`
    as the new observation. `a` is dampened by `damping` before adding `b`. Over N calls
    the old activations decay geometrically, so recent prompts have more influence.

    Use this as the `aggregator` argument to `Patient.analyse()` when processing a list
    of prompts — the result will be a recency-weighted summary of all activations seen.

    Args:
        damping (float): Decay factor in (0, 1] applied to the accumulated tensor `a`
                         each step. Default 0.9 (mild decay; 1.0 = no decay).

    Returns:
        Aggregator: fn(a, b) -> {layer: damping * a[layer] + b[layer]}
    """
    if not (0.0 < damping <= 1.0):
        raise ValueError(f"damping must be in (0, 1], got {damping}")

    def aggregate(a: Activations, b: Activations) -> Activations:
        result: Activations = {}
        common_layers = set(a.keys()) & set(b.keys())
        for layer in common_layers:
            ta, tb = _align_tensors(a[layer], b[layer])
            result[layer] = damping * ta + tb
        return result
    return aggregate


def difference_aggregator(absolute: bool = False) -> Aggregator:
    """
    Returns an aggregator that computes the activation difference between two prompts.

    Subtracts `b` from `a` layer-by-layer. This is useful for finding *contrastive*
    directions: what activation pattern separates one concept from another. The result
    can be used as a steering vector or for probing.

    Args:
        absolute (bool): If True, return the absolute difference |a - b|.
                         If False (default), return the signed difference a - b.

    Returns:
        Aggregator: fn(a, b) -> {layer: a[layer] - b[layer]}  (or |.|)
    """
    def aggregate(a: Activations, b: Activations) -> Activations:
        result: Activations = {}
        common_layers = set(a.keys()) & set(b.keys())
        for layer in common_layers:
            ta, tb = _align_tensors(a[layer], b[layer])
            diff = ta - tb
            result[layer] = diff.abs() if absolute else diff
        return result
    return aggregate


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _align_tensors(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aligns two tensors along the sequence dimension (dim 0) by padding the shorter one.

    Padding is added on the *left* to match the left-padding convention used by the
    tokenizer in patient.py. The hidden dimension must already match.

    Args:
        a: Tensor of shape (seq_len_a, hidden_dim)
        b: Tensor of shape (seq_len_b, hidden_dim)

    Returns:
        Tuple of tensors both with shape (max(seq_len_a, seq_len_b), hidden_dim)
    """
    len_a, len_b = a.shape[0], b.shape[0]
    if len_a == len_b:
        return a, b
    max_len = max(len_a, len_b)
    if len_a < max_len:
        pad = torch.zeros(max_len - len_a, a.shape[1], dtype=a.dtype, device=a.device)
        a = torch.cat([pad, a], dim=0)
    if len_b < max_len:
        pad = torch.zeros(max_len - len_b, b.shape[1], dtype=b.dtype, device=b.device)
        b = torch.cat([pad, b], dim=0)
    return a, b