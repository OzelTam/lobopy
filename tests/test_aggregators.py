import pytest
import torch
from lobopy.aggregators import (
    mean_aggregator,
    weighted_mean_aggregator,
    normalized_overlay_aggregator,
    common_ground_aggregator,
    dampened_accumulator,
    difference_aggregator,
    _align_tensors
)

def test_align_tensors():
    a = torch.ones(2, 4)
    b = torch.ones(3, 4)
    aa, bb = _align_tensors(a, b)
    assert aa.shape == (3, 4)
    assert bb.shape == (3, 4)
    assert (aa[0] == 0).all() # Pad left

def test_mean_aggregator():
    a = {0: torch.ones(2, 4) * 2}
    b = {0: torch.ones(2, 4) * 4}
    agg = mean_aggregator()
    res = agg(a, b)
    assert torch.allclose(res[0], torch.ones(2, 4) * 3)

def test_weighted_mean_aggregator():
    a = {0: torch.ones(2, 4) * 10}
    b = {0: torch.ones(2, 4) * 0}
    agg = weighted_mean_aggregator(alpha=0.8)
    res = agg(a, b)
    assert torch.allclose(res[0], torch.ones(2, 4) * 8)

def test_normalized_overlay_aggregator():
    a = {1: torch.tensor([[3.0, 4.0]])}
    b = {1: torch.tensor([[0.0, 5.0]])}
    agg = normalized_overlay_aggregator()
    res = agg(a, b)
    assert torch.allclose(res[1].norm(dim=-1), torch.ones(1), atol=1e-5)

def test_common_ground_aggregator():
    a = {0: torch.tensor([[2.0, -3.0, 4.0]])}
    b = {0: torch.tensor([[5.0, 1.0, -4.0]])}
    agg = common_ground_aggregator(threshold=1.0)
    res = agg(a, b)
    # same sign for index 0 (keeps min: 2.0), diff sign for index 1,2 (zeroed)
    expected = torch.tensor([[2.0, 0.0, 0.0]])
    assert torch.allclose(res[0], expected)

def test_dampened_accumulator():
    a = {0: torch.ones(2, 4) * 10}
    b = {0: torch.ones(2, 4) * 2}
    agg = dampened_accumulator(damping=0.5)
    res = agg(a, b)
    # 10 * 0.5 + 2 = 7
    assert torch.allclose(res[0], torch.ones(2, 4) * 7)

def test_difference_aggregator():
    a = {0: torch.ones(2, 4) * 5}
    b = {0: torch.ones(2, 4) * 2}
    agg1 = difference_aggregator(absolute=False)
    assert torch.allclose(agg1(a, b)[0], torch.ones(2, 4) * 3)
    
    a2 = {0: torch.ones(2, 4) * -2}
    agg2 = difference_aggregator(absolute=True)
    assert torch.allclose(agg2(a2, b)[0], torch.ones(2, 4) * 4)
