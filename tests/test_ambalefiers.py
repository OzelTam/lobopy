import pytest
import torch
from lobopy.ambalefiers import (
    scale_activation,
    dampen_activation,
    replace_activation,
    project_out_activation,
    project_in_activation,
    clamp_activation,
    blend_activation,
    safe_scale_activation,
    normalize_path,
    top_k_layers,
    path_stats,
    _align_to_live
)

def test_align_to_live():
    stored = torch.ones(3, 4)
    live = torch.ones(2, 4)
    res = _align_to_live(stored, live)
    assert res.shape == (2, 4)
    
    live_longer = torch.ones(5, 4)
    res2 = _align_to_live(stored, live_longer)
    assert res2.shape == (5, 4)
    assert (res2[:2] == 0).all() # Pad left

def test_scale_activation():
    stored = torch.ones(2, 4) * 2
    live = torch.ones(2, 4) * 5
    fn = scale_activation(factor=2.0)
    out = fn(stored, live)
    assert torch.allclose(out, torch.ones(2, 4) * 9.0)

def test_dampen_activation():
    stored = torch.ones(2, 4) * 2
    live = torch.ones(2, 4) * 5
    fn = dampen_activation(factor=1.0)
    out = fn(stored, live)
    assert torch.allclose(out, torch.ones(2, 4) * 3.0)

def test_replace_activation():
    stored = torch.ones(2, 4) * 2
    live = torch.ones(2, 4) * 5
    fn = replace_activation()
    out = fn(stored, live)
    assert torch.allclose(out, torch.ones(2, 4) * 2.0)

def test_project_out_activation():
    stored = torch.tensor([[1.0, 0.0]])
    live = torch.tensor([[2.0, 3.0]])
    fn = project_out_activation()
    out = fn(stored, live)
    assert torch.allclose(out, torch.tensor([[0.0, 3.0]]))

def test_project_in_activation():
    stored = torch.tensor([[1.0, 0.0]])
    live = torch.tensor([[2.0, 3.0]])
    fn = project_in_activation()
    out = fn(stored, live)
    assert torch.allclose(out, torch.tensor([[2.0, 0.0]]))

def test_clamp_activation():
    stored = torch.ones(2, 4) * 2
    live = torch.ones(2, 4) * 5
    fn = clamp_activation(min_val=-1.0, max_val=1.5)
    # bounds will be [-2, 3]
    out = fn(stored, live)
    assert torch.allclose(out, torch.ones(2, 4) * 3.0)

def test_blend_activation():
    stored = torch.ones(2, 4) * 2
    live = torch.ones(2, 4) * 6
    fn = blend_activation(alpha=0.5)
    out = fn(stored, live)
    assert torch.allclose(out, torch.ones(2, 4) * 4.0)

def test_safe_scale_activation():
    stored = torch.ones(2, 4) * 100
    live = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    # std deviation is around 2.45
    fn = safe_scale_activation(factor=1.0, clamp_sigma=1.0)
    out = fn(stored, live)
    # Delta will be clamped to std
    diff = out - live
    assert diff.abs().max() <= live.std() + 1e-4

def test_normalize_path():
    activations = {0: torch.tensor([[3.0, 4.0]])}
    normed = normalize_path(activations)
    assert torch.allclose(normed[0].norm(dim=-1), torch.ones(1))

def test_top_k_layers():
    acts = {
        0: torch.ones(1, 4) * 1,
        1: torch.ones(1, 4) * 10,
        2: torch.ones(1, 4) * 5,
        3: torch.ones(1, 4) * 0
    }
    top = top_k_layers(acts, k=2, metric="mean_abs")
    assert set(top.keys()) == {1, 2}
    
    # Layer range test
    top2 = top_k_layers(acts, k=1, metric="mean_abs", layer_range=(0.0, 0.5))
    # range 0.0 to 0.5 allows indices 0, 1 (total layers=4)
    # layer 1 is highest in that range
    assert set(top2.keys()) == {1}
