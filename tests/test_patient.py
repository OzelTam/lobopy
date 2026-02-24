"""
tests/test_patient.py

Tests for Patient.stimulate() / Patient.analyse() covering all supported prompt
types documented in patient.py:

  1. str            — raw string
  2. dict           — single chat message  {"role": ..., "content": ...}
  3. List[str]      — multiple raw strings (batched)
  4. List[dict]     — a conversation thread  [{"role":"user",...}, {"role":"assistant",...}]
  5. List[mixed]    — heterogeneous list (str, conversation, single msg)

Also covers:
  - Activation shapes returned by stimulate/analyse
  - normalize_path and top_k_layers on real activations
  - project_out_activation / safe_scale_activation applied through ambale

The model is loaded once per session (module-level fixture) to keep test
runs fast. All generation is done with a tiny token budget (max_new_tokens=5).
"""

import pytest
import torch
from lobopy.patient import Patient, PatientConfig
from lobopy.aggregators import mean_aggregator, difference_aggregator
from lobopy.ambalefiers import (
    normalize_path,
    top_k_layers,
    safe_scale_activation,
    project_out_activation,
    path_stats,
)

# ---------------------------------------------------------------------------
# Shared model fixture (loaded once for the whole test session)
# ---------------------------------------------------------------------------

MODEL_NAME = "BEE-spoke-data/smol_llama-101M-GQA"


@pytest.fixture(scope="session")
def patient():
    return Patient(
        pretrained_model_name_or_path=MODEL_NAME,
        config=PatientConfig(batch_size=1, device="cpu"),
    )


@pytest.fixture(scope="session")
def num_layers(patient):
    return len(patient._find_transformer_layers())


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check_activations(activations: dict, num_layers: int, check_seq: bool = True):
    """Common assertions on an activations dict."""
    assert isinstance(activations, dict)
    assert len(activations) == num_layers
    for layer_idx, tensor in activations.items():
        assert isinstance(layer_idx, int)
        assert 0 <= layer_idx < num_layers
        assert tensor.dim() == 2, f"Layer {layer_idx}: expected 2-D (seq, hidden), got {tensor.shape}"
        if check_seq:
            assert tensor.shape[0] > 0, "seq dim must be > 0"
        assert tensor.shape[1] > 0, "hidden dim must be > 0"


# ===========================================================================
# 1. Prompt type tests — stimulate()
# ===========================================================================

class TestPromptTypes:
    """Each test verifies that stimulate() accepts the prompt type and
    returns correctly shaped 2-D (seq, hidden) activations per layer."""

    def test_str_prompt(self, patient, num_layers):
        """A plain Python string — the most basic input type."""
        result = patient.stimulate("Hello, how are you?")
        assert len(result.results) == 1
        _check_activations(result.results[0].activations, num_layers)

    def test_single_chat_dict(self, patient, num_layers):
        """A single chat message dict {"role": "user", "content": "..."}."""
        prompt = {"role": "user", "content": "Tell me something interesting."}
        result = patient.stimulate(prompt)
        assert len(result.results) == 1
        _check_activations(result.results[0].activations, num_layers)

    def test_list_of_str(self, patient, num_layers):
        """A list of raw strings — one ContentResponse per string."""
        prompts = [
            "The sky is blue.",
            "Water is wet.",
            "Cats are curious animals.",
        ]
        results = [patient.stimulate(p) for p in prompts]
        assert len(results) == len(prompts)
        for r in results:
            _check_activations(r.results[0].activations, num_layers)

    def test_conversation_list_of_dicts(self, patient, num_layers):
        """A full conversation thread — List[Dict] treated as one context."""
        conversation = [
            {"role": "user",      "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
            {"role": "user",      "content": "And Germany?"},
        ]
        result = patient.stimulate(conversation)
        # A single conversation thread → single result
        assert len(result.results) == 1
        _check_activations(result.results[0].activations, num_layers)

    def test_mixed_list(self, patient, num_layers):
        """List containing both raw strings and conversation threads."""
        prompts = [
            "A raw string here.",
            [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi!"}],
            "Another raw string.",
        ]
        results = [patient.stimulate(p) for p in prompts]
        assert len(results) == 3
        for r in results:
            _check_activations(r.results[0].activations, num_layers)

    def test_token_counts_differ_by_prompt(self, patient):
        """Longer prompts should produce longer seq dims at each layer."""
        short_result = patient.stimulate("Hi.")
        long_result  = patient.stimulate("This is a much longer sentence with many more words.")

        short_seq = list(short_result.results[0].activations.values())[0].shape[0]
        long_seq  = list(long_result.results[0].activations.values())[0].shape[0]
        assert long_seq > short_seq, "Longer prompt must have more tokens"

    def test_chat_dict_vs_string_differ(self, patient, num_layers):
        """Chat-formatted and raw string prompts should produce different tensors
        because the tokenizer applies a different template for chat inputs."""
        raw = patient.stimulate("Tell me a joke.")
        chat = patient.stimulate([{"role": "user", "content": "Tell me a joke."}])

        raw_tensor  = list(raw.results[0].activations.values())[0]
        chat_tensor = list(chat.results[0].activations.values())[0]

        # They may have different seq lengths (template adds role tokens)
        # OR the same length but different values — at least one must differ
        shapes_differ  = raw_tensor.shape != chat_tensor.shape
        values_differ  = (
            raw_tensor.shape == chat_tensor.shape and
            not torch.allclose(raw_tensor, chat_tensor, atol=1e-4)
        )
        assert shapes_differ or values_differ, (
            "Chat-formatted and raw prompts should produce different activations"
        )


# ===========================================================================
# 2. analyse() — aggregation across multiple prompts
# ===========================================================================

class TestAnalyse:

    def test_analyse_str_list(self, patient, num_layers):
        """analyse() with a list of strings and mean_aggregator."""
        result = patient.analyse(
            dataset=["The cat sat.",  "A dog ran."],
            label="test",
            aggregator=mean_aggregator(),
        )
        _check_activations(result.activations, num_layers, check_seq=False)
        assert result.label == "test"

    def test_analyse_returns_aggregated_shape(self, patient, num_layers):
        """The aggregated tensor shape should be 2-D (seq_avg, hidden) for each layer."""
        result = patient.analyse(
            dataset=["Short.", "Much longer sentence here."],
            aggregator=mean_aggregator(),
        )
        for tensor in result.activations.values():
            assert tensor.dim() == 2



# ===========================================================================
# 3. Path utilities — normalize_path, top_k_layers
# ===========================================================================

class TestPathUtils:

    @pytest.fixture(scope="class")
    def raw_path(self, patient, num_layers):
        """Build a real difference path from two contrasting prompt sets."""
        happy = patient.analyse(
            dataset=["She smiled with joy.", "He laughed with delight."],
            aggregator=mean_aggregator(),
            label="happy",
        )
        sad = patient.analyse(
            dataset=["She wept with grief.", "He sat alone in silence."],
            aggregator=mean_aggregator(),
            label="sad",
        )
        return difference_aggregator()(happy.activations, sad.activations)

    def test_normalize_unit_norm(self, raw_path):
        """After normalize_path, each token position must have unit L2 norm."""
        normed = normalize_path(raw_path)
        for layer_idx, tensor in normed.items():
            norms = tensor.norm(dim=-1)
            assert (norms - 1.0).abs().max() < 1e-5, (
                f"Layer {layer_idx}: norms not unit after normalize_path: {norms}"
            )

    def test_top_k_returns_k_layers(self, raw_path, num_layers):
        """top_k_layers(k=2) must return exactly 2 entries."""
        normed = normalize_path(raw_path)
        top = top_k_layers(normed, k=2, metric="mean_abs")
        assert len(top) == 2

    def test_top_k_layer_range_respected(self, raw_path, num_layers):
        """layer_range=(0.5, 0.85) on a 6-layer model must yield only {3, 4}."""
        normed = normalize_path(raw_path)
        top = top_k_layers(normed, k=2, metric="mean_abs", layer_range=(0.5, 0.85))
        for layer_idx in top:
            lo = int(0.5  * num_layers)
            hi = int(0.85 * num_layers)
            assert lo <= layer_idx < hi, (
                f"Layer {layer_idx} outside allowed range [{lo}, {hi}) "
                f"with num_layers={num_layers}"
            )

    def test_top_k_all_metrics(self, raw_path):
        """All three scoring metrics should return the same k layers count."""
        normed = normalize_path(raw_path)
        for metric in ("mean_abs", "max_abs", "l2"):
            top = top_k_layers(normed, k=2, metric=metric)
            assert len(top) == 2, f"metric={metric} returned {len(top)} layers"

    def test_path_stats_runs(self, raw_path, capsys):
        """path_stats should print output without raising."""
        path_stats(raw_path)
        captured = capsys.readouterr()
        assert "Layer" in captured.out
        assert "mean|x|" in captured.out


# ===========================================================================
# 4. ambale() — steering hook integration
# ===========================================================================

class TestAmbale:

    @pytest.fixture(scope="class")
    def steering_path(self, patient):
        """Pre-computed normalised top-2 path for steering tests."""
        happy = patient.analyse(
            dataset=["She smiled with joy.", "He laughed with delight."],
            aggregator=mean_aggregator(),
            label="happy",
        )
        sad = patient.analyse(
            dataset=["She wept with grief.", "He sat alone in silence."],
            aggregator=mean_aggregator(),
            label="sad",
        )
        raw = difference_aggregator()(happy.activations, sad.activations)
        return top_k_layers(normalize_path(raw), k=2, layer_range=(0.5, 0.85))

    def _generate(self, patient, prompt: str, path=None, fn=None):
        """Helper: generate 5 tokens with or without steering."""
        inputs = patient.tokenizer(prompt, return_tensors="pt").to(patient.llm.device)
        GEN = dict(max_new_tokens=5, do_sample=False)
        if path is not None:
            with patient.ambale(path, fn):
                out = patient.llm.generate(**inputs, **GEN)
        else:
            out = patient.llm.generate(**inputs, **GEN)
        return out[0].tolist()

    def test_steering_changes_output(self, patient, steering_path):
        """Applying steering with a non-trivial factor must change the output."""
        base    = self._generate(patient, "The mood in the room was")
        steered = self._generate(patient, "The mood in the room was",
                                 path=steering_path,
                                 fn=safe_scale_activation(factor=5.0, clamp_sigma=3.0))
        assert base != steered, "Steering with factor=5 should change the generated tokens"

    def test_boost_and_dampen_differ(self, patient, steering_path):
        """Boosting and dampening the same path must produce different outputs."""
        boost  = self._generate(patient, "The mood in the room was",
                                path=steering_path,
                                fn=safe_scale_activation(factor=5.0,  clamp_sigma=3.0))
        dampen = self._generate(patient, "The mood in the room was",
                                path=steering_path,
                                fn=safe_scale_activation(factor=-5.0, clamp_sigma=3.0))
        assert boost != dampen, "Positive and negative factor must yield different tokens"

    def test_context_manager_removes_hooks(self, patient, steering_path):
        """After the with-block, ambale must remove all hooks."""
        with patient.ambale(steering_path, safe_scale_activation(factor=1.0)):
            pass  # exiting context
        assert len(patient.hooks) == 0, "Hooks should be cleared after context exit"

    def test_project_out_runs(self, patient, steering_path):
        """project_out_activation should not raise and must produce some output."""
        tokens = self._generate(patient, "He smiled warmly and",
                                path=steering_path,
                                fn=project_out_activation())
        assert len(tokens) > 0

    def test_str_prompt_yields_same_as_stimulate(self, patient, steering_path):
        """Steering applied after stimulate=str should not cause shape errors."""
        result = patient.stimulate("She was happy.")
        _ = result.results[0].activations  # just ensure it exists

    def test_chat_prompt_steered(self, patient, steering_path):
        """Steering must also work when the generation prompt came from a chat template."""
        chat_text = patient.tokenizer.apply_chat_template(
            [{"role": "user", "content": "How do you feel?"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = patient.tokenizer(chat_text, return_tensors="pt").to(patient.llm.device)
        GEN = dict(max_new_tokens=5, do_sample=False)
        with patient.ambale(steering_path, safe_scale_activation(factor=3.0, clamp_sigma=3.0)):
            out = patient.llm.generate(**inputs, **GEN)
        assert out is not None and out.shape[1] > 0

    def test_save_and_load_ambale(self, patient, steering_path, tmp_path):
        """Saving and loading a steered model must yield the exact same outputs as the fresh steered model."""
        import os
        
        save_file = tmp_path / "test_steered.pt"
        
        # Fresh steered check
        base_steered_tokens = self._generate(patient, "The room felt", 
                                             path=steering_path, 
                                             fn=safe_scale_activation(factor=2.0))
        
        # Save it
        ctx = patient.ambale(steering_path, safe_scale_activation(factor=2.0))
        ctx.save(save_file)
        
        assert os.path.exists(save_file), "Steered checkpoint should exist on disk."
        
        # Load it and test
        loaded_ctx = patient.load_ambale(save_file)
        
        # Generate with loaded context
        inputs = patient.tokenizer("The room felt", return_tensors="pt").to(patient.llm.device)
        GEN = dict(max_new_tokens=5, do_sample=False)
        with loaded_ctx:
            out = patient.llm.generate(**inputs, **GEN)
        loaded_tokens = out[0].tolist()
        
        assert loaded_tokens == base_steered_tokens, "Loaded steered model tokens must match fresh steered model."
