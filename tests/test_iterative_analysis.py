import pytest
import os
import torch
import shutil
from unittest.mock import patch, MagicMock
from lobopy.patient import Patient, PatientConfig
from lobopy.iterative_analysis import IterativeAnalysis
from lobopy.aggregators import mean_aggregator

# Model for fast testing, utilizing a tinier LLM specifically for speed.
TINY_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# TinyLlama has 22 transformer layers. We use the last layer for quick verification.
LAST_LAYER = [21] 

@pytest.fixture(scope="module")
def patient():
    config = PatientConfig(batch_size=1, device="cpu", torch_dtype="float32")
    return Patient(pretrained_model_name_or_path=TINY_MODEL, config=config)

@pytest.fixture
def temp_checkpoint_dir():
    dir_name = "test_checkpoints"
    os.makedirs(dir_name, exist_ok=True)
    yield dir_name
    shutil.rmtree(dir_name)

class TestIterativeAnalysis:

    def test_initialization(self, patient):
        dataset = ["test 1", "test 2"]
        def match_fn(x): return "foo" in x
        def mismatch_fn(x): return "bar" in x
        
        analysis = IterativeAnalysis(patient, dataset, match_fn, mismatch_fn)
        assert analysis.patient is patient
        assert analysis.dataset == dataset
        assert analysis.match_function is match_fn
        assert analysis.mismatch_function is mismatch_fn

    def test_aggregated_analysis_sequential(self, patient):
        dataset = ["The sky is", "Water is"]
        
        def match_fn(text):
            # Target words it will likely generate
            return "blue" in text.lower() or "wet" in text.lower()
            
        def mismatch_fn(text):
            # Early exit if the sentence ends
            return "." in text
            
        analysis = IterativeAnalysis(patient, dataset, match_fn, mismatch_fn)
        
        result_acts = analysis.aggregated_analysis(
            aggregator=mean_aggregator(),
            max_token_per_prompt=5,
            match_token_iteration=1,
            layers=LAST_LAYER,
            is_parallel=False,
            show_progress=False
        )
        
        assert hasattr(result_acts, "activations")
        assert hasattr(result_acts, "iterative_generation_results")
        acts = result_acts.activations
        
        assert isinstance(acts, dict)
        if acts:
             # If it found matches before early-exiting, verify tensor bounds
             assert LAST_LAYER[0] in acts
             tensor = acts[LAST_LAYER[0]]
             assert type(tensor) == torch.Tensor
             assert tensor.ndim == 2  # (seq_len, hidden_dim)

    def test_dataset_analysis_parallel(self, patient):
        dataset = ["A cat says", "A dog says"]
        
        def match_fn(text):
            return "meow" in text.lower() or "bark" in text.lower() or "woof" in text.lower()
            
        def mismatch_fn(text):
            return "moo" in text.lower()
            
        analysis = IterativeAnalysis(patient, dataset, match_fn, mismatch_fn)
        
        result = analysis.dataset_analysis(
            max_token_per_prompt=6,
            match_token_iteration=2,
            layers=LAST_LAYER,
            is_parallel=True,
            max_workers=2,
            show_progress=False
        )
        
        assert hasattr(result, "results")
        # Since these tokens may or may not be hit, just assert we got a valid object
        if result.results:
            response = result.results[0]
            assert hasattr(response, "content")
            assert hasattr(response, "activations")
            assert hasattr(response, "input_tokens")
            assert hasattr(response, "output_tokens")

    def test_early_exit_mismatch(self, patient):
        dataset = ["This will instantly bail."]
        
        def match_fn(text):
            return False # Never matches
            
        def mismatch_fn(text):
            return True # Instantly bails
            
        analysis = IterativeAnalysis(patient, dataset, match_fn, mismatch_fn)
        
        result_acts = analysis.aggregated_analysis(
            aggregator=mean_aggregator(),
            max_token_per_prompt=10,
            layers=LAST_LAYER,
            is_parallel=False,
            show_progress=False
        )
        
        # It should bail immediately, returning empty activations
        assert result_acts.activations == {}

    def test_checkpointing_triggered(self, patient, temp_checkpoint_dir):
        # We process 3 files and save every 1.
        dataset = ["One", "Two", "Three"]
        
        def match_fn(text): return False
        def mismatch_fn(text): return True
            
        analysis = IterativeAnalysis(patient, dataset, match_fn, mismatch_fn)
        
        label = "test_cp"
        analysis.aggregated_analysis(
            aggregator=mean_aggregator(),
            max_token_per_prompt=2,
            layers=LAST_LAYER,
            is_parallel=False,
            show_progress=False,
            checkpoint_dir=temp_checkpoint_dir,
            save_checkpoint_every=1,
            label=label
        )
        
        # Checkpoint directory should contain saved '.pt' states.
        files = os.listdir(temp_checkpoint_dir)
        matching_files = [f for f in files if f.endswith(".pt") and label in f]
        assert len(matching_files) > 0

    def test_checkpoint_load_continue(self, patient, temp_checkpoint_dir):
        # We process 3 files and simulate a restart to see if it skips successfully.
        dataset = ["One", "Two", "Three"]
        
        def match_fn(text): return False
        def mismatch_fn(text): return True
        
        analysis = IterativeAnalysis(patient, dataset, match_fn, mismatch_fn)
        
        label = "test_cp_resume"
        
        # We will mock `_generate_and_match` to track how many times it was actually called.
        with patch.object(analysis, '_generate_and_match', wraps=analysis._generate_and_match) as mock_generate:
            analysis.aggregated_analysis(
                aggregator=mean_aggregator(),
                max_token_per_prompt=2,
                layers=LAST_LAYER,
                is_parallel=False,
                show_progress=False,
                checkpoint_dir=temp_checkpoint_dir,
                save_checkpoint_every=1,
                label=label
            )
            # Should have called it 3 times for a fresh run
            assert mock_generate.call_count == 3
            
        # The checkpoint logic explicitly saves the (current_state, start_idx + len(remaining_prompts)) at the very end.
        # So wait, if it finished completely, the start_idx saved is 3. 
        # Let's manually overwrite the checkpoint to fake a "died halfway" state at start_idx=2 (Meaning 2 items processed: idx 0 and 1).
        # We can just fetch the checkpoint, change the idx, and resave it.
        ckpt_path = os.path.join(temp_checkpoint_dir, f"{label}_checkpoint.pt")
        state, old_idx = torch.load(ckpt_path, weights_only=False)
        torch.save((state, 2), ckpt_path) # Force it to act like it is resuming exactly at index "2" (the third item).
        
        # Now create a fresh analysis object with the exact same dataset
        analysis2 = IterativeAnalysis(patient, dataset, match_fn, mismatch_fn)
        
        with patch.object(analysis2, '_generate_and_match', wraps=analysis2._generate_and_match) as mock_generate_resume:
            analysis2.aggregated_analysis(
                aggregator=mean_aggregator(),
                max_token_per_prompt=2,
                layers=LAST_LAYER,
                is_parallel=False,
                show_progress=False,
                checkpoint_dir=temp_checkpoint_dir,
                save_checkpoint_every=1,
                resume_from_checkpoint=True,
                label=label
            )
            
            # Since it resumed from index 2, there is only 1 item remaining in a length 3 array (the item at idx 2).
            # Therefore, _generate_and_match should only have been called EXACTLY ONCE!
            assert mock_generate_resume.call_count == 1

