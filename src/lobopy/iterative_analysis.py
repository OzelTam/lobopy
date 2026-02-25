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

from typing import Literal, List, Union, Dict, Any, Optional, Iterable, Callable
from .patient import Patient
from tqdm import tqdm
import torch
from .aggregators import Aggregator, Activations
from .models import ContentResponse, StimulationResult, ContentType, IterativeGenerationResult, IterativeAggregatedResult

class IterativeAnalysis:
    """
    A class for doing iterative token generation and analysis.
    """
    patient: Patient
    dataset: Iterable[ContentType]
    match_function: Callable[[str], bool]
    mismatch_function: Callable[[str], bool]
    
    def __init__(self,
    patient: Patient,
    dataset: Iterable[ContentType],
    match_function: Callable[[str], bool],
    mismatch_function: Callable[[str], bool]):
        """
        Initializes the IterativeAnalysis class.
        
        Args:
            patient: The patient to use for analysis.
            dataset: The dataset of contents to iterate over.
            match_function: The function to use for matching tokens. It should return True if the generated text is a match.
            mismatch_function: The function to use for mismatching tokens. It should return True if the generated text is a mismatch.
        
        Example:
            >>> dataset = ["Capital of France is", "My favorite souce is Café de"]
            >>> match_function = lambda x: "Paris" in x # This is a condition for stopping the token generation and capturing activations.
            >>> mismatch_function = lambda x: "Hello" not in x # This is an early exit condition, if this function returns True, the prompt will be skipped.
            >>> iterative_analysis = IterativeAnalysis(patient, dataset, match_function, mismatch_function)
            >>> iterative_analysis.aggregated_analysis(aggregator=mean_aggregator(), max_token_per_prompt=10, match_token_iteration=1, layers=None, show_progress=True, is_parallel=True, max_workers=5)
        """
        self.patient = patient
        self.dataset = dataset
        self.match_function = match_function
        self.mismatch_function = mismatch_function

    def _generate_and_match(self, content_str: str, max_token_per_prompt: int, match_token_iteration: int) -> tuple[Optional[Dict[int, torch.Tensor]], torch.Tensor, torch.Tensor]:
        """
        Processes a single content string manually.
        Returns:
            (activations, input_tokens_cpu, generated_ids_tensor)
        """
        input_ids = self.patient.tokenizer(content_str, return_tensors="pt").input_ids.to(self.patient.llm.device)
        attention_mask = torch.ones_like(input_ids)
        
        past_key_values = None
        generated_ids = []
        
        res_activations = None
        
        thread_acts = self.patient._get_thread_activations()
        for k in thread_acts:
            thread_acts[k].clear()
        is_match = False
        is_mismatch = False
        iteration_count = max_token_per_prompt
        
        for step in range(max_token_per_prompt):
            with self.patient._forward_lock:
                with torch.no_grad():
                    outputs = self.patient.llm(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
            
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            input_ids = next_token
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.patient.llm.device)], dim=-1)
            past_key_values = outputs.past_key_values
            
            generated_ids.append(next_token.item())
            
            if (step + 1) % match_token_iteration == 0:
                text = self.patient.tokenizer.decode(generated_ids, skip_special_tokens=True)
                is_match = self.match_function(text)
                is_mismatch = self.mismatch_function(text)
                
                if is_match and res_activations is None:
                    current_acts = {}
                    for layer_idx, act_list in thread_acts.items():
                        if act_list:
                            cat_tensor = torch.cat(act_list, dim=1) # [1, seq_len, hidden_dim]
                            # remove batch dimension
                            current_acts[layer_idx] = cat_tensor[0] # [seq_len, hidden_dim]
                    
                    if current_acts:
                        res_activations = current_acts
                            
                is_eos = next_token.item() == getattr(self.patient.tokenizer, 'eos_token_id', None)
                if is_mismatch or is_eos or (is_match and res_activations is not None):
                    iteration_count = step + 1
                    break
                    
        # clear thread_acts avoiding memory leak
        for k in thread_acts:
            thread_acts[k].clear()
            
        input_tokens_cpu = self.patient.tokenizer(content_str, return_tensors="pt").input_ids[0].cpu()    
        generated_ids_tensor = torch.tensor(generated_ids, dtype=torch.long)
        
        is_max_token = not is_match and not is_mismatch and len(generated_ids) == max_token_per_prompt
        gen_res = IterativeGenerationResult(
            input_text=content_str,
            output_text=self.patient.tokenizer.decode(generated_ids, skip_special_tokens=True),
            is_matched=is_match,
            is_mismatched=is_mismatch,
            is_max_token=is_max_token,
            iteration_count=iteration_count
        )
        
        return res_activations, input_tokens_cpu, generated_ids_tensor, gen_res

    def _analyse_sequential(self, 
                            contents: List[Tuple[ContentType, str]], 
                            max_token_per_prompt: int,
                            match_token_iteration: int,
                            aggregator_callback: Callable, 
                            initial_state: Any, 
                            label: str, 
                            checkpoint_dir: Optional[str], 
                            save_every: Optional[int], 
                            start_idx: int,
                            desc: str) -> Any:
        
        current_state = initial_state
        for idx, (original_content, content_str) in enumerate(tqdm(contents, desc=f"{desc} {f'({label})' if label else '' } [Seq]")):
            
            res_activations, input_toks, output_toks, gen_res = self._generate_and_match(content_str, max_token_per_prompt, match_token_iteration)
            
            current_state = aggregator_callback(current_state, original_content, res_activations, input_toks, output_toks, gen_res)
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            current_batch_idx = start_idx + idx + 1
            if checkpoint_dir and save_every is not None and save_every > 0 and current_batch_idx % save_every == 0:
                self.patient._save_checkpoint((current_state, current_batch_idx), checkpoint_dir, label)
                
        if checkpoint_dir and len(contents) > 0:
            self.patient._save_checkpoint((current_state, start_idx + len(contents)), checkpoint_dir, label)
            
        return current_state

    def _analyse_parallel(self, 
                          contents: List[Tuple[ContentType, str]], 
                          max_token_per_prompt: int,
                          match_token_iteration: int,
                          aggregator_callback: Callable, 
                          initial_state: Any, 
                          label: str, 
                          max_workers: int, 
                          checkpoint_dir: Optional[str], 
                          save_every: Optional[int], 
                          start_idx: int,
                          desc: str) -> Any:
        import concurrent.futures
        
        current_state = initial_state
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            
            futures = [executor.submit(self._generate_and_match, c[1], max_token_per_prompt, match_token_iteration) for c in contents]
            
            for idx, (future, (original_content, content_str)) in enumerate(tqdm(zip(futures, contents), total=len(contents), desc=f"{desc} {f'({label})' if label else '' } [Par]")):
                res_activations, input_toks, output_toks, gen_res = future.result()
                
                current_state = aggregator_callback(current_state, original_content, res_activations, input_toks, output_toks, gen_res)
                
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                current_batch_idx = start_idx + idx + 1
                if checkpoint_dir and save_every is not None and save_every > 0 and current_batch_idx % save_every == 0:
                     self.patient._save_checkpoint((current_state, current_batch_idx), checkpoint_dir, label)
                     
        if checkpoint_dir and len(contents) > 0:
            self.patient._save_checkpoint((current_state, start_idx + len(contents)), checkpoint_dir, label)
            
        return current_state

    def aggregated_analysis(self,
                            aggregator: Aggregator,
                            max_token_per_prompt: int,
                            match_token_iteration: int = 1,
                            layers: Optional[List[int]] = None,
                            label: Optional[str] = None,
                            show_progress: bool = True,
                            is_parallel: bool = True,
                            max_workers: int = 5,
                            checkpoint_dir: Optional[str] = None,
                            save_checkpoint_every: Optional[int] = None,
                            resume_from_checkpoint: bool = False,
                            populate_generation_results: bool = True
                            ) -> IterativeAggregatedResult:
        """
        Performs iterative token generation and analysis.
        
        Args:
            aggregator: The aggregator to use for combining activations of matched tokens.
            max_token_per_prompt: The maximum number of tokens to generate per prompt until give up if no match found.
            match_token_iteration: The number of tokens to generate between each match check.
            layers: The layers to hook. If None, all layers will be hooked.
            label: The label for the analysis checkpointing.
            show_progress: Whether to show progress.
            is_parallel: Whether to use parallel processing.
            max_workers: The number of workers to use for parallel processing.
            checkpoint_dir: The directory to save checkpoints to.
            save_checkpoint_every: The number of processing iterations to save checkpoints every, or None to not save periodic checkpoints.
            resume_from_checkpoint: Whether to resume from a checkpoint.
            populate_generation_results: Boolean flag to determine whether raw string output results should be attached.
            
        Returns:
            An IterativeAggregatedResult with dictionary of activations and generated strings context.
        """
        
        dataset_to_analyze = [(c, self.patient._format_content(c)) for c in self.dataset]
        label = label if label else f"iter_aggregated_analysis_{id(dataset_to_analyze)}"
        
        current_state = None
        start_idx = 0
        
        if resume_from_checkpoint and checkpoint_dir:
            ckpt = self.patient._load_checkpoint(checkpoint_dir, label)
            if ckpt is not None:
                current_state, start_idx = ckpt

        if current_state is None:
            current_state = IterativeAggregatedResult(activations={})

        remaining_contents = dataset_to_analyze[start_idx:]

        def _aggregation_callback(state: IterativeAggregatedResult, prompt_str, res_activations, input_toks, output_toks, gen_res: IterativeGenerationResult):
            if res_activations is not None:
                if not state.activations:
                    state.activations = res_activations
                else:
                    state.activations = aggregator(state.activations, res_activations)
            if populate_generation_results:
                if state.iterative_generation_results is None:
                    state.iterative_generation_results = []
                state.iterative_generation_results.append(gen_res)
            return state

        self.patient._register_layer_hooks(layers_to_hook=layers)
        
        try:
            if remaining_contents:
                if is_parallel:
                    current_state = self._analyse_parallel(
                         remaining_contents, max_token_per_prompt, match_token_iteration, _aggregation_callback, current_state, label, max_workers, checkpoint_dir, save_checkpoint_every, start_idx, desc="Analysing Generations" if show_progress else ""
                    )
                else:
                    current_state = self._analyse_sequential(
                        remaining_contents, max_token_per_prompt, match_token_iteration, _aggregation_callback, current_state, label, checkpoint_dir, save_checkpoint_every, start_idx, desc="Analysing Generations" if show_progress else ""
                    )
        finally:
            self.patient._clear_hooks()
            
        return current_state if current_state is not None else IterativeAggregatedResult(activations={})

    def dataset_analysis(self,
                        max_token_per_prompt: int,
                        match_token_iteration: int = 1,
                        layers: Optional[List[int]] = None,
                        label: Optional[str] = None,
                        show_progress: bool = True,
                        is_parallel: bool = True,
                        max_workers: int = 5,
                        checkpoint_dir: Optional[str] = None,
                        save_checkpoint_every: Optional[int] = None,
                        resume_from_checkpoint: bool = False
                        ) -> StimulationResult:
        """
        Performs iterative token generation and analysis, returning the input/output tokens and activations.
        
        Args:
            max_token_per_prompt: The maximum number of tokens to generate per prompt until give up if no match found.
            match_token_iteration: The number of tokens to generate between each match check.
            layers: The layers to hook. If None, all layers will be hooked.
            label: The label for the analysis checkpointing.
            show_progress: Whether to show progress.
            is_parallel: Whether to use parallel processing.
            max_workers: The number of workers to use for parallel processing.
            checkpoint_dir: The directory to save checkpoints to.
            save_checkpoint_every: The number of processing iterations to save checkpoints every, or None to not save periodic checkpoints.
            resume_from_checkpoint: Whether to resume from a checkpoint.
            
        Returns:
            A StimulationResult containing ContentResponse objects.
        """
        dataset_to_analyze = [(c, self.patient._format_content(c)) for c in self.dataset]
        label = label if label else f"iter_prompt_analysis_{id(dataset_to_analyze)}"
        
        current_state = []
        start_idx = 0
        
        if resume_from_checkpoint and checkpoint_dir:
            ckpt = self.patient._load_checkpoint(checkpoint_dir, label)
            if ckpt is not None:
                current_state, start_idx = ckpt

        remaining_contents = dataset_to_analyze[start_idx:]

        def _aggregation_callback(state: list, original_content, res_activations, input_toks, output_toks, gen_res):
            if res_activations is not None:
                state.append(ContentResponse(
                    content=original_content,
                    activations=res_activations,
                    input_tokens=input_toks,
                    output_tokens=output_toks
                ))
            return state

        self.patient._register_layer_hooks(layers_to_hook=layers)
        
        try:
            if remaining_contents:
                if is_parallel:
                    current_state = self._analyse_parallel(
                         remaining_contents, max_token_per_prompt, match_token_iteration, _aggregation_callback, current_state, label, max_workers, checkpoint_dir, save_checkpoint_every, start_idx, desc="Analysing Prompts" if show_progress else ""
                    )
                else:
                    current_state = self._analyse_sequential(
                        remaining_contents, max_token_per_prompt, match_token_iteration, _aggregation_callback, current_state, label, checkpoint_dir, save_checkpoint_every, start_idx, desc="Analysing Prompts" if show_progress else ""
                    )
        finally:
            self.patient._clear_hooks()
            
        return StimulationResult(
            results=current_state,
            metadata={
                "model_name": getattr(self.patient.llm.config, "_name_or_path", getattr(self.patient.llm.config, "name_or_path", "unknown")),
                "analysis_type": "iterative_prompt_analysis",
                "label": label
            }
        )
        