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

from transformers import AutoTokenizer, AutoModelForCausalLM
import os, torch, gc, threading
from typing import Literal, List, Union, Dict, Any, Optional, Iterable, Callable, Tuple
import torch.nn as nn
from dataclasses import dataclass, field
from tqdm import tqdm
from .models import AnalysisResult, PatientConfig, ContentResponse, StimulationResult, ContentType
from .aggregators import Aggregator

# ---------------------------------------------------------------------------
# Context manager returned by Patient.ambale()
# ---------------------------------------------------------------------------

class AmbaleContext:
    """
    Returned by ``Patient.ambale()``. Lazily registers hooks when used as a context manager 
    or when calling methods like ``generate()``.

    Usage as context manager:

        with model.ambale(activations=happy_path, applied_function=scale_activation(2.0)) as happy_model:
            output = happy_model.generate(...)

    Usage as an independent object:

        happy_model = model.ambale(activations=emotion_path, applied_function=dampen_activation(1.0))
        output = happy_model.generate(...)
        # The base 'model' remains unaffected outside of generation.
    """

    def __init__(self, patient: "Patient", activations: Dict[int, torch.Tensor], applied_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self._patient = patient
        self._activations = activations
        self._applied_function = applied_function
        self._handles = []
        self._active_count = 0

    def __enter__(self) -> "AmbaleContext":
        if self._active_count == 0:
            self._register_hooks()
        self._active_count += 1
        return self

    def __exit__(self, *_) -> None:
        self._active_count -= 1
        if self._active_count == 0:
            self._remove_hooks()
            
    def _register_hooks(self):
        transformer_layers = self._patient._find_transformer_layers()
        num_layers = len(transformer_layers)

        for layer_idx, stored_tensor in self._activations.items():
            abs_idx = layer_idx if layer_idx >= 0 else layer_idx + num_layers
            if not (0 <= abs_idx < num_layers):
                raise IndexError(
                    f"Layer index {layer_idx} is out of range for a model with "
                    f"{num_layers} transformer layers."
                )

            layer = transformer_layers[abs_idx]

            # Capture loop variables
            def _make_hook(
                _stored: torch.Tensor,
                _fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            ):
                def hook(module, inputs, output):
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                        rest = output[1:]
                    else:
                        hidden_states = output
                        rest = None

                    if hidden_states.dim() == 3:
                        batch_size = hidden_states.shape[0]
                        modified_list = []
                        for b in range(batch_size):
                            out_b = _fn(_stored, hidden_states[b])  # (seq, hidden)
                            modified_list.append(out_b)
                        modified = torch.stack(modified_list, dim=0)  # (batch, seq, hidden)
                    else:
                        modified = _fn(_stored, hidden_states)

                    if rest is not None:
                        return (modified,) + rest
                    return modified

                return hook

            handle = layer.register_forward_hook(_make_hook(stored_tensor, self._applied_function))
            self._handles.append(handle)

    def _remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def generate(self, *args, **kwargs):
        """
        Temporarily registers hooks (if not already active), generates tokens, and removes hooks.
        """
        with self:
            return self._patient.generate(*args, **kwargs)

    def save(self, path: Union[str, os.PathLike[str]]):
        """
        Saves the steering configuration to disk using safetensors.
        """
        from safetensors.torch import save_file
        import json
        import warnings
        
        factory_name = getattr(self._applied_function, "factory_name", "custom")
        kwargs = getattr(self._applied_function, "kwargs", {})
        
        if factory_name == "custom":
            warnings.warn("The applied function does not have serialization metadata (not from lobopy.ambalefiers). Saved as 'custom'.")
            
        tensors_dict = {f"layer_{k}": v.contiguous() for k, v in self._activations.items()}
        
        metadata = {
            "applied_function_factory": factory_name,
            "applied_function_kwargs": json.dumps(kwargs),
            "model_name_or_path": getattr(self._patient.llm.config, "_name_or_path", getattr(self._patient.llm.config, "name_or_path", "unknown")),
            "version": "1.1"
        }
        
        save_file(tensors_dict, path, metadata=metadata)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._patient, name)



class Patient:

    def __init__(self, 
                 pretrained_model_name_or_path: str | os.PathLike[str],
                 config: Optional[PatientConfig] = None):
        

        self.config = config or PatientConfig()

        
        self.llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            output_attentions=True,
            output_hidden_states=True,
            torch_dtype=self.config.torch_dtype,
            device_map=self.config.device,
            trust_remote_code=self.config.trust_remote_code,
            **self.config.llm_kwargs
        )        
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,  
            trust_remote_code=self.config.trust_remote_code,
            **self.config.tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" # Important for batch generation
        
        # Set default chat template if missing to support chat-based diagnostics
        if self.tokenizer.chat_template is None:
            # Simple fallback template: Role: Content\n
            self.tokenizer.chat_template = "{% for message in messages %}\n{{ message['role']|title }}: {{ message['content'] }}{% endfor %}\n{% if add_generation_prompt %}Assistant: {% endif %}"
            
        self.hooks = []
        self._thread_local = threading.local()
        self._forward_lock = threading.Lock()
        
    def _get_thread_activations(self) -> Dict[int, list]:
        """Helper to safely get or initialize thread-local layer activations."""
        if not hasattr(self._thread_local, 'layer_activations'):
            self._thread_local.layer_activations = {}
        return self._thread_local.layer_activations

    def _find_transformer_layers(self) -> nn.ModuleList:
        """
        Robustly finds the ModuleList containing transformer blocks.
        """
        # 1. Try known paths
        known_paths = [
            "model.layers",          # Llama, Mistral, Qwen
            "transformer.h",         # GPT-2, Falcon
            "model.decoder.layers",  # OPT
            "gpt_neox.layers",       # GPT-NeoX
        ]
        
        for path in known_paths:
            module = self.llm
            try:
                sub_module = module
                for part in path.split("."):
                    sub_module = getattr(sub_module, part)
                if isinstance(sub_module, nn.ModuleList):
                    return sub_module
            except AttributeError:
                continue

        # 2. Heuristic fallback
        candidates = []
        for name, module in self.llm.named_modules():
            if isinstance(module, nn.ModuleList):
                if len(module) >= self.config.transformer_detection_threshold:
                    candidates.append((name, module))

        if not candidates:
            raise ValueError("No transformer layer ModuleList found. Please specify manually or check model architecture.")

        # Pick the deepest (most specific) one
        candidates.sort(key=lambda x: x[0].count("."))
        return candidates[-1][1]
    
    def _register_layer_hooks(self, layers_to_hook: Optional[List[int]] = None):
        """
        Registers forward hooks to capture activations.
        """
        self._clear_hooks()
        transformer_layers = self._find_transformer_layers()
        
        if layers_to_hook is None:
            layers_to_hook = range(len(transformer_layers))

        # Normalize layers_to_hook to positive indices
        normalized_layers = []
        
        for idx in layers_to_hook:
            abs_idx = idx
            if abs_idx < 0:
                abs_idx += len(transformer_layers)
            if 0 <= abs_idx < len(transformer_layers):
                normalized_layers.append(abs_idx)
        
        normalized_layers = list(set(normalized_layers))

        for i in normalized_layers:
            layer = transformer_layers[i]
            
            def make_hook(idx):
                def hook(module, inputs, output):
                    activations_dict = self._get_thread_activations()
                    if idx not in activations_dict:
                        activations_dict[idx] = []
                        
                    # output is usually (hidden_states, ...) or just hidden_states
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                    
                    # Detach and move to CPU to save GPU memory
                    activations_dict[idx].append(hidden_states.detach().cpu())
                return hook

            self.hooks.append(layer.register_forward_hook(make_hook(i)))
    
    def _clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        if hasattr(self._thread_local, 'layer_activations'):
            self._thread_local.layer_activations = {}

    
    
    def _process_batch(self, batch_contents: List[Tuple[ContentType, str]]) -> List[ContentResponse]:
        """
        Processes a batch of contents to generate responses and capture activations.
        """
        batch_strings = [c[1] for c in batch_contents]
        inputs = self.tokenizer(batch_strings, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}
        
        activations_dict = self._get_thread_activations()
        
        # Serialize the forward pass to avoid GPU OOM, but allow everything else
        # to run in parallel in different threads (tokenization, CPU moves, etc.)
        with torch.no_grad():
            with self._forward_lock:
                self.llm(**inputs)
            
        batch_results = []
        curr_batch_size = len(batch_contents)
        
        for b in range(curr_batch_size):
            activations = {}
            for layer_idx, batch_activations_list in activations_dict.items():
                if batch_activations_list:
                    current_batch_tensor = batch_activations_list[-1]
                    act_tensor = current_batch_tensor[b] # [seq_len, hidden_dim]
                    activations[layer_idx] = act_tensor
            
            # Filter tokens for this specific prompt (remove padding)
            input_len = inputs["attention_mask"][b].sum().item()
            
            if self.tokenizer.padding_side == "left":
                input_tokens = inputs["input_ids"][b][-input_len:].cpu()
                for k in activations:
                    activations[k] = activations[k][-input_len:]
            else:
                input_tokens = inputs["input_ids"][b][:input_len].cpu()
                for k in activations:
                    activations[k] = activations[k][:input_len]

            batch_results.append(ContentResponse(
                content=batch_contents[b][0],
                activations=activations,
                input_tokens=input_tokens
            ))
            
        # Clean up batch artifacts from hooks lists for this thread
        for k in activations_dict:
            activations_dict[k].clear()
            
        return batch_results


    def _format_content(self, content: ContentType) -> str:
        """
        Formats a single content item into a standardized string format.
        """
        if isinstance(content, str):
            return content
            
        elif isinstance(content, dict):
            return self.tokenizer.apply_chat_template([content], tokenize=False, add_generation_prompt=True)
            
        elif isinstance(content, list):
            if all(isinstance(x, dict) for x in content):
                 return self.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
            else:
                 raise ValueError("A list content must be a sequence of message dictionaries representing a single conversation.")
                 
        else:
             raise ValueError(f"Unsupported content type: {type(content)}")

    # -----------------------------------------------------------------------
    # Analysis & Checkpointing Helpers
    # -----------------------------------------------------------------------
    
    def _get_checkpoint_path(self, checkpoint_dir: str, label: str) -> str:
        """
        Gets the checkpoint path for a given label.
        """
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        safe_label = "".join([c for c in label if c.isalnum() or c in (' ', '-', '_')]).rstrip()
        return os.path.join(checkpoint_dir, f"{safe_label}_checkpoint.pt")

    def _save_checkpoint(self, data: Tuple[Dict[int, torch.Tensor], int], checkpoint_dir: str, label: str):
        """Saves a tuple of (activations, next_batch_index)"""
        path = self._get_checkpoint_path(checkpoint_dir, label)
        torch.save(data, path)

    def _load_checkpoint(self, checkpoint_dir: str, label: str) -> Optional[Tuple[Dict[int, torch.Tensor], int]]:
        path = self._get_checkpoint_path(checkpoint_dir, label)
        if os.path.exists(path):
            return torch.load(path, weights_only=False)
        return None

    def _analyse_sequential(self, 
                            batches: List[List[Tuple[ContentType, str]]], 
                            aggregator: Callable[[Dict[int, torch.Tensor], Dict[int, torch.Tensor]], Dict[int, torch.Tensor]],  
                            last_activations: Optional[Dict[int, torch.Tensor]], 
                            label: str, 
                            checkpoint_dir: Optional[str], 
                            save_every: Optional[int], 
                            start_batch_idx: int) -> Dict[int, torch.Tensor]:
        for idx, batch in enumerate(tqdm(batches, desc=f"Analysing {f'({label})' if label else '' } [[Seq]")):
            batch_results = self._process_batch(batch)
            for res in batch_results:
                if last_activations is not None:
                    last_activations = aggregator(last_activations, res.activations)
                else:
                    last_activations = res.activations
            del batch_results
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            current_batch_idx = start_batch_idx + idx + 1
            if checkpoint_dir and save_every is not None and save_every > 0 and current_batch_idx % save_every == 0:
                self._save_checkpoint((last_activations, current_batch_idx), checkpoint_dir, label)
                
        if checkpoint_dir and len(batches) > 0:
            self._save_checkpoint((last_activations, start_batch_idx + len(batches)), checkpoint_dir, label)
            
        return last_activations

    def _analyse_parallel(self, 
                          batches: List[List[Tuple[ContentType, str]]], 
                          aggregator: Callable[[Dict[int, torch.Tensor], Dict[int, torch.Tensor]], Dict[int, torch.Tensor]],  
                          last_activations: Optional[Dict[int, torch.Tensor]], 
                          label: str, 
                          max_workers: Optional[int], 
                          checkpoint_dir: Optional[str], 
                          save_every: Optional[int], 
                          start_batch_idx: int) -> Dict[int, torch.Tensor]:
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # executor.map yields results exactly in the order they were submitted
            results = executor.map(self._process_batch, batches)
            
            for idx, batch_results in enumerate(tqdm(results, total=len(batches), desc=f"Analysing {f'({label})' if label else '' } [Par]")):
                for res in batch_results:
                    if last_activations is not None:
                        last_activations = aggregator(last_activations, res.activations)
                    else:
                        last_activations = res.activations
                del batch_results
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                current_batch_idx = start_batch_idx + idx + 1
                if checkpoint_dir and save_every is not None and save_every > 0 and current_batch_idx % save_every == 0:
                     self._save_checkpoint((last_activations, current_batch_idx), checkpoint_dir, label)
                     
        if checkpoint_dir and len(batches) > 0:
            self._save_checkpoint((last_activations, start_batch_idx + len(batches)), checkpoint_dir, label)
            
        return last_activations


    # def generation_analysis()

    def analyse(self,
                dataset: Iterable[ContentType],
                aggregator: Optional[Union[Aggregator, Callable[[Dict[int, torch.Tensor], Dict[int, torch.Tensor]], Dict[int, torch.Tensor]]]] = None,
                label: Optional[str]= None,
                layers: Optional[List[int]] = None,
                metadata: Optional[Dict[str, Any]] = None,
                parallel: bool = False,
                max_workers: Optional[int] = None,
                checkpoint_dir: Optional[str] = None,
                save_checkpoint_every: Optional[int] = None,
                resume_from_checkpoint: bool = False
                )->AnalysisResult:
        """
        Analyses the model with the given prompts and returns the activations.
        For each prompt, it returns the activations of the model.
        If an aggregator is provided, it will combine the activations of the prompts into single output.
        
        Args:
            dataset: The dataset of contents to analyse. Iterate over each item.
            aggregator: The aggregator to use for combining activations.
            label: The label for the analysis.
            layers: The layers to analyse.
            metadata: The metadata for the analysis.
            parallel: Whether to use parallel processing.
            max_workers: The number of workers to use for parallel processing.
            checkpoint_dir: The directory to save checkpoints to.
            save_checkpoint_every: The number of batches to save checkpoints every, or None to not save periodic checkpoints.
            resume_from_checkpoint: Whether to resume from a checkpoint.
        
        Returns:
            AnalysisResult: The activations of the model.
        """
        final_contents = [(c, self._format_content(c)) for c in dataset]
        label = label if label else f"analysis_{id(final_contents)}"
        last_activations = None
        start_batch_idx = 0
        
        if resume_from_checkpoint and checkpoint_dir:
            ckpt = self._load_checkpoint(checkpoint_dir, label)
            if ckpt is not None:
                last_activations, start_batch_idx = ckpt

        self._register_layer_hooks(layers_to_hook=layers)
        batch_size = self.config.batch_size
        
        try:
            if len(final_contents) > 1 and aggregator is None:
                raise ValueError("Aggregator is required for multiple contents in analyse")

            batches = [final_contents[i : i + batch_size] for i in range(start_batch_idx * batch_size, len(final_contents), batch_size)]
            
            if batches:
                if parallel:
                    last_activations = self._analyse_parallel(
                        batches=batches, aggregator=aggregator, last_activations=last_activations,
                        label=label, max_workers=max_workers, checkpoint_dir=checkpoint_dir,
                        save_every=save_checkpoint_every, start_batch_idx=start_batch_idx
                    )
                else:
                    last_activations = self._analyse_sequential(
                        batches=batches, aggregator=aggregator, last_activations=last_activations,
                        label=label, checkpoint_dir=checkpoint_dir, save_every=save_checkpoint_every,
                        start_batch_idx=start_batch_idx
                    )

        finally:
            self._clear_hooks()

        return AnalysisResult(
            label=label,
            metadata=metadata if metadata else {},
            activations=last_activations if last_activations is not None else {})

    def stimulate(self, 
                 content: ContentType, 
                 layers: Optional[List[int]] = None) -> StimulationResult:
        """
        Stimulates the model with a single content.
        Supports raw string, chat dictionary, or list of chat dictionaries.
        """
        formatted_str = self._format_content(content)

        self._register_layer_hooks(layers_to_hook=layers)
        
        results = []
        try:
            batch_results = self._process_batch([(content, formatted_str)])
            results.extend(batch_results)
        finally:
            self._clear_hooks()
            
        return StimulationResult(
            results=results,
            metadata={
                "model_name": self.llm.config.name_or_path,
                "config": {k: str(v) for k, v in self.config.__dict__.items()}
            }
        )

    # -----------------------------------------------------------------------
    # Steering / intervention
    # -----------------------------------------------------------------------

    def generate(self, content: Union[ContentType, torch.Tensor] = None, *args, **kwargs):
        """
        Convenience method to call generate on the underlying language model.
        Accepts a single content (raw string, chat map, or conversation array), or pre-tokenized inputs.
        """
        if content is None:
            return self.llm.generate(*args, **kwargs)
            
        if isinstance(content, torch.Tensor):
            return self.llm.generate(content, *args, **kwargs)

        formatted_str = self._format_content(content)
        
        inputs = self.tokenizer(formatted_str, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}
        
        kwargs.update(inputs)
        output_ids = self.llm.generate(*args, **kwargs)
        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        if isinstance(content, (str, dict)):
            return decoded[0]
        if isinstance(content, list) and all(isinstance(x, dict) for x in content):
            return decoded[0]
            
        return decoded

    def ambale(
        self,
        activations: Dict[int, torch.Tensor],
        applied_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> AmbaleContext:
        """
        Steer the model with activations. 
        Returns an ``AmbaleContext`` which can lazily register activation-steering hooks.

        Args:
            activations:
                A ``{layer_num: tensor}`` dict — typically the ``.activations``
                attribute of an ``AnalysisResult``.

            applied_function:
                A callable ``(stored, live) -> tensor`` that controls how the
                stored activation is applied to the live forward-pass tensor.

        Returns:
            AmbaleContext: A context object that can be used directly to `.generate()` 
            or within a ``with`` block.
        """
        return AmbaleContext(self, activations, applied_function)

    def load_ambale(self, path: Union[str, os.PathLike[str]], applied_function: Optional[Callable] = None) -> AmbaleContext:
        """
        Loads a saved ambale (steered model) configuration from disk and returns an AmbaleContext.
        Optionally accepts a custom `applied_function` to override or provide the missing one.
        """
        from safetensors import safe_open
        from . import ambalefiers
        import json
        import warnings
        
        activations = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            for key in f.keys():
                if key.startswith("layer_"):
                    layer_idx = int(key.replace("layer_", ""))
                    activations[layer_idx] = f.get_tensor(key)
        
        saved_model_name = metadata.get("model_name_or_path", "unknown")
        current_model_name = getattr(self.llm.config, "_name_or_path", getattr(self.llm.config, "name_or_path", "unknown"))
        
        if saved_model_name != "unknown" and current_model_name != "unknown" and saved_model_name != current_model_name:
            warnings.warn(f"Steering was saved for model '{saved_model_name}' but current model is '{current_model_name}'.")

        if applied_function is None:
            factory_name = metadata.get("applied_function_factory", "custom")
            if factory_name == "custom":
                raise ValueError("This steering was saved with a custom function. You must provide `applied_function` to `load_ambale(..., applied_function=...)`.")
                
            kwargs = json.loads(metadata.get("applied_function_kwargs", "{}"))
            
            if not hasattr(ambalefiers, factory_name):
                raise ValueError(f"Could not find ambalefier factory: {factory_name}")
                
            factory = getattr(ambalefiers, factory_name)
            applied_function = factory(**kwargs)
            
        return self.ambale(activations=activations, applied_function=applied_function)

