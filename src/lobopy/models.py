from transformers import AutoTokenizer, AutoModelForCausalLM
import os, torch
from typing import Literal, List, Union, Dict, Any, Optional, Iterable
import torch.nn as nn
from dataclasses import dataclass, field
from tqdm import tqdm


@dataclass
class PatientConfig:
    device: str = "auto"
    batch_size: int = 1
    max_new_tokens: int = 50
    torch_dtype: Union[torch.dtype, str] = torch.float32
    transformer_detection_threshold: int = 6
    trust_remote_code: bool = False
    llm_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    tokenizer_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    
ContentType = Union[
    str,
    Dict[str, str],
    List[Dict[str, str]]
]

@dataclass
class ContentResponse:
    content: ContentType 
    activations: Dict[int, torch.Tensor]  # layer_index -> activation tensor (seq_len, hidden_dim)
    input_tokens: torch.Tensor
    output_tokens: Optional[torch.Tensor] = None
    
    def __hash__(self):
        return hash(str(self.content))

@dataclass
class StimulationResult:
    results: List[ContentResponse]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self):
        return len(self.results)
    
    def __getitem__(self, idx):
        return self.results[idx]
    
    def __iter__(self):
        return iter(self.results)
    
@dataclass
class AnalysisResult:
    label:str
    activations:Dict[int, torch.Tensor]
    metadata: Dict[str, Any] = field(default_factory=dict)