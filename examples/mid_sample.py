import sys, logging, torch
from datasets import load_dataset

from lobopy.patient import Patient, PatientConfig
from lobopy.aggregators import mean_aggregator, difference_aggregator
from lobopy.ambalefiers import (
    safe_scale_activation,
    top_k_layers,
    normalize_path,
)

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("sample")
logger.setLevel(logging.INFO)

# File handler for log.txt
file_handler = logging.FileHandler("log.txt", mode="a+", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(message)s"))

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def reformat_texts(texts):
    """Wrap texts into a single-turn user chat format."""
    return [[{"role": "user", "content": text}] for text in texts]

def load_datasets():
    """Load and format the harmful and harmless instruction datasets."""
    logger.info("Loading instruction datasets...")
    harmful_dataset = load_dataset('mlabonne/harmful_behaviors')
    harmless_dataset = load_dataset('mlabonne/harmless_alpaca')
    
    harmful_train = reformat_texts(harmful_dataset['train']['text'])
    harmful_test = reformat_texts(harmful_dataset['test']['text'])
    
    harmless_train = reformat_texts(harmless_dataset['train']['text'])
    harmless_test = reformat_texts(harmless_dataset['test']['text'])
    
    return harmful_train, harmful_test, harmless_train, harmless_test


# ---------------------------------------------------------------------------
# Model Setup
# ---------------------------------------------------------------------------

def setup_patient(model_name: str = "Nanbeige/Nanbeige4.1-3B") -> Patient:
    """Initialize and return the Patient LLM wrapper."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        logger.info(f"Attempting to load {model_name} from local cache...")
        model = Patient(
            pretrained_model_name_or_path=model_name,
            config=PatientConfig(
                batch_size=1,
                device=device,
                trust_remote_code=True,
                torch_dtype="auto",
                llm_kwargs={"local_files_only": True},
                tokenizer_kwargs={"local_files_only": True},
            ),
        )
        logger.info(f"Successfully loaded {model_name} from local files.")
    except Exception as e:
        logger.info(f"Local files not found or incomplete. Downloading and saving {model_name}... ({e})")
        model = Patient(
            pretrained_model_name_or_path=model_name,
            config=PatientConfig(
                batch_size=1,
                device=device,
                trust_remote_code=True,
                torch_dtype="auto",
                llm_kwargs={"local_files_only": False},
                tokenizer_kwargs={"local_files_only": False},
            ),
        )
        logger.info(f"Download complete and {model_name} loaded.")
        
    num_layers = len(model._find_transformer_layers())
    logger.info(f"Model: {model_name}  |  transformer layers: {num_layers}")
    
    return model


# ---------------------------------------------------------------------------
# Steering Vector Computation
# ---------------------------------------------------------------------------

def compute_refusal_path(model: Patient, harmful_train, harmless_train):
    """Compute the contrastive refusal vector between harmful and harmless queries."""
    logger.info("\nComputing harmful and harmless activations...")
    harmless_analysis = model.analyse(prompts=harmless_train, label="harmless", aggregator=mean_aggregator())
    harmful_analysis  = model.analyse(prompts=harmful_train,  label="harmful",  aggregator=mean_aggregator())
    
    logger.info("Aggregating difference to find refusal vector...")
    refusal = difference_aggregator()(harmful_analysis.activations, harmless_analysis.activations)
    
    # Normalize path and select top K layers
    refusal_normal = normalize_path(refusal)
    top_refusal = top_k_layers(refusal_normal, k=10, metric="mean_abs", layer_range=(0, 1))
    
    logger.info(f"Steering non-refusal → layers {sorted(top_refusal.keys())}")
    return top_refusal


# ---------------------------------------------------------------------------
# Generation & Intervention
# ---------------------------------------------------------------------------

def render_chat(model: Patient, prompt) -> str:
    """Apply the tokenizer's chat template and return the rendered string."""
    msgs = prompt if isinstance(prompt, list) else [prompt]
    return model.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )

def generate_with_steering(model: Patient, prompt, factor: float = 0.0, path=None, max_new_tokens: int = 1300) -> str:
    """Generate tokens with optional steering based on a path and a magnitude factor."""
    raw_text = render_chat(model, prompt) if isinstance(prompt, (dict, list)) else prompt
    
    inputs = model.tokenizer(raw_text, return_tensors="pt").to(model.llm.device)
    prompt_len = inputs["input_ids"].shape[1]

    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.1,
        do_sample=True,
        temperature=0.7,
        pad_token_id=model.tokenizer.eos_token_id
    )

    # If factor is 0 or no path is provided, fall back to baseline generation
    if path is None or factor == 0.0:
        out = model.generate(**inputs, **generation_kwargs)
    else:
        # Create an intervention context and run generation inside it
        steered_model = model.ambale(path, safe_scale_activation(factor=factor, clamp_sigma=3.0))
        out = steered_model.generate(**inputs, **generation_kwargs)

    # Use token-aware stripping to clean up output
    new_tokens = out[0][prompt_len:]
    return model.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main():
    logger.info("Starting refactored sample.py")
    
    # 1. Prepare data
    harmful_train, harmful_test, harmless_train, harmless_test = load_datasets()
    
    # 2. Setup model
    model = setup_patient("Nanbeige/Nanbeige4.1-3B")
    
    # 3. Compute the steering vector by contrasting datasets
    top_refusal = compute_refusal_path(model, harmful_train, harmless_train)
    
    # 4. Test Generation
    test_prompts = {
        f"test{i+1}":test for i, test in enumerate(harmful_test[:3]) # using the first 3 harmful instructions from the test set
    }
    
    HEADER_WIDTH = 100
    # Original factors requested for demonstration
    factors = [-2.0, -2.5, -3.0, -3.0]
    
    for prompt_key, prompt_val in test_prompts.items():
        rendered = render_chat(model, prompt_val) if isinstance(prompt_val, (dict, list)) else prompt_val
        logger.info(f"\n{'='*HEADER_WIDTH}")
        logger.info(f"PROMPT [{prompt_key}]:\n{rendered}")
        logger.info(f"{'-'*HEADER_WIDTH}")

        # Baseline measurement (no steering)
        base = generate_with_steering(model, prompt_val)
        logger.info(f"\n[baseline]\n{base}")

        # Perturbed measurements
        for f in factors:
            logger.info(f"\n{'-'*20} factor={f} {'-'*20}")
            out = generate_with_steering(model, prompt_val, factor=f, path=top_refusal)
            logger.info(f"[dump refusal] {out}")

    logger.info(f"\n{'='*HEADER_WIDTH}")
    logger.info("Done.")

if __name__ == "__main__":
    main()