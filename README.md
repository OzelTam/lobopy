# Lobotomize (lobopy)

![Lobopy Banner](./assets/banner.png)

**Lobopy** is a lightweight PyTorch/HuggingFace library for analyzing and steering the activations of causal language models. It provides an intuitive interface for computing and applying contrastive activation pathways during text generation.

## Aim

With Lobopy, you can analyze how a model represents concepts, sentiments, or any other abstract idea, and then use this information to steer the model towards or away from those concepts without fine-tuning.

### Nomenclature (In Context of Lobopy)

- [**Lobopy**](https://en.wikipedia.org/wiki/Lobotomy): Name of the module.
- [**Patient**](/src/lobopy/patient.py): The wrapped language model that is being analyzed and manipulated.
- [**Ambale**:](https://tureng.com/en/turkish-english/ambale) The steered model, or the act of applying steering functions to the model.
- [**Content**](/src/lobopy/models.py): The input provided to the model (`ContentType`). Lobopy handles raw strings, dictionaries, and full conversation histories.
- **Dataset**: An iterable (e.g. list) of `ContentType` objects. Functions like `analyse` take a `dataset` to process multiple concepts at once, whereas functions like `stimulate` operate on a single `Content`.

## Installation

You can install Lobopy directly from the repository:

```bash
git clone https://github.com/OzelTam/lobopy.git
cd lobopy
pip install -e .
```

_(Note: To stick to the project's ecosystem, you can also use `uv` instead of pip if you prefer!)_

## Quick Start

Here is a quick example of how you can extract a concept (like "calmness") and steer a model toward it.

```python
from lobopy.patient import Patient, PatientConfig
from lobopy.aggregators import mean_aggregator, difference_aggregator
from lobopy.ambalefiers import (
    safe_scale_activation,
    top_k_layers,
    normalize_path,
)

# 1. Initialize the Patient with your HuggingFace model
model = Patient(
    pretrained_model_name_or_path="HUGGINGFACE_OR_LOCAL_MODEL_PATH",
    config=PatientConfig(batch_size=1, device="cuda"),
)

# 2. Define the concept datasets you want to contrast.
# These can be strings, chat dictionaries, or lists of dictionaries.
calm_contents = ["I feel peaceful and relaxed.", "Taking a deep breath by the ocean.", "Quiet and serene."]
anxious_contents = ["I feel incredibly anxious.", "My heart is racing and I'm stressed.", "Everything is overwhelming."]

# 3. Analyze the concepts to extract activations
calm_reaction = model.analyse(
    dataset=calm_contents,
    aggregator=mean_aggregator(),
    label="calm",
    parallel=True,
    max_workers=3,
    save_checkpoint_every=2,
    checkpoint_dir="checkpoint"
)

# Or simply without parallel processing
anxious_reaction = model.analyse(
    dataset=anxious_contents,
    aggregator=mean_aggregator(),
    label="anxious"
)

# 4. Find the neutral middle-ground of the conflicting sentiments
mean_reaction = mean_aggregator()(calm_reaction.activations, anxious_reaction.activations)

# 5. Isolate the "calm" semantic pathway
calm_path = difference_aggregator()(calm_reaction.activations, mean_reaction)

# 6. Normalize the pathway and select the most impactful layers
calm_path = normalize_path(calm_path)
# k=3 selects the top 3 layers. layer_range limits the search to the middle layers
# (15% to 75% depth) to avoid lobotomizing core syntax or final output layers.
calm_path = top_k_layers(calm_path, k=3, layer_range=(0.15, 0.75))

# 7. Steer the model! Create a new context with the steered activations applied.
calm_model = model.ambale(calm_path, safe_scale_activation(factor=3.0))

# Generate text using the newly steered model
output = calm_model.generate("How are you feeling today?", max_new_tokens=50)
print(output)

# 8. Save and Load steered models for future use
calm_model.save("calm_model.lobo")
loaded_calm_model = model.load_ambale("calm_model.lobo")
```

### Data Structure Clarification

When creating a `dataset` of conversations, be mindful of list nesting to avoid ambiguity between a "dataset of single messages" and a "single conversation."

If each item in your dataset is an independent conversation, wrap each sequence of dictionaries in an outer list:

```python
dataset = [
    [{"role": "user", "content": "Hello!"}],
    [{"role": "assistant", "content": "Hi!"}]
]
```

If your dataset contains multiple conversations with multiple turns, it looks like this:

```python
dataset = [
    [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "How are you?"}],
    [{"role": "assistant", "content": "Hi!"}, {"role": "user", "content": "hello"}]
]
```

If the model does not support templated chat, you can use raw strings to define a dataset:

```python
dataset = [
    "Hello!",
    "How are you?",
    "Hi!",
    "hello"
]
```

## Examples

We provide ready-to-use examples in the `examples/` directory:

- [Tiny Sample](./examples/tiny_sample.py): Uses **TinyLlama-1.1B-Chat-v1.0** for a quick sentiment steering test.
- [Iterative Sample](./examples/tiny_sample_iterative.py): Demonstrates how to use Iterative Analysis on **TinyLlama-1.1B-Chat-v1.0** to steer generation step-by-step.
- [Mid Sample](./examples/mid_sample.py): Uses **Nanbeige4.1-3B** to build a contrastive refusal vector from harmful and harmless instruction datasets.

## Sources & Inspiration

Here are some of the main sources that inspired the creation of this module:

- [Maxime Labonne - Uncensor any LLM with abliteration](https://huggingface.co/blog/mlabonne/abliteration)
- [Huggingface (YouTube) - Steering LLM Behavior Without Fine-Tuning](https://www.youtube.com/watch?v=F2jd5WuT-zg)
- [Welch Labs (YouTube) - The most complex model we actually understand](https://www.youtube.com/watch?v=D8GOeCFFby4)

## Iterative Analysis

With [`iterative_analysis.py`](/src/lobopy/iterative_analysis.py), Lobopy now supports capturing and analyzing the model's activations specifically when a certain phrase, token, or concept is generated (rather than just prompted). See the [Iterative Sample](./examples/tiny_sample_iterative.py) for a demonstration.

## License

lobopy is licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).

Copyright (C) 2026 OzelTam

See the LICENSE file for full license text.
