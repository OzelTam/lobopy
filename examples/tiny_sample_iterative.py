# tiny_sample_iterative.py
from lobopy.patient import Patient, PatientConfig
from lobopy.aggregators import mean_aggregator
from lobopy.iterative_analysis import IterativeAnalysis
from lobopy.ambalefiers import safe_scale_activation, top_k_layers

"""
GOAL OF THIS SCRIPT:
IterativeAnalysis is designed to pinpoint the exact neural state of the model when it spontaneously generates a very specific target concept.

In this example, our goal is to isolate the concept of "Paris".
We are going to feed the model 30+ prompts that naturally steer it toward generating the word "Paris".
The IterativeAnalysis engine will let the model generate tokens one by one. The moment it generates "Paris", 
it captures the intermediate layer activations at that exact microsecond, aggregates them all together, 
and gives us the pure, isolated "Paris" concept tensor. 
"""

VERBOSE = True

print("Initializing TinyLlama Patient...")
model = Patient(
    pretrained_model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    config=PatientConfig(
        batch_size=1,
        trust_remote_code=True,
        torch_dtype="auto",
    ),
)

# A dataset of prompts designed to coax the model into generating "Paris"
prompts = [
    "The capital of France is",
    "The Eiffel Tower is located in the magnificent city of",
    "The Louvre Museum can be found in",
    "Notre Dame Cathedral is a famous landmark in",
    "The city of light is famously known as",
    "Montmartre is a beautiful historic district in",
    "The Champs-Élysées is a famous avenue in",
    "The Arc de Triomphe stands proudly in",
    "The Seine river flows directly through",
    "A city famous for love and croissants is",
    "Disneyland in Europe is located just outside",
    "The Palace of Versailles is situated very close to",
    "When I want to see the Mona Lisa, I travel to",
    "If you want to see the Pantheon in France, you go to",
    "The largest city in France is",
    "The French Open tennis tournament takes place in",
    "Fashion Week in France always happens in",
    "The classic movie 'Amélie' takes place in",
    "To visit the Catacombs in France, you must go to",
    "The famous cabaret Moulin Rouge is located in",
    "To see the bridge Pont Neuf, you must visit",
    "The Pompidou Center is an iconic modern art museum in",
    "Place de la Concorde is the largest square in",
    "To eat at a classic French bistro overlooking the Seine, go to",
    "The capital city where the French Revolution began is",
    "The famous bookstore Shakespeare and Company is in",
    "If someone says they are going to the 'City of Love', they mean",
    "The central hub of the French railway network is",
    "Macarons from Ladurée are most famous in",
    "The Sorbonne, one of the oldest universities, is in",
    "To walk through the beautiful Luxembourg Gardens, you travel to",
]

# We want to capture activations precisely when the model generates "Paris"
def match_fn(text: str) -> bool:
    return "paris" in text.lower()

# We abort generating further tokens if the model outputs a period or goes totally off-topic
def mismatch_fn(text: str) -> bool:
    return "." in text or "\n" in text

print("\nStarting Iterative Analysis to isolate 'Paris'...")

iterative_engine = IterativeAnalysis(
    patient=model, 
    dataset=prompts, 
    match_function=match_fn, 
    mismatch_function=mismatch_fn
)

# Run the generation loops! The engine will generate tokens, and the second `match_fn` spots "Paris", 
# it captures the activations and throws them into the `mean_aggregator()`.
aggregated_result = iterative_engine.aggregated_analysis(
    aggregator=mean_aggregator(),
    max_token_per_prompt=15,       # Maximum tokens before giving up if "Paris" is not found
    match_token_iteration=1,       # Check for "Paris" after every single token
    layers=None,                   # Hook all layers
    show_progress=True,
    is_parallel=True,
    max_workers=3,
    populate_generation_results=VERBOSE
)
aggregated_paris_activations = aggregated_result.activations

if VERBOSE:
    for i, result in enumerate(aggregated_result.iterative_generation_results):
        print(f"Generation {i+1}: {result.input_text!r} -> {result.output_text!r} (matched={result.is_matched})")


print(f"\nSuccessfully extracted target-matched 'Paris' activations across {len(aggregated_paris_activations)} layers!")
print(f"Captured {len(aggregated_result.iterative_generation_results)} generation tracks.")
if aggregated_result.iterative_generation_results:
    print(f"Sample Tracker: {aggregated_result.iterative_generation_results[0].output_text!r} (matched={aggregated_result.iterative_generation_results[0].is_matched})")

print("\nPreparing an ambalefier to boost the 'Paris' concept globally...")
from lobopy.ambalefiers import normalize_path
normed_paris_activations = normalize_path(aggregated_paris_activations)
top_layers = top_k_layers(normed_paris_activations, k=3, layer_range=(0.3, 0.7))
my_ambalefier = safe_scale_activation(factor=3.0)

# Steer the model! Even when asked about unrelated geography, the ambalefier pushes the neural network toward Paris concepts.
print("\nSteered Generation:")
with model.ambale(top_layers, my_ambalefier):
    print(model.generate("Tell me about a great city to travel to in Europe:", max_new_tokens=20, do_sample=False))
