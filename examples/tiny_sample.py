from lobopy.patient import Patient, PatientConfig
from lobopy.aggregators import mean_aggregator, difference_aggregator
from lobopy.ambalefiers import (
    safe_scale_activation,
    top_k_layers,
    normalize_path,
)

model = Patient(
    pretrained_model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    config=PatientConfig(
        batch_size=1,
        trust_remote_code=True,
        torch_dtype="auto",
    ),
)


happy_prompts = [
    "Describe your happiest childhood memory.",
    "What always makes you smile instantly?",
    "Write about a perfect sunny day.",
    "Describe a moment when you felt truly proud.",
    "What song instantly lifts your mood?",
    "Write about a surprise party.",
    "Describe your dream vacation.",
    "What does success feel like to you?",
    "Write about laughing uncontrollably with friends.",
    "Describe the feeling of achieving a big goal.",
    "What is your favorite celebration memory?",
    "Write about receiving good news.",
    "Describe your ideal weekend.",
    "What hobby brings you the most joy?",
    "Write about reconnecting with an old friend.",
    "Describe a random act of kindness you experienced.",
    "What food makes you happiest?",
    "Write about a beautiful sunset.",
    "Describe a time you helped someone.",
    "What accomplishment means the most to you?",
    "Write about a joyful family gathering.",
    "Describe your favorite holiday.",
    "What does happiness mean to you?",
    "Write about adopting a pet.",
    "Describe your favorite place in nature.",
    "What compliment meant the most to you?",
    "Write about learning something exciting.",
    "Describe your best birthday.",
    "What makes your life meaningful?",
    "Write about a dream coming true.",
    "Describe your favorite tradition.",
    "What motivates you every day?",
    "Write about a personal breakthrough.",
    "Describe a perfect morning.",
    "What makes you feel grateful today?"
]

sad_prompts = [
    "Describe a time you felt deeply disappointed.",
    "Write about losing something important to you.",
    "Describe a moment when you felt alone.",
    "What does heartbreak feel like?",
    "Write about saying goodbye to someone.",
    "Describe a missed opportunity.",
    "Write about a time you felt misunderstood.",
    "Describe a day that did not go as planned.",
    "What does regret mean to you?",
    "Write about a difficult decision you had to make.",
    "Describe a moment of failure.",
    "Write about feeling left out.",
    "Describe a time you felt overwhelmed.",
    "What does grief feel like?",
    "Write about a broken promise.",
    "Describe a personal setback.",
    "Write about a time you felt powerless.",
    "Describe a memory that still hurts.",
    "What does loneliness mean to you?",
    "Write about a relationship that ended.",
    "Describe a moment of self-doubt.",
    "Write about feeling homesick.",
    "Describe a time you were disappointed in yourself.",
    "What does emotional exhaustion feel like?",
    "Write about a time you felt rejected.",
    "Describe a painful lesson you learned.",
    "Write about watching someone drift away.",
    "Describe a difficult apology.",
    "What does sadness look like to you?",
    "Write about a time you struggled silently.",
    "Describe a moment of vulnerability.",
    "Write about feeling uncertain about the future.",
    "Describe a time you felt forgotten.",
    "What does letting go feel like?",
    "Write about a quiet, heavy evening."
]


happy_reaction = model.analyse(dataset=happy_prompts, aggregator=mean_aggregator(), label="happy",parallel=True, max_workers=3, save_checkpoint_every=2,  checkpoint_dir="checkpoint")
sad_reaction = model.analyse(dataset=sad_prompts, aggregator=mean_aggregator(), label="sad",parallel=True, max_workers=3, save_checkpoint_every=2, checkpoint_dir="checkpoint")

# find middlegroud of happy and sad:
neutral_emotion_activations = mean_aggregator()(happy_reaction.activations, sad_reaction.activations)

# and now we can use this to isolate happy path from happyiness reaction
happy_path = difference_aggregator()(happy_reaction.activations, neutral_emotion_activations)
sad_path = difference_aggregator()(sad_reaction.activations, neutral_emotion_activations)

# Normalize paths
happy_path = normalize_path(happy_path)
sad_path = normalize_path(sad_path)

# Get top k layers with a range so we dont completely lobotomize the model
happy_path = top_k_layers(happy_path, 3, layer_range=(0.15, 0.75))
sad_path = top_k_layers(sad_path, 3, layer_range=(0.15, 0.75))

# Generate steered models
happy_model = model.ambale(activations=happy_path, applied_function=safe_scale_activation(2))
sad_model = model.ambale(activations=sad_path, applied_function=safe_scale_activation(2))

# Save steered models
happy_model.save("happy_model.lobo")
sad_model.save("sad_model.lobo")


test_prompts = [
    "Then she asked.",
    "So, ",
    "An then it was",
    "The story was"]

for prompt in test_prompts:
    print("="*100+"\n")
    print("Prompt: ", prompt)
    print("-"*10,"\n")
    print("[Baseline]: ", model.generate(prompt, max_new_tokens=50))
    print("-"*10,"\n")
    print("[Happy]: ", happy_model.generate(prompt, max_new_tokens=50))
    print("[Sad]: ", sad_model.generate(prompt, max_new_tokens=50))
    