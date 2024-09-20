import spacy
from spacy.training import Example
import random

# Load the base model
nlp = spacy.load("en_core_web_sm")

# Get the NER pipeline component
ner = nlp.get_pipe("ner")

# Add your custom labels
ner.add_label("SPICE_LEVEL")
ner.add_label("DIET_TYPE")
ner.add_label("MEAL_TYPE")

# Training data (after adjusting for correct tokenization)
TRAIN_DATA = [
    ("I want a spicy vegetarian meal for lunch.", {"entities": [(9, 14, "SPICE_LEVEL"), (15, 25, "DIET_TYPE"), (35, 40, "MEAL_TYPE")]}),
    ("Can I have a mild gluten-free dinner?", {"entities": [(13, 17, "SPICE_LEVEL"), (18, 29, "DIET_TYPE"), (30, 36, "MEAL_TYPE")]}),
    ("I would like a medium low-carb dish for lunch.", {"entities": [(16, 22, "SPICE_LEVEL"), (23, 32, "DIET_TYPE"), (43, 48, "MEAL_TYPE")]}),
    ("Please suggest a hot vegan meal for dinner.", {"entities": [(19, 22, "SPICE_LEVEL"), (23, 28, "DIET_TYPE"), (33, 37, "MEAL_TYPE")]}),
    ("Can you recommend a non-spicy keto meal for breakfast?", {"entities": [(21, 30, "SPICE_LEVEL"), (31, 35, "DIET_TYPE"), (40, 49, "MEAL_TYPE")]}),
    ("I prefer a mildly spicy pescatarian dinner.", {"entities": [(11, 17, "SPICE_LEVEL"), (18, 29, "DIET_TYPE"), (30, 36, "MEAL_TYPE")]}),
    ("Is there a high-protein non-spicy option for lunch?", {"entities": [(12, 23, "DIET_TYPE"), (24, 33, "SPICE_LEVEL"), (44, 49, "MEAL_TYPE")]}),
    ("I'd like a low-fat vegetarian lunch, please.", {"entities": [(10, 16, "DIET_TYPE"), (17, 27, "DIET_TYPE"), (28, 33, "MEAL_TYPE")]}),
    ("Give me a low-calorie non-spicy breakfast suggestion.", {"entities": [(11, 22, "DIET_TYPE"), (23, 32, "SPICE_LEVEL"), (33, 42, "MEAL_TYPE")]}),
    ("I am looking for a hot gluten-free meal for dinner.", {"entities": [(21, 24, "SPICE_LEVEL"), (25, 36, "DIET_TYPE"), (41, 47, "MEAL_TYPE")]}),
    ("Do you have a keto non-spicy meal for lunch?", {"entities": [(13, 17, "DIET_TYPE"), (18, 27, "SPICE_LEVEL"), (32, 37, "MEAL_TYPE")]}),
    ("I feel like having a spicy paleo dinner.", {"entities": [(21, 26, "SPICE_LEVEL"), (27, 32, "DIET_TYPE"), (33, 39, "MEAL_TYPE")]}),
    ("I would like a mild pescatarian dish for breakfast.", {"entities": [(16, 20, "SPICE_LEVEL"), (21, 32, "DIET_TYPE"), (43, 52, "MEAL_TYPE")]}),
    ("Please suggest a gluten-free medium-spicy dish for dinner.", {"entities": [(16, 27, "DIET_TYPE"), (28, 40, "SPICE_LEVEL"), (51, 57, "MEAL_TYPE")]}),
    ("What is the best low-fat non-spicy option for lunch?", {"entities": [(19, 25, "DIET_TYPE"), (26, 35, "SPICE_LEVEL"), (46, 51, "MEAL_TYPE")]}),
    ("Can I get a hot vegetarian dish for breakfast?", {"entities": [(13, 16, "SPICE_LEVEL"), (17, 27, "DIET_TYPE"), (38, 47, "MEAL_TYPE")]}),
    ("I need a low-calorie vegan meal for dinner.", {"entities": [(10, 21, "DIET_TYPE"), (22, 27, "DIET_TYPE"), (32, 37, "MEAL_TYPE")]}),
    ("Do you have a high-protein non-spicy breakfast option?", {"entities": [(12, 23, "DIET_TYPE"), (24, 33, "SPICE_LEVEL"), (34, 43, "MEAL_TYPE")]}),
    ("Could you recommend a low-carb mild dish for lunch?", {"entities": [(21, 29, "DIET_TYPE"), (30, 34, "SPICE_LEVEL"), (45, 50, "MEAL_TYPE")]}),
    ("Give me a keto spicy meal for dinner.", {"entities": [(10, 14, "DIET_TYPE"), (15, 20, "SPICE_LEVEL"), (25, 30, "MEAL_TYPE")]}),
    ("Can I order a vegetarian medium-spicy dish for dinner?", {"entities": [(14, 24, "DIET_TYPE"), (25, 37, "SPICE_LEVEL"), (48, 54, "MEAL_TYPE")]}),
]

# Convert training data into Example objects
train_examples = [Example.from_dict(nlp.make_doc(text), annot) for text, annot in TRAIN_DATA]

# Disable other pipeline components (parser, tagger) during training to speed up and avoid affecting them
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()

    # Training loop
    for epoch in range(10):
        random.shuffle(train_examples)
        losses = {}
        for batch in spacy.util.minibatch(train_examples, size=2):
            nlp.update(
                batch, 
                sgd=optimizer,  # Optimizer
                drop=0.35,      # Dropout for regularization
                losses=losses
            )
        print(f"Epoch {epoch}, Losses: {losses}")

# Save the trained model
nlp.to_disk("restaurant_ner_model")
