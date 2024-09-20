from spacy.training import offsets_to_biluo_tags
import spacy

nlp = spacy.load("en_core_web_sm")

def print_tokens_with_biluo(text, entities):
    # Create the doc object
    doc = nlp.make_doc(text)
    
    # Convert the entity offsets to BILUO tags
    biluo_tags = offsets_to_biluo_tags(doc, entities)
    
    # Print tokens and their corresponding BILUO tags
    for token, tag in zip(doc, biluo_tags):
        print(f"{token.text} | {tag}")

# Input text
text = "I want a spicy vegetarian meal for lunch."

# Entity offsets
entities = [(9, 14, "SPICE_LEVEL"), (15, 25, "DIET_TYPE"), (35, 40, "MEAL_TYPE")]

print_tokens_with_biluo(text, entities)
