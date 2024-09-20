import spacy
from spacy.training import offsets_to_biluo_tags

# Load the trained model
nlp = spacy.load("restaurant_ner_model")

# Process text
text = "I want a spicy vegetarian meal for lunch."
doc = nlp(text)

# # Print tokens and entity recognition process
# for token in doc:
#     print(f"Token: {token.text}, Entity: {token.ent_type_}")

# # Check if any entities were extracted
# for ent in doc.ents:
#     print(f"Entity: {ent.text}, Label: {ent.label_}")

for ent in doc.ents:
    print(ent.text, ent.label_)