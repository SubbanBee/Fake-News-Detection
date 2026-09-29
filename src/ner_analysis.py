import nltk

from nltk import word_tokenize, pos_tag, ne_chunk


# ============================================================
# NAMED ENTITY RECOGNITION
# ============================================================

def analyze_ner(text):
    """
    Perform Named Entity Recognition using NLTK.

    Returns:
        List of tuples:
        [
            ("Entity Name", "ENTITY_TYPE"),
            ...
        ]
    """

    if not text or not text.strip():
        return []

    # Tokenize
    tokens = word_tokenize(text)

    # POS tagging
    tagged_tokens = pos_tag(tokens)

    # Named Entity Recognition
    tree = ne_chunk(tagged_tokens)

    entities = []

    for chunk in tree:

        if hasattr(chunk, "label"):

            entity_name = " ".join(
                word
                for word, tag in chunk.leaves()
            )

            entity_label = chunk.label()

            entities.append(
                (entity_name, entity_label)
            )

    return entities


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample_text = """
    Narendra Modi visited Hyderabad and met
    representatives from Microsoft in India.
    """

    print("\n========== NAMED ENTITY RECOGNITION ==========\n")

    entities = analyze_ner(sample_text)

    if entities:

        for entity, label in entities:
            print(
                f"Entity: {entity} | Label: {label}"
            )

    else:

        print("No named entities detected.")