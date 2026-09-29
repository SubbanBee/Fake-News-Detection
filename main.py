from src.features import extract_features
from src.classifier import classify_news


def main():

    print("\n==============================================")
    print("          FAKE NEWS DETECTION SYSTEM")
    print("             NLP + MACHINE LEARNING")
    print("==============================================")

    print("\nEnter your news article.")
    print("Type your complete article and press ENTER twice.\n")

    lines = []

    while True:
        line = input()

        if line == "":
            break

        lines.append(line)

    text = " ".join(lines)

    if not text.strip():
        print("\nNo news article entered.")
        return

    print("\nAnalyzing article using NLP and Machine Learning...")
    print("----------------------------------------------")

    # ==========================================
    # NLP FEATURE ANALYSIS
    # ==========================================

    features = extract_features(text)

    print("\n========== NLP FEATURE ANALYSIS ==========\n")

    print(f"Total Words        : {features['total_words']}")
    print(f"Unique Words       : {features['unique_words']}")
    print(f"Lexical Diversity  : {features['lexical_diversity']:.3f}")

    print("\n--- Sentiment ---")

    print(f"Positive Score     : {features['positive_score']:.3f}")
    print(f"Negative Score     : {features['negative_score']:.3f}")
    print(f"Neutral Score      : {features['neutral_score']:.3f}")
    print(f"Compound Score     : {features['compound_score']:.3f}")

    if features["compound_score"] >= 0.05:
        sentiment = "Positive"

    elif features["compound_score"] <= -0.05:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    print(f"Overall Sentiment  : {sentiment}")

    print("\n--- Writing Style ---")

    print(f"Exclamation Marks  : {features['exclamation_count']}")
    print(f"Question Marks     : {features['question_count']}")
    print(f"Capitalized Words  : {features['capitalized_count']}")

    print("\n--- Sensational Language ---")

    print(f"Count              : {features['sensational_count']}")
    print(f"Words Found        : {features['sensational_words']}")

    print("\n--- Clickbait Phrases ---")

    print(f"Count              : {features['clickbait_count']}")
    print(f"Phrases Found      : {features['clickbait_phrases']}")

    print("\n--- Repeated Words ---")

    print(features["repeated_words"])

    print("\n--- POS Features ---")

    print(f"Nouns              : {features['noun_count']}")
    print(f"Verbs              : {features['verb_count']}")
    print(f"Adjectives         : {features['adjective_count']}")

    print("\n--- Named Entities ---")

    if features["named_entities"]:

        for entity, label in features["named_entities"]:
            print(f"{entity} -> {label}")

    else:
        print("No named entities detected")


    # ==========================================
    # ML PREDICTION
    # ==========================================

    classification, confidence, fake_probability, real_probability = (
        classify_news(text)
    )


    print("\n==============================================")
    print("              ML PREDICTION")
    print("==============================================")

    print(f"\nPrediction        : {classification}")
    print(f"Confidence        : {confidence * 100:.2f}%")
    print(f"Fake Probability  : {fake_probability * 100:.2f}%")
    print(f"Real Probability  : {real_probability * 100:.2f}%")


    # ==========================================
    # FINAL RESULT
    # ==========================================

    print("\n==============================================")
    print("              FINAL RESULT")
    print("==============================================")

    if classification == "FAKE":

        print("\n⚠ Prediction: FAKE NEWS")

    else:

        print("\n✓ Prediction: REAL NEWS")

    print(f"\nModel Confidence : {confidence * 100:.2f}%")

    print("\n==============================================")
    print("This prediction is based on the trained")
    print("NLP text-classification model.")
    print("==============================================\n")


if __name__ == "__main__":
    main()
