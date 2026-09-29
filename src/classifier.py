import joblib


MODEL_PATH = "outputs/fake_news_model.pkl"
VECTORIZER_PATH = "outputs/tfidf_vectorizer.pkl"


# Load trained model and TF-IDF vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def classify_news(text):
    """
    Predict whether a news article is FAKE or REAL
    using the previously trained ML model.
    """

    # Convert article into TF-IDF features
    text_tfidf = vectorizer.transform([text])

    # Prediction
    prediction = model.predict(text_tfidf)[0]

    # Prediction probability
    probabilities = model.predict_proba(text_tfidf)[0]

    fake_probability = probabilities[0]
    real_probability = probabilities[1]

    if prediction == 0:
        classification = "FAKE"
        confidence = fake_probability

    else:
        classification = "REAL"
        confidence = real_probability

    return classification, confidence, fake_probability, real_probability


if __name__ == "__main__":

    sample_text = """
    BREAKING!!! Scientists reveal a shocking secret discovery!
    You won't believe what happens next!
    """

    classification, confidence, fake_probability, real_probability = (
        classify_news(sample_text)
    )

    print("\n==============================================")
    print("          FAKE NEWS ML PREDICTION")
    print("==============================================")

    print(f"\nPrediction        : {classification}")
    print(f"Confidence        : {confidence * 100:.2f}%")
    print(f"Fake Probability  : {fake_probability * 100:.2f}%")
    print(f"Real Probability  : {real_probability * 100:.2f}%")

    print("\n==============================================")
