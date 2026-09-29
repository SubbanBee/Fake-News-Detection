from nltk.sentiment import SentimentIntensityAnalyzer


def analyze_sentiment(text):
    """
    Analyze the sentiment of a news article using NLTK VADER.
    """

    analyzer = SentimentIntensityAnalyzer()

    scores = analyzer.polarity_scores(text)

    return scores


if __name__ == "__main__":

    sample_text = """
    Scientists have made an amazing breakthrough in healthcare.
    This incredible discovery could help millions of people.
    """

    result = analyze_sentiment(sample_text)

    print("\n========== SENTIMENT ANALYSIS ==========")

    print(f"Positive Score : {result['pos']:.3f}")
    print(f"Negative Score : {result['neg']:.3f}")
    print(f"Neutral Score  : {result['neu']:.3f}")
    print(f"Compound Score : {result['compound']:.3f}")

    if result["compound"] >= 0.05:
        sentiment = "Positive"
    elif result["compound"] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    print(f"\nOverall Sentiment: {sentiment}")

    print("\n========================================")
