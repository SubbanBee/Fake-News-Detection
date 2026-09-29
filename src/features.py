import re
from collections import Counter

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer


# ============================================================
# SENSATIONAL WORDS
# ============================================================

SENSATIONAL_WORDS = {
    "shocking",
    "breaking",
    "unbelievable",
    "amazing",
    "secret",
    "exposed",
    "urgent",
    "viral",
    "incredible",
    "scandal",
    "warning",
    "exclusive",
    "revealed",
    "miracle",
    "bombshell",
    "controversial"
}


# ============================================================
# CLICKBAIT PHRASES
# ============================================================

CLICKBAIT_PHRASES = {
    "you won't believe",
    "you will not believe",
    "what happens next",
    "shocking truth",
    "must read",
    "share this",
    "before it's too late",
    "you need to know",
    "this will change your life"
}


# ============================================================
# SENTIMENT ANALYZER
# ============================================================

sia = SentimentIntensityAnalyzer()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(text):

    # --------------------------------------------------------
    # Basic words
    # --------------------------------------------------------

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text.lower()
    )

    total_words = len(words)

    unique_words = len(set(words))

    if total_words > 0:
        lexical_diversity = unique_words / total_words
    else:
        lexical_diversity = 0.0


    # --------------------------------------------------------
    # SENTIMENT ANALYSIS
    # --------------------------------------------------------

    sentiment_scores = sia.polarity_scores(text)

    positive_score = sentiment_scores["pos"]
    negative_score = sentiment_scores["neg"]
    neutral_score = sentiment_scores["neu"]
    compound_score = sentiment_scores["compound"]

    if compound_score >= 0.05:
        sentiment = "Positive"

    elif compound_score <= -0.05:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"


    # --------------------------------------------------------
    # WRITING STYLE
    # --------------------------------------------------------

    exclamation_count = text.count("!")

    question_count = text.count("?")

    uppercase_word_count = sum(
        1
        for word in text.split()
        if word.isupper() and len(word) > 1
    )


    # --------------------------------------------------------
    # SENSATIONAL WORDS
    # --------------------------------------------------------

    sensational_words = []

    for word in words:

        if word in SENSATIONAL_WORDS:
            sensational_words.append(word)


    # --------------------------------------------------------
    # CLICKBAIT PHRASES
    # --------------------------------------------------------

    text_lower = text.lower()

    clickbait_phrases = []

    for phrase in CLICKBAIT_PHRASES:

        if phrase in text_lower:
            clickbait_phrases.append(phrase)


    # --------------------------------------------------------
    # REPEATED WORDS
    # --------------------------------------------------------

    word_counts = Counter(words)

    repeated_words = {
        word: count
        for word, count in word_counts.items()
        if count > 1
    }


    # --------------------------------------------------------
    # POS TAGGING
    # --------------------------------------------------------

    try:

        tokens = word_tokenize(text)

        tagged_words = pos_tag(tokens)

        noun_count = sum(
            1
            for word, tag in tagged_words
            if tag.startswith("NN")
        )

        verb_count = sum(
            1
            for word, tag in tagged_words
            if tag.startswith("VB")
        )

        adjective_count = sum(
            1
            for word, tag in tagged_words
            if tag.startswith("JJ")
        )

    except Exception:

        noun_count = 0
        verb_count = 0
        adjective_count = 0


    # --------------------------------------------------------
    # NAMED ENTITY RECOGNITION
    # --------------------------------------------------------

    named_entities = []

    try:

        tokens = word_tokenize(text)

        tagged_words = pos_tag(tokens)

        entity_tree = ne_chunk(tagged_words)

        for chunk in entity_tree:

            if hasattr(chunk, "label"):

                entity_name = " ".join(
                    word
                    for word, tag in chunk.leaves()
                )

                entity_label = chunk.label()

                named_entities.append(
                    (entity_name, entity_label)
                )

    except Exception:

        named_entities = []


    # --------------------------------------------------------
    # RETURN ALL FEATURES
    # --------------------------------------------------------

    return {

        # Basic statistics
        "total_words": total_words,
        "unique_words": unique_words,
        "lexical_diversity": lexical_diversity,

        # Sentiment
        "positive_score": positive_score,
        "negative_score": negative_score,
        "neutral_score": neutral_score,
        "compound_score": compound_score,
        "sentiment": sentiment,

        # Writing style
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "uppercase_word_count": uppercase_word_count,

        # Sensational / clickbait
        "sensational_words": sensational_words,
        "clickbait_phrases": clickbait_phrases,

        # Repeated words
        "repeated_words": repeated_words,

        # POS
        "noun_count": noun_count,
        "verb_count": verb_count,
        "adjective_count": adjective_count,

        # NER
        "named_entities": named_entities
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample_text = """
    BREAKING! Scientists have discovered an unbelievable
    secret that could change the world. You won't believe
    what happens next! Share this before it's too late.
    """

    result = extract_features(sample_text)

    print("\n========== FEATURE EXTRACTION TEST ==========\n")

    for key, value in result.items():

        print(f"{key}: {value}")