import re
import nltk
from collections import Counter

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer


# ============================================================
# DOWNLOAD REQUIRED NLTK RESOURCES
# ============================================================

def setup_nltk():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("chunkers/maxent_ne_chunker", "maxent_ne_chunker"),
        ("chunkers/maxent_ne_chunker_tab", "maxent_ne_chunker_tab"),
        ("corpora/words", "words"),
        ("sentiment/vader_lexicon", "vader_lexicon"),
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


setup_nltk()


# ============================================================
# VADER SENTIMENT ANALYZER
# ============================================================

sia = SentimentIntensityAnalyzer()


# ============================================================
# SENSATIONAL / CLICKBAIT WORDS
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
# FEATURE EXTRACTION
# ============================================================

def extract_features(text):

    if not text or not text.strip():
        return {
            "total_words": 0,
            "unique_words": 0,
            "lexical_diversity": 0,
            "positive_score": 0,
            "negative_score": 0,
            "neutral_score": 0,
            "compound_score": 0,
            "sentiment": "Neutral",
            "exclamation_count": 0,
            "question_count": 0,
            "uppercase_word_count": 0,
            "sensational_words": [],
            "clickbait_phrases": [],
            "repeated_words": [],
            "noun_count": 0,
            "verb_count": 0,
            "adjective_count": 0,
            "named_entities": []
        }

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    tokens = word_tokenize(text)

    words = [
        token.lower()
        for token in tokens
        if token.isalpha()
    ]

    total_words = len(words)
    unique_words = len(set(words))

    lexical_diversity = (
        unique_words / total_words
        if total_words > 0
        else 0
    )

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
        1 for word in text.split()
        if word.isupper() and len(word) > 1
    )

    # --------------------------------------------------------
    # SENSATIONAL WORDS
    # --------------------------------------------------------

    sensational_found = [
        word
        for word in words
        if word in SENSATIONAL_WORDS
    ]

    # --------------------------------------------------------
    # CLICKBAIT PHRASES
    # --------------------------------------------------------

    text_lower = text.lower()

    clickbait_found = [
        phrase
        for phrase in CLICKBAIT_PHRASES
        if phrase in text_lower
    ]

    # --------------------------------------------------------
    # REPEATED WORDS
    # --------------------------------------------------------

    word_counts = Counter(words)

    repeated_words = [
        (word, count)
        for word, count in word_counts.items()
        if count > 1
    ]

    repeated_words.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # --------------------------------------------------------
    # POS ANALYSIS
    # --------------------------------------------------------

    tagged_tokens = pos_tag(tokens)

    noun_count = sum(
        1 for _, tag in tagged_tokens
        if tag.startswith("NN")
    )

    verb_count = sum(
        1 for _, tag in tagged_tokens
        if tag.startswith("VB")
    )

    adjective_count = sum(
        1 for _, tag in tagged_tokens
        if tag.startswith("JJ")
    )

    # --------------------------------------------------------
    # NAMED ENTITY RECOGNITION
    # --------------------------------------------------------

    tree = ne_chunk(tagged_tokens)

    named_entities = []

    for chunk in tree:

        if hasattr(chunk, "label"):

            entity_name = " ".join(
                word
                for word, tag in chunk.leaves()
            )

            entity_label = chunk.label()

            named_entities.append(
                (entity_name, entity_label)
            )

    # --------------------------------------------------------
    # RETURN ALL FEATURES
    # --------------------------------------------------------

    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "lexical_diversity": lexical_diversity,

        "positive_score": positive_score,
        "negative_score": negative_score,
        "neutral_score": neutral_score,
        "compound_score": compound_score,
        "sentiment": sentiment,

        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "uppercase_word_count": uppercase_word_count,

        "sensational_words": sensational_found,
        "clickbait_phrases": clickbait_found,
        "repeated_words": repeated_words,

        "noun_count": noun_count,
        "verb_count": verb_count,
        "adjective_count": adjective_count,

        "named_entities": named_entities
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    sample_text = """
    This is a shocking news article!
    You won't believe what happens next.
    """

    result = extract_features(sample_text)

    for key, value in result.items():
        print(f"{key}: {value}")