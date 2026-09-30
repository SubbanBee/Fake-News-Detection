# ============================================================
# NLP FEATURE EXTRACTION
# Fake News Detection System
# ============================================================

import re
from collections import Counter

import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer


# ============================================================
# NLTK DOWNLOADS
# ============================================================

nltk_packages = [
    "punkt",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "maxent_ne_chunker",
    "words",
    "vader_lexicon"
]

for package in nltk_packages:

    try:
        nltk.download(
            package,
            quiet=True
        )
    except Exception:
        pass


# ============================================================
# NLP OBJECTS
# ============================================================

stop_words = set(
    stopwords.words("english")
)

stemmer = PorterStemmer()

lemmatizer = WordNetLemmatizer()

sentiment_analyzer = (
    SentimentIntensityAnalyzer()
)


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
# EXTRACT FEATURES
# ============================================================

def extract_features(text):

    # ========================================================
    # BASIC CLEANING
    # ========================================================

    if not text:
        text = ""

    original_text = text

    lowercase_text = text.lower()


    # ========================================================
    # TOKENIZATION
    # ========================================================

    tokens = re.findall(
        r"\b[a-zA-Z]+\b",
        lowercase_text
    )


    # ========================================================
    # STOPWORD REMOVAL
    # ========================================================

    filtered_tokens = [
        token
        for token in tokens
        if token not in stop_words
    ]


    # ========================================================
    # STEMMING
    # ========================================================

    stemmed_tokens = [
        stemmer.stem(token)
        for token in filtered_tokens
    ]


    # ========================================================
    # LEMMATIZATION
    # ========================================================

    lemmatized_tokens = [
        lemmatizer.lemmatize(token)
        for token in filtered_tokens
    ]


    # ========================================================
    # TEXT STATISTICS
    # ========================================================

    total_words = len(tokens)

    unique_words = len(
        set(tokens)
    )


    if total_words > 0:

        lexical_diversity = (
            unique_words / total_words
        )

    else:

        lexical_diversity = 0.0


    # ========================================================
    # SENTIMENT ANALYSIS
    # ========================================================

    sentiment_scores = (
        sentiment_analyzer.polarity_scores(
            original_text
        )
    )

    positive_score = (
        sentiment_scores["pos"]
    )

    negative_score = (
        sentiment_scores["neg"]
    )

    neutral_score = (
        sentiment_scores["neu"]
    )

    compound_score = (
        sentiment_scores["compound"]
    )


    if compound_score >= 0.05:

        sentiment = "Positive"

    elif compound_score <= -0.05:

        sentiment = "Negative"

    else:

        sentiment = "Neutral"


    # ========================================================
    # WRITING STYLE
    # ========================================================

    exclamation_count = (
        original_text.count("!")
    )

    question_count = (
        original_text.count("?")
    )


    uppercase_words = re.findall(
        r"\b[A-Z]{2,}\b",
        original_text
    )

    uppercase_word_count = len(
        uppercase_words
    )


    # ========================================================
    # SENSATIONAL LANGUAGE
    # ========================================================

    sensational_words_found = []

    for word in tokens:

        if word in SENSATIONAL_WORDS:

            if word not in sensational_words_found:

                sensational_words_found.append(
                    word
                )


    # ========================================================
    # CLICKBAIT ANALYSIS
    # ========================================================

    clickbait_phrases_found = []

    for phrase in CLICKBAIT_PHRASES:

        if phrase in lowercase_text:

            clickbait_phrases_found.append(
                phrase
            )


    # ========================================================
    # POS TAGGING
    # ========================================================

    noun_count = 0

    verb_count = 0

    adjective_count = 0


    try:

        pos_tags = nltk.pos_tag(
            tokens
        )


        for word, tag in pos_tags:

            if tag.startswith("NN"):

                noun_count += 1

            elif tag.startswith("VB"):

                verb_count += 1

            elif tag.startswith("JJ"):

                adjective_count += 1

    except Exception:

        noun_count = 0
        verb_count = 0
        adjective_count = 0


    # ========================================================
    # NAMED ENTITY RECOGNITION
    # ========================================================

    named_entities = []


    try:

        pos_tags = nltk.pos_tag(
            tokens
        )

        chunks = nltk.ne_chunk(
            pos_tags
        )


        for chunk in chunks:

            if hasattr(
                chunk,
                "label"
            ):

                entity = " ".join(
                    c[0]
                    for c in chunk
                )

                label = chunk.label()

                named_entities.append(
                    (
                        entity,
                        label
                    )
                )

    except Exception:

        named_entities = []


    # ========================================================
    # REPEATED WORDS
    # IMPORTANT:
    # Stopwords such as the, and, to, of are removed.
    # ========================================================

    meaningful_tokens = [
        token
        for token in tokens
        if (
            token not in stop_words
            and len(token) > 2
        )
    ]


    word_counts = Counter(
        meaningful_tokens
    )


    repeated_words = [
        (
            word,
            count
        )

        for word, count
        in word_counts.most_common(10)

        if count > 1
    ]


    # ========================================================
    # RETURN ALL FEATURES
    # ========================================================

    return {

        # Text statistics
        "total_words":
            total_words,

        "unique_words":
            unique_words,

        "lexical_diversity":
            lexical_diversity,


        # Preprocessing
        "tokens":
            tokens,

        "without_stopwords":
            filtered_tokens,

        "stemmed":
            stemmed_tokens,

        "lemmatized":
            lemmatized_tokens,


        # Sentiment
        "sentiment":
            sentiment,

        "positive_score":
            positive_score,

        "negative_score":
            negative_score,

        "neutral_score":
            neutral_score,

        "compound_score":
            compound_score,


        # Writing style
        "exclamation_count":
            exclamation_count,

        "question_count":
            question_count,

        "uppercase_word_count":
            uppercase_word_count,


        # Sensational language
        "sensational_words":
            sensational_words_found,


        # Clickbait
        "clickbait_phrases":
            clickbait_phrases_found,


        # POS
        "noun_count":
            noun_count,

        "verb_count":
            verb_count,

        "adjective_count":
            adjective_count,


        # NER
        "named_entities":
            named_entities,


        # Repeated words
        "repeated_words":
            repeated_words
    }