import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


# Initialize NLP tools
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """
    Perform basic NLP preprocessing on a news article.
    """

    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Tokenization
    tokens = word_tokenize(text)

    # 3. Remove punctuation and non-alphabetic tokens
    tokens = [
        word for word in tokens
        if word.isalpha()
    ]

    # 4. Remove stopwords
    filtered_tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # 5. Stemming
    stemmed_tokens = [
        stemmer.stem(word)
        for word in filtered_tokens
    ]

    # 6. Lemmatization
    lemmatized_tokens = [
        lemmatizer.lemmatize(word)
        for word in filtered_tokens
    ]

    return {
        "tokens": tokens,
        "without_stopwords": filtered_tokens,
        "stemmed": stemmed_tokens,
        "lemmatized": lemmatized_tokens
    }


if __name__ == "__main__":

    sample_text = """
    Scientists have discovered an amazing new technology.
    The technology could change the future of healthcare.
    """

    result = preprocess_text(sample_text)

    print("\n========== NLP TEXT PREPROCESSING ==========")

    print("\nOriginal Text:")
    print(sample_text)

    print("\nTokens:")
    print(result["tokens"])

    print("\nAfter Stopword Removal:")
    print(result["without_stopwords"])

    print("\nAfter Stemming:")
    print(result["stemmed"])

    print("\nAfter Lemmatization:")
    print(result["lemmatized"])

    print("\n============================================")
