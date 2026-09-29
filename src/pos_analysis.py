from nltk.tokenize import word_tokenize
from nltk import pos_tag


def analyze_pos(text):
    """
    Perform Part-of-Speech tagging on the given text.
    """

    tokens = word_tokenize(text)

    # POS tagging
    tagged_words = pos_tag(tokens)

    return tagged_words


if __name__ == "__main__":

    sample_text = """
    Scientists discovered an amazing new technology.
    The technology could change the future of healthcare.
    """

    result = analyze_pos(sample_text)

    print("\n========== POS TAGGING ==========")

    for word, tag in result:
        print(f"{word:15} -> {tag}")

    print("\n=================================")
