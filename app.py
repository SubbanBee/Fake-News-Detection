import streamlit as st
import pandas as pd

from src.classifier import classify_news
from src.features import extract_features
from src.history import save_prediction, load_history


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📰 Fake News AI")

    st.divider()

    st.subheader("🔍 System Modules")

    st.markdown("""
    - 📝 News Article Input
    - 🧹 Text Preprocessing
    - 🔤 TF-IDF Feature Extraction
    - 🤖 ML Prediction
    - 😊 Sentiment Analysis
    - 🏷️ POS Analysis
    - 📍 Named Entity Recognition
    - 🚨 Clickbait Detection
    - ⚡ Sensational Language Detection
    - 📊 Prediction History
    """)

    st.divider()

    st.subheader("🤖 Model")

    st.info("TF-IDF + Logistic Regression")

    st.divider()

    st.caption(
        "Fake News Detection System\n"
        "Machine Learning + NLP"
    )


# ============================================================
# HEADER
# ============================================================

st.title("📰 Fake News Detection System")

st.write(
    "Analyze news articles using Natural Language Processing "
    "and Machine Learning to predict whether the article is "
    "likely to be FAKE or REAL."
)

st.divider()


# ============================================================
# ARTICLE INPUT
# ============================================================

st.header("📝 News Article Analysis")

article_text = st.text_area(
    "Enter News Article",
    height=230,
    placeholder="Paste the complete news article here..."
)

analyze_button = st.button(
    "🔍 Analyze News",
    type="primary",
    width="stretch"
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    if not article_text.strip():

        st.warning(
            "⚠️ Please enter a news article before analysis."
        )

    else:

        # ----------------------------------------------------
        # NLP FEATURES
        # ----------------------------------------------------

        features = extract_features(article_text)

        # ----------------------------------------------------
        # ML PREDICTION
        # ----------------------------------------------------

        (
            classification,
            confidence,
            fake_probability,
            real_probability
        ) = classify_news(article_text)

        # ----------------------------------------------------
        # SAVE PREDICTION
        # ----------------------------------------------------

        save_prediction(
            article_text,
            classification,
            confidence,
            fake_probability,
            real_probability
        )

        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.header("🎯 Prediction Result")

        if classification == "FAKE":

            st.error(
                f"🚨 FAKE NEWS\n\n"
                f"Confidence: {confidence * 100:.2f}%\n\n"
                "The trained machine learning model classified "
                "this article as FAKE based on learned textual "
                "patterns."
            )

        else:

            st.success(
                f"✓ REAL NEWS\n\n"
                f"Confidence: {confidence * 100:.2f}%\n\n"
                "The trained machine learning model classified "
                "this article as REAL based on learned textual "
                "patterns."
            )

        # ====================================================
        # PROBABILITY
        # ====================================================

        st.header("📊 Prediction Probability")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🚨 Fake Probability",
                f"{fake_probability * 100:.2f}%"
            )

        with col2:

            st.metric(
                "✓ Real Probability",
                f"{real_probability * 100:.2f}%"
            )

        probability_df = pd.DataFrame(
            {
                "Probability": [
                    fake_probability,
                    real_probability
                ]
            },
            index=["FAKE", "REAL"]
        )

        st.bar_chart(
            probability_df,
            width="stretch"
        )

        # ====================================================
        # TEXT STATISTICS
        # ====================================================

        st.header("📈 Text Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Words",
                features["total_words"]
            )

        with col2:

            st.metric(
                "Unique Words",
                features["unique_words"]
            )

        with col3:

            st.metric(
                "Lexical Diversity",
                f"{features['lexical_diversity']:.3f}"
            )

        # ====================================================
        # SENTIMENT
        # ====================================================

        st.header("😊 Sentiment Analysis")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Positive",
                f"{features['positive_score']:.3f}"
            )

        with col2:

            st.metric(
                "Negative",
                f"{features['negative_score']:.3f}"
            )

        with col3:

            st.metric(
                "Neutral",
                f"{features['neutral_score']:.3f}"
            )

        with col4:

            st.metric(
                "Compound",
                f"{features['compound_score']:.3f}"
            )

        sentiment = features["sentiment"]

        if sentiment == "Positive":

            st.success(
                f"Overall Sentiment: {sentiment}"
            )

        elif sentiment == "Negative":

            st.error(
                f"Overall Sentiment: {sentiment}"
            )

        else:

            st.info(
                f"Overall Sentiment: {sentiment}"
            )

        # ====================================================
        # WRITING STYLE
        # ====================================================

        st.header("✍️ Writing Style")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "❗ Exclamation Marks",
                features["exclamation_count"]
            )

        with col2:

            st.metric(
                "❓ Question Marks",
                features["question_count"]
            )

        with col3:

            st.metric(
                "🔠 Capitalized Words",
                features["uppercase_word_count"]
            )

        with col4:

            st.metric(
                "🔁 Repeated Words",
                len(features["repeated_words"])
            )

        # ====================================================
        # SENSATIONAL LANGUAGE
        # ====================================================

        st.header("🚨 Sensational Language")

        sensational_words = features["sensational_words"]

        if sensational_words:

            st.write(
                "Detected sensational words:"
            )

            st.write(
                ", ".join(sensational_words)
            )

        else:

            st.success(
                "No predefined sensational words detected."
            )

        # ====================================================
        # CLICKBAIT
        # ====================================================

        st.header("🎯 Clickbait Pattern Analysis")

        clickbait_phrases = features["clickbait_phrases"]

        if clickbait_phrases:

            st.write(
                "Detected clickbait phrases:"
            )

            for phrase in clickbait_phrases:

                st.warning(
                    f"• {phrase}"
                )

        else:

            st.success(
                "No predefined clickbait phrases detected."
            )

        # ====================================================
        # POS
        # ====================================================

        st.header("🏷️ Part-of-Speech Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Nouns",
                features["noun_count"]
            )

        with col2:

            st.metric(
                "Verbs",
                features["verb_count"]
            )

        with col3:

            st.metric(
                "Adjectives",
                features["adjective_count"]
            )

        # ====================================================
        # NER
        # ====================================================

        st.header("📍 Named Entity Recognition")

        entities = features["named_entities"]

        if entities:

            for entity, label in entities:

                st.write(
                    f"**{entity}** — `{label}`"
                )

        else:

            st.info(
                "No named entities detected."
            )

        # ====================================================
        # REPEATED WORDS
        # ====================================================

        st.header("🔁 Repeated Words")

        repeated_words = features["repeated_words"]

        if repeated_words:

            for word, count in sorted(
                repeated_words.items(),
                key=lambda x: x[1],
                reverse=True
            ):

                st.write(
                    f"**{word}** → {count} times"
                )

        else:

            st.info(
                "No repeated words detected."
            )

        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.header("🤖 Model Information")

        st.info(
            "This system uses TF-IDF feature extraction with "
            "a Logistic Regression classifier.\n\n"
            "The model learns textual patterns from labelled "
            "fake and real news articles.\n\n"
            "The prediction indicates how the article resembles "
            "patterns learned from the training dataset. It does "
            "not independently verify the factual truth of the article."
        )


# ============================================================
# PREDICTION HISTORY
# ============================================================

st.divider()

st.header("📚 Prediction History")

history = load_history()

if history:

    history_df = pd.DataFrame(history)

    total_predictions = len(history_df)

    fake_count = (
        history_df["Prediction"] == "FAKE"
    ).sum()

    real_count = (
        history_df["Prediction"] == "REAL"
    ).sum()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📊 Total Analyzed",
            total_predictions
        )

    with col2:

        st.metric(
            "🚨 Fake Predictions",
            fake_count
        )

    with col3:

        st.metric(
            "✓ Real Predictions",
            real_count
        )

    st.subheader("🕒 Previous Predictions")

    # Show newest prediction first
    history_df = history_df.iloc[::-1]

    # --------------------------------------------------------
    # Display each prediction as a clean card
    # No dataframe/chart = no canvas artifact
    # --------------------------------------------------------

    for index, row in history_df.iterrows():

        prediction = row["Prediction"]

        if prediction == "FAKE":

            st.error(
                f"🚨 FAKE NEWS\n\n"
                f"Time: {row['Time']}\n\n"
                f"Confidence: {row['Confidence']}\n\n"
                f"Fake Probability: {row['Fake Probability']}\n\n"
                f"Real Probability: {row['Real Probability']}\n\n"
                f"Article: {row['Article']}"
            )

        else:

            st.success(
                f"✓ REAL NEWS\n\n"
                f"Time: {row['Time']}\n\n"
                f"Confidence: {row['Confidence']}\n\n"
                f"Fake Probability: {row['Fake Probability']}\n\n"
                f"Real Probability: {row['Real Probability']}\n\n"
                f"Article: {row['Article']}"
            )

else:

    st.info(
        "No prediction history yet. "
        "Analyze an article to create history."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fake News Detection System | "
    "Machine Learning + Natural Language Processing"
)

st.caption(
    "⚠️ This system is a text classification model, "
    "not an independent fact-checking system."
)