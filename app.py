import streamlit as st
import pandas as pd

from src.classifier import classify_news
from src.features import extract_features
from src.history import save_prediction, load_history
from src.web_verification import verify_news


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake News Detection AI",
    page_icon="📰",
    layout="wide"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📰 Fake News AI")

    st.markdown("---")

    st.subheader("🔍 System Modules")

    st.markdown("""
    - 📝 News Article Input
    - 🧹 Text Preprocessing
    - 🔤 TF-IDF Feature Extraction
    - 🤖 ML Prediction
    - 🌐 Real-Time Source Verification
    - 😊 Sentiment Analysis
    - 🏷️ POS Analysis
    - 📍 Named Entity Recognition
    - 🚨 Clickbait Detection
    - ⚡ Sensational Language Detection
    - 📊 Prediction History
    """)

    st.markdown("---")

    st.subheader("🤖 Model")

    st.write("TF-IDF + Logistic Regression")

    st.markdown("---")

    st.caption("Machine Learning + NLP + Real-Time Verification")


# ============================================================
# MAIN TITLE
# ============================================================

st.title("📰 Fake News Detection System")

st.write(
    "Analyze news articles using Natural Language Processing "
    "and Machine Learning to predict whether the article is "
    "likely to be FAKE or REAL."
)


# ============================================================
# NEWS ARTICLE INPUT
# ============================================================

st.header("📝 News Article Analysis")

article_text = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste the news article here..."
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button("🔍 Analyze News", type="primary"):

    if not article_text.strip():

        st.warning("⚠️ Please enter a news article first.")

    else:

        # ====================================================
        # MACHINE LEARNING PREDICTION
        # ====================================================

        prediction, confidence, fake_probability, real_probability = (
            classify_news(article_text)
        )


        # ====================================================
        # NLP FEATURE EXTRACTION
        # ====================================================

        features = extract_features(article_text)


        # ====================================================
        # REAL-TIME SOURCE VERIFICATION
        # ====================================================

        verification = verify_news(article_text)


        # ====================================================
        # SAVE PREDICTION HISTORY
        # ====================================================

        save_prediction(
            article_text,
            prediction,
            confidence,
            fake_probability,
            real_probability
        )


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.header("🎯 Prediction Result")

        if prediction == "FAKE":

            st.error("🚨 FAKE NEWS")

        else:

            st.success("✅ REAL NEWS")


        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


        st.info(
            "The trained machine learning model classified this "
            "article based on learned textual patterns."
        )


        # ====================================================
        # PREDICTION PROBABILITY
        # ====================================================

        st.header("📊 Prediction Probability")

        st.bar_chart(
            {
                "Fake": [fake_probability * 100],
                "Real": [real_probability * 100]
            }
        )


        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🚨 Fake Probability",
                f"{fake_probability * 100:.2f}%"
            )

        with col2:

            st.metric(
                "✅ Real Probability",
                f"{real_probability * 100:.2f}%"
            )


        # ====================================================
        # REAL-TIME SOURCE VERIFICATION
        # ====================================================

        st.markdown("---")

        st.header("🌐 Real-Time Source Verification")

        if verification["status"] == "error":

            st.warning(
                "⚠️ Real-time source verification is currently unavailable."
            )

            st.caption(
                verification.get(
                    "message",
                    "Unable to retrieve current news sources."
                )
            )


        elif verification["status"] == "not_found":

            st.info(
                "🔎 No related current news reports were found."
            )

            if verification.get("query"):

                st.write(
                    f"**Search Query:** {verification['query']}"
                )


        elif verification["status"] == "found":

            st.success(
                f"📰 Found {len(verification['results'])} "
                f"related news report(s)."
            )

            st.write(
                f"**Search Query:** {verification['query']}"
            )

            st.write(
                f"**Trusted Sources Found:** "
                f"{verification['trusted_count']}"
            )

            st.markdown("---")

            for result in verification["results"]:

                source_label = result["source"]

                if result["trusted"]:

                    source_label += " ✅ Trusted Source"


                st.markdown(
                    f"### 📰 {result['title']}"
                )

                st.write(
                    f"**Source:** {source_label}"
                )

                if result["published"]:

                    st.write(
                        f"**Published:** {result['published']}"
                    )

                if result["link"]:

                    st.markdown(
                        f"[🔗 Read Source]({result['link']})"
                    )

                st.markdown("---")


            st.info(
                "ℹ️ Source verification provides supporting evidence "
                "from current news reports. It does not automatically "
                "change the ML model's FAKE/REAL prediction."
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
        # SENTIMENT ANALYSIS
        # ====================================================

        st.header("😊 Sentiment Analysis")

        st.write(
            f"**Overall Sentiment:** "
            f"{features['sentiment']}"
        )


        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Positive",
                f"{features['positive_score'] * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Negative",
                f"{features['negative_score'] * 100:.2f}%"
            )

        with col3:

            st.metric(
                "Neutral",
                f"{features['neutral_score'] * 100:.2f}%"
            )

        with col4:

            st.metric(
                "Compound",
                f"{features['compound_score']:.3f}"
            )


        # ====================================================
        # WRITING STYLE
        # ====================================================

        st.header("✍️ Writing Style")

        col1, col2, col3 = st.columns(3)

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
                "🔠 Uppercase Words",
                features["uppercase_word_count"]
            )


        # ====================================================
        # SENSATIONAL LANGUAGE
        # ====================================================

        st.header("🚨 Sensational Language")

        if features["sensational_words"]:

            st.write(
                "Detected sensational words:"
            )

            for word in features["sensational_words"]:

                st.warning(
                    f"⚡ {word}"
                )

        else:

            st.success(
                "No predefined sensational words detected."
            )


        # ====================================================
        # CLICKBAIT ANALYSIS
        # ====================================================

        st.header("🎯 Clickbait Pattern Analysis")

        if features["clickbait_phrases"]:

            st.write(
                "Detected clickbait phrases:"
            )

            for phrase in features["clickbait_phrases"]:

                st.warning(
                    f"🎯 {phrase}"
                )

        else:

            st.success(
                "No predefined clickbait phrases detected."
            )


        # ====================================================
        # POS ANALYSIS
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
        # NAMED ENTITY RECOGNITION
        # ====================================================

        st.header("📍 Named Entity Recognition")

        if features["named_entities"]:

            for entity, label in features["named_entities"]:

                st.write(
                    f"**{entity}** — `{label}`"
                )

        else:

            st.write(
                "No named entities detected."
            )


        # ====================================================
        # REPEATED WORDS
        # ====================================================

        st.header("🔁 Repeated Words")

        repeated_words = features["repeated_words"]

        if repeated_words:

            for word, count in repeated_words:

                st.write(
                    f"**{word}** — {count} times"
                )

        else:

            st.write(
                "No repeated words detected."
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

st.header("🤖 Model Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.info(
        "**Feature Extraction**\n\n"
        "TF-IDF"
    )

with col2:

    st.info(
        "**Classification Model**\n\n"
        "Logistic Regression"
    )

with col3:

    st.info(
        "**Approach**\n\n"
        "Machine Learning + NLP + Real-Time Verification"
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

st.markdown("---")

st.header("📊 Prediction History")

history = load_history()

if history:

    total_predictions = len(history)

    fake_count = sum(
        1
        for item in history
        if item["Prediction"] == "FAKE"
    )

    real_count = sum(
        1
        for item in history
        if item["Prediction"] == "REAL"
    )


    # --------------------------------------------------------
    # HISTORY SUMMARY
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Analyzed",
            total_predictions
        )

    with col2:

        st.metric(
            "Fake",
            fake_count
        )

    with col3:

        st.metric(
            "Real",
            real_count
        )


    # --------------------------------------------------------
    # HISTORY ENTRIES
    # --------------------------------------------------------

    st.subheader("Recent Predictions")

    for item in reversed(history):

        prediction = item["Prediction"]

        if prediction == "FAKE":

            st.error(
                f"🚨 **FAKE** | "
                f"{item['Time']} | "
                f"Confidence: {item['Confidence']}\n\n"
                f"{item['Article']}"
            )

        else:

            st.success(
                f"✅ **REAL** | "
                f"{item['Time']} | "
                f"Confidence: {item['Confidence']}\n\n"
                f"{item['Article']}"
            )

else:

    st.info(
        "No prediction history available yet."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "📰 Fake News Detection System | "
    "Machine Learning + Natural Language Processing + "
    "Real-Time Source Verification"
)