# 📰 Fake News Detection using NLP

An NLP-based **Fake News Detection** system that analyzes news article text using Natural Language Processing techniques and Machine Learning to identify **potentially suspicious or fake news content**.

---

## 🚀 Project Overview

Fake and misleading news can spread rapidly through online platforms. This project uses **Natural Language Processing (NLP)** techniques to analyze the linguistic characteristics of a news article.

The system performs text preprocessing, feature extraction, sentiment analysis, POS tagging, Named Entity Recognition, clickbait detection, sensational language detection, and machine learning-based classification.

The final output provides an **NLP Suspicion Score** and a classification indicating whether the given article is potentially fake or not.

---

## 🎯 Objectives

* Detect potentially fake or suspicious news content.
* Preprocess and clean raw news articles.
* Extract meaningful textual features using **TF-IDF**.
* Perform sentiment analysis on news content.
* Analyze Parts of Speech (POS).
* Identify named entities using NER.
* Detect clickbait phrases and sensational language.
* Classify news using **Logistic Regression**.
* Provide an interactive command-line interface.

---

## 🧠 System Workflow

```text
                News Article
                     ↓
             Text Preprocessing
                     ↓
        Tokenization + Stopword Removal
                     ↓
                  Stemming
                     ↓
             TF-IDF Feature Extraction
                     ↓
        ┌────────────┬─────────────┐
        │            │             │
   Sentiment      POS Tagging     NER
   Analysis
        │            │             │
        └────────────┴─────────────┘
                     ↓
       Clickbait & Sensational Detection
                     ↓
             NLP Suspicion Score
                     ↓
            Machine Learning Model
                     ↓
            Logistic Regression
                     ↓
                 Prediction
```

---

## 🔍 System Features

### 🧹 1. Text Preprocessing

The input news article is cleaned and prepared using:

* Word Tokenization
* Stopword Removal
* Text Normalization
* Stemming

---

### 🔤 2. TF-IDF Feature Extraction

**Term Frequency–Inverse Document Frequency (TF-IDF)** is used to convert text into numerical features.

It helps identify words that are important within the news content.

---

### 😊 3. Sentiment Analysis

The system uses **VADER Sentiment Analysis** to identify the emotional characteristics of the article.

It provides:

* Positive Score
* Negative Score
* Neutral Score
* Compound Score

---

### 🏷️ 4. POS Tagging

Part-of-Speech tagging is performed to identify the grammatical structure of the text.

Examples include:

* Nouns
* Verbs
* Adjectives
* Adverbs

---

### 📍 5. Named Entity Recognition

NER identifies important entities present in the article, such as:

* People
* Organizations
* Locations
* Other named entities

---

### 🚨 6. Clickbait Detection

The system detects commonly used clickbait patterns and phrases that may be associated with attention-seeking news content.

Examples:

```text
You won't believe what happened next!
This shocking secret...
Click here to find out...
```

---

### ⚡ 7. Sensational Language Detection

The system identifies exaggerated, emotionally charged, or sensational words and phrases that may increase the suspicion level of an article.

---

### 📊 8. NLP Suspicion Score

Different linguistic indicators are combined to generate an **NLP Suspicion Score**.

Based on the analysis, the system provides a classification such as:

```text
POTENTIALLY FAKE
```

or

```text
POTENTIALLY GENUINE
```

> **Note:** The prediction is based on NLP and machine-learning patterns. It does not independently verify whether the claims in an article are factually true or false.

---

## 🤖 Machine Learning Model

The primary machine learning pipeline used in this project is:

### TF-IDF + Logistic Regression

```text
Raw News Text
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Logistic Regression
      ↓
News Classification
```

Logistic Regression is used as the primary text classification algorithm because it works effectively with high-dimensional sparse text features such as TF-IDF vectors.

---

## 🛠️ Technologies Used

| Technology          | Purpose                     |
| ------------------- | --------------------------- |
| Python              | Core programming language   |
| NLTK                | Natural Language Processing |
| Scikit-learn        | Machine Learning            |
| TF-IDF              | Text Feature Extraction     |
| Logistic Regression | News Classification         |
| VADER               | Sentiment Analysis          |
| Pandas              | Data Processing             |
| NumPy               | Numerical Operations        |

---

## 📁 Project Structure

```text
Fake-News-Detection/
│
├── data/
│
├── outputs/
│
├── src/
│   ├── classifier.py
│   ├── features.py
│   └── preprocessing.py
│
├── main.py
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SubbanBee/Fake-News-Detection.git
```

### 2. Navigate to the Project

```bash
cd Fake-News-Detection
```

### 3. Install Dependencies

```bash
pip install nltk scikit-learn pandas numpy
```

---

## 📚 NLTK Resources

Download the required NLTK packages:

```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')
```

---

## ▶️ How to Run

Run the main application:

```bash
python main.py
```

The program will ask you to enter a news article.

Example:

```text
Enter the news article:
```

After entering the article, the system performs NLP analysis and displays the prediction.

---

## 📊 Example

### Input

```text
BREAKING!!! You won't believe what happened next! 
This shocking news will change everything!
```

### Example Output

```text
NLP Suspicion Score: ...

Classification: POTENTIALLY FAKE
```

The actual score depends on the content of the article.

---

## 🔬 NLP Techniques Used

This project demonstrates multiple NLP concepts:

* Text Preprocessing
* Tokenization
* Stopword Removal
* Stemming
* TF-IDF
* Sentiment Analysis
* POS Tagging
* Named Entity Recognition
* Clickbait Detection
* Sensational Language Detection
* Text Classification

---

## ⚠️ Limitations

* NLP patterns alone cannot establish whether a news article is factually true.
* Genuine news can also contain emotional or sensational language.
* Model performance depends on the quality and representativeness of the training data.
* The current system does not independently verify claims using external trusted sources.
* A prediction should therefore be treated as an analytical signal rather than a final fact-check.

---

## 🔮 Future Enhancements

The project can be further improved by adding:

* Larger labeled fake-news datasets
* Random Forest and Naive Bayes comparison
* Deep Learning models
* BERT / Transformer-based models
* Streamlit web interface
* Real-time news analysis
* Source credibility analysis
* External fact-checking APIs
* Explainable AI
* Prediction history and analytics dashboard

---

## 🎓 Key Learning Outcomes

Through this project, the following concepts were implemented:

* Natural Language Processing
* Machine Learning
* Text Classification
* Feature Engineering
* Sentiment Analysis
* Linguistic Analysis
* Named Entity Recognition
* Model-based Prediction
* Python-based NLP Pipeline

---

## 👩‍💻 Author

**E.G. Subban Bee**

GitHub:
https://github.com/SubbanBee

---

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.

**Fake News Detection using NLP — Analyze. Detect. Understand.**
