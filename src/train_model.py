import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


print("\n==============================================")
print("        FAKE NEWS ML MODEL TRAINING")
print("==============================================\n")


# 1. LOAD DATASET

print("Loading datasets...")

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

print(f"Fake articles : {len(fake)}")
print(f"True articles : {len(true)}")


# 2. CREATE LABELS

fake["label"] = 0
true["label"] = 1


# 3. COMBINE DATASETS

data = pd.concat([fake, true], ignore_index=True)

data["content"] = (
    data["title"].fillna("") + " " +
    data["text"].fillna("")
)

data = data[["content", "label"]]

data = data[data["content"].str.strip() != ""]

data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


print(f"\nTotal articles : {len(data)}")


# 4. TRAIN / TEST SPLIT

X = data["content"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training articles : {len(X_train)}")
print(f"Testing articles  : {len(X_test)}")


# 5. TF-IDF FEATURE EXTRACTION

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape :", X_train_tfidf.shape)
print("TF-IDF testing shape  :", X_test_tfidf.shape)


# 6. TRAIN LOGISTIC REGRESSION

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# 7. PREDICTION

y_pred = model.predict(X_test_tfidf)


# 8. EVALUATION

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n==============================================")
print("              MODEL RESULTS")
print("==============================================")

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


print("\n--- Classification Report ---")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["FAKE", "REAL"]
    )
)


print("\n--- Confusion Matrix ---")

print(confusion_matrix(y_test, y_pred))


# 9. SAVE MODEL

print("\nSaving model...")

joblib.dump(
    model,
    "outputs/fake_news_model.pkl"
)

joblib.dump(
    vectorizer,
    "outputs/tfidf_vectorizer.pkl"
)

print("Model saved successfully!")

print("\n==============================================")
print("          TRAINING COMPLETED")
print("==============================================\n")
