import csv
import os
from datetime import datetime


HISTORY_FILE = "outputs/prediction_history.csv"


def save_prediction(
    article,
    prediction,
    confidence,
    fake_probability,
    real_probability
):
    """
    Save one prediction into the prediction history CSV file.
    """

    os.makedirs("outputs", exist_ok=True)

    file_exists = os.path.exists(HISTORY_FILE)

    with open(
        HISTORY_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        # Create header when the file is created for the first time
        if not file_exists:
            writer.writerow([
                "Time",
                "Article",
                "Prediction",
                "Confidence",
                "Fake Probability",
                "Real Probability"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            article.replace("\n", " ")[:500],
            prediction,
            f"{confidence * 100:.2f}%",
            f"{fake_probability * 100:.2f}%",
            f"{real_probability * 100:.2f}%"
        ])


def load_history():
    """
    Load all previously saved predictions.
    """

    if not os.path.exists(HISTORY_FILE):
        return []

    with open(
        HISTORY_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        return list(reader)