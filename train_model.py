import json
import os

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = "Iris.csv"
MODEL_FILE = "iris_model.pkl"
METRICS_FILE = "metrics.json"

FEATURES = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]

TARGET = "Species"


def train_model():
    print("Loading Iris dataset...")

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"{DATA_FILE} not found in the project root."
        )

    data = pd.read_csv(DATA_FILE)

    required_columns = FEATURES + [TARGET]
    missing = set(required_columns) - set(data.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = data.dropna(subset=required_columns)

    print("Dataset shape:", data.shape)
    print("Species:", sorted(data[TARGET].unique()))

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(max_iter=1000, random_state=42),
        ),
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    labels = sorted(y.unique())
    matrix = confusion_matrix(
        y_test, predictions, labels=labels
    )

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))
    print("Species order:", labels)
    print("Confusion matrix:")
    print(matrix)

    joblib.dump(model, MODEL_FILE)

    metrics = {
        "accuracy": float(accuracy),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "total_records": int(len(data)),
        "species": labels,
        "confusion_matrix": matrix.tolist(),
    }

    with open(METRICS_FILE, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nModel saved:", MODEL_FILE)
    print("Metrics saved:", METRICS_FILE)

    return accuracy


if __name__ == "__main__":
    train_model()
