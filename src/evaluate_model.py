
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report,
    confusion_matrix
)
from sklearn.dummy import DummyClassifier


# File paths
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "processed" / "matches_features.csv"
MODEL_PATH = ROOT_DIR / "models" / "football_model.joblib"
REPORTS_DIR = ROOT_DIR / "reports"


# Original core features.
# These are only used to remove matches without enough historical statistics.
CORE_FEATURES = [
    "home_shots_avg",
    "away_shots_avg",
    "home_shots_on_target_avg",
    "away_shots_on_target_avg",
    "home_corners_avg",
    "away_corners_avg",
]


def main():

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data and trained model
    df = pd.read_csv(DATA_PATH)
    model_bundle = joblib.load(MODEL_PATH)

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    model_name = model_bundle["model_name"]

    print(f"Evaluating model: {model_name}")

    # Check that required columns exist
    required_columns = CORE_FEATURES + feature_columns + ["result", "date"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")

    # Keep matches with the required historical statistics
    df = df.dropna(subset=CORE_FEATURES).copy()

    # Sort chronologically
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Separate features and target
    X = df[feature_columns]
    y = df["result"]

    # Chronological 70/15/15 split
    train_end = int(len(df) * 0.70)
    validation_end = int(len(df) * 0.85)

    X_train_validation = X.iloc[:validation_end]
    y_train_validation = y.iloc[:validation_end]

    X_test = X.iloc[validation_end:]
    y_test = y.iloc[validation_end:]

    print(f"\nTraining + validation matches: {len(X_train_validation)}")
    print(f"Test matches: {len(X_test)}")

    # Model predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    logloss = log_loss(y_test, probabilities, labels=model.classes_)

    # Multiclass Brier score
    y_one_hot = pd.get_dummies(y_test).reindex(
        columns=model.classes_,
        fill_value=0
    ).values

    brier_score = ((y_one_hot - probabilities) ** 2).sum(axis=1).mean()

    # Baseline: always predict the most common training class
    baseline = DummyClassifier(strategy="prior")
    baseline.fit(
        X_train_validation,
        y_train_validation
    )

    baseline_predictions = baseline.predict(X_test)
    baseline_probabilities = baseline.predict_proba(X_test)

    baseline_accuracy = accuracy_score(y_test, baseline_predictions)
    baseline_logloss = log_loss(
        y_test,
        baseline_probabilities,
        labels=baseline.classes_
    )

    # Print results
    print("\n===== MODEL EVALUATION =====")
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log loss: {logloss:.4f}")
    print(f"Brier score: {brier_score:.4f}")

    print("\n===== BASELINE =====")
    print(f"Accuracy: {baseline_accuracy:.4f}")
    print(f"Log loss: {baseline_logloss:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    # Confusion matrix
    labels = model.classes_

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    print("\n===== CONFUSION MATRIX =====")
    print(pd.DataFrame(
        matrix,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels]
    ))

    # Save evaluation metrics
    evaluation = pd.DataFrame([{
        "model": model_name,
        "accuracy": accuracy,
        "log_loss": logloss,
        "brier_score": brier_score,
        "baseline_accuracy": baseline_accuracy,
        "baseline_log_loss": baseline_logloss,
        "training_validation_matches": len(X_train_validation),
        "test_matches": len(X_test)
    }])

    evaluation.to_csv(
        REPORTS_DIR / "model_evaluation.csv",
        index=False
    )

    pd.DataFrame(report).transpose().to_csv(
        REPORTS_DIR / "classification_report.csv"
    )

    pd.DataFrame(
        matrix,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels]
    ).to_csv(REPORTS_DIR / "confusion_matrix.csv")

    print("\nEvaluation reports saved in the reports folder.")


if __name__ == "__main__":
    main()
