
import pandas as pd
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss


# File paths
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "processed" / "matches_features.csv"
MODEL_PATH = ROOT_DIR / "models" / "football_model.joblib"
REPORTS_DIR = ROOT_DIR / "reports"


# Required features for a match to be included.
# These are eligibility checks, not the complete model input.
CORE_FEATURES = [
    "home_shots_avg",
    "away_shots_avg",
    "home_shots_on_target_avg",
    "away_shots_on_target_avg",
    "home_corners_avg",
    "away_corners_avg",
]


# Columns that should not be used as model inputs
NON_FEATURE_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "result",
]


def main():

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load processed data
    df = pd.read_csv(DATA_PATH)

    # Check required columns
    for column in CORE_FEATURES + NON_FEATURE_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    # Remove matches without the required historical statistics
    df = df.dropna(subset=CORE_FEATURES).copy()

    # Sort chronologically
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Select all numeric features, including Elo
    feature_columns = [
        column for column in df.select_dtypes(include="number").columns
        if column not in NON_FEATURE_COLUMNS
    ]

    X = df[feature_columns]
    y = df["result"]

    print(f"Matches available: {len(df)}")
    print(f"Number of features: {len(feature_columns)}")

    # Chronological 70/15/15 split
    train_end = int(len(df) * 0.70)
    validation_end = int(len(df) * 0.85)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_validation = X.iloc[train_end:validation_end]
    y_validation = y.iloc[train_end:validation_end]

    X_test = X.iloc[validation_end:]
    y_test = y.iloc[validation_end:]

    print(f"\nTraining matches: {len(X_train)}")
    print(f"Validation matches: {len(X_validation)}")
    print(f"Test matches: {len(X_test)}")

    # Define models
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ))
        ])
    }

    # Train and compare models using validation data
    results = []

    print("\n===== MODEL COMPARISON =====")

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)
        probabilities = model.predict_proba(X_validation)

        accuracy = accuracy_score(y_validation, predictions)
        loss = log_loss(
            y_validation,
            probabilities,
            labels=model.classes_
        )

        results.append({
            "model": name,
            "accuracy": accuracy,
            "log_loss": loss
        })

        print(f"\n{name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Log loss: {loss:.4f}")

    # Select model with lowest validation log loss
    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    best_model_name = results_df.sort_values("log_loss").iloc[0]["model"]

    print(f"\nSelected model: {best_model_name}")

    # Retrain selected model on training + validation data
    X_train_validation = X.iloc[:validation_end]
    y_train_validation = y.iloc[:validation_end]

    best_model = models[best_model_name]
    best_model.fit(X_train_validation, y_train_validation)

    # Save model and information needed for predictions
    model_bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": feature_columns,
        "classes": best_model.classes_
    }

    joblib.dump(model_bundle, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print("Training complete.")


if __name__ == "__main__":
    main()
