
import pandas as pd
from pathlib import Path


def preprocess_data():
    # Project paths
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "raw" / "results.csv"
    output_path = project_root / "data" / "processed" / "matches_clean.csv"

    # Load dataset
    df = pd.read_csv(input_path, encoding="cp1252")
    # Columns we want to preserve
    columns = [
        "Season", "DateTime",
        "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",
        "HS", "AS", "HST", "AST",
        "HC", "AC", "HF", "AF",
        "HY", "AY", "HR", "AR"
    ]

    df = df[columns].copy()

    # Rename columns
    df = df.rename(columns={
        "Season": "season",
        "DateTime": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "FTR": "result",
        "HS": "home_shots",
        "AS": "away_shots",
        "HST": "home_shots_on_target",
        "AST": "away_shots_on_target",
        "HC": "home_corners",
        "AC": "away_corners",
        "HF": "home_fouls",
        "AF": "away_fouls",
        "HY": "home_yellow_cards",
        "AY": "away_yellow_cards",
        "HR": "home_red_cards",
        "AR": "away_red_cards"
    })

    # Convert dates and numeric columns
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    numeric_columns = [
        "home_goals", "away_goals",
        "home_shots", "away_shots",
        "home_shots_on_target", "away_shots_on_target",
        "home_corners", "away_corners",
        "home_fouls", "away_fouls",
        "home_yellow_cards", "away_yellow_cards",
        "home_red_cards", "away_red_cards"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove rows without essential match information
    df = df.dropna(subset=[
        "date", "home_team", "away_team",
        "home_goals", "away_goals", "result"
    ])

    # Clean team names
    df["home_team"] = df["home_team"].str.strip()
    df["away_team"] = df["away_team"].str.strip()

    # Remove invalid scores and duplicate matches
    df = df[
        (df["home_goals"] >= 0) &
        (df["away_goals"] >= 0)
    ]

    df = df.drop_duplicates(
        subset=["date", "home_team", "away_team"]
    )

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # Save processed data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Report dataset information
    print("Preprocessing complete!")
    print("Total matches:", len(df))
    print("Date range:", df["date"].min(), "to", df["date"].max())

    print("\nAvailable detailed statistics:")
    print(df[[
        "home_shots", "away_shots",
        "home_shots_on_target", "away_shots_on_target",
        "home_corners", "away_corners"
    ]].notna().sum())

    print("\nMissing values:")
    print(df.isna().sum())


if __name__ == "__main__":
    preprocess_data()
