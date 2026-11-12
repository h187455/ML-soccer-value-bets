import pandas as pd
from collections import defaultdict
from pathlib import Path


# Number of previous matches used to calculate team form.
WINDOW_SIZE = 5

# Match statistics recorded separately for the home and away teams.
STATISTICS = [
    "goals_scored",
    "goals_conceded",
    "shots",
    "shots_on_target",
    "corners",
    "fouls",
    "yellow_cards",
    "red_cards",
]

def get_team_features(previous_matches):
    """
    Calculate a team's recent statistics from its previous matches.

    Missing values are ignored when calculating an average. For example,
    if only 3 of the previous 5 matches have recorded shot statistics,
    the shots average is calculated from those 3 matches.
    """
    features = {}

    features["matches_played"] = len(previous_matches)

    for statistic in STATISTICS:
        values = [
            match[statistic]
            for match in previous_matches
            if pd.notna(match[statistic])
        ]

        # Leave the feature missing if no previous match has this statistic.
        features[f"{statistic}_avg"] = (
            sum(values) / len(values) if values else float("nan")
        )

    # Average points per match: win = 3, draw = 1, loss = 0.
    points = [match["points"] for match in previous_matches]
    features["points_avg"] = (
        sum(points) / len(points) if points else float("nan")
    )

    # Win rate over the previous matches.
    wins = [match["points"] == 3 for match in previous_matches]
    features["win_rate"] = (
        sum(wins) / len(wins) if wins else float("nan")
    )

    return features


def make_team_match_record(row, is_home):
    """Convert one match into the perspective of a single team."""
    if is_home:
        goals_scored = row["home_goals"]
        goals_conceded = row["away_goals"]
        result = row["result"]

        stat_columns = {
            "shots": "home_shots",
            "shots_on_target": "home_shots_on_target",
            "corners": "home_corners",
            "fouls": "home_fouls",
            "yellow_cards": "home_yellow_cards",
            "red_cards": "home_red_cards",
        }
    else:
        goals_scored = row["away_goals"]
        goals_conceded = row["home_goals"]

        # Reverse the result from the away team's perspective.
        result = {"H": "A", "D": "D", "A": "H"}[row["result"]]

        stat_columns = {
            "shots": "away_shots",
            "shots_on_target": "away_shots_on_target",
            "corners": "away_corners",
            "fouls": "away_fouls",
            "yellow_cards": "away_yellow_cards",
            "red_cards": "away_red_cards",
        }

    points = {"H": 3, "D": 1, "A": 0}[result]
    record = {
        "goals_scored": goals_scored,
        "goals_conceded": goals_conceded,
        "points": points,
    }

    for statistic, column in stat_columns.items():
        record[statistic] = row[column]

    return record


def add_prefixed_features(output_row, prefix, features):
    """Add features with home_ or away_ at the start of each column name."""
    for name, value in features.items():
        output_row[f"{prefix}_{name}"] = value


def create_features(df):
    """
    Create pre-match features for every match.

    All matches on the same date are processed before any of them are added
    to team history. This prevents one match on a date from influencing
    another match on that same date.
    """
    team_history = defaultdict(list)
    feature_rows = []

    # Ensure matches are processed in chronological order.
    df = df.sort_values(["date", "home_team", "away_team"]).copy()

    for match_date, matches_on_date in df.groupby("date", sort=True):
        # First calculate features for every match on this date.
        for _, row in matches_on_date.iterrows():
            home_history = team_history[row["home_team"]][-WINDOW_SIZE:]
            away_history = team_history[row["away_team"]][-WINDOW_SIZE:]

            home_features = get_team_features(home_history)
            away_features = get_team_features(away_history)

            output_row = {
                "date": row["date"],
                "season": row["season"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "result": row["result"],
            }

            add_prefixed_features(output_row, "home", home_features)
            add_prefixed_features(output_row, "away", away_features)

            # Differences can help the model compare the two teams directly.
            for name in home_features:
                home_value = home_features[name]
                away_value = away_features[name]

                if pd.notna(home_value) and pd.notna(away_value):
                    output_row[f"difference_{name}"] = home_value - away_value
                else:
                    output_row[f"difference_{name}"] = float("nan")

            feature_rows.append(output_row)

        # Only after all matches on this date have features do we update
        # team histories with the results and statistics from that date.
        for _, row in matches_on_date.iterrows():
            team_history[row["home_team"]].append(
                make_team_match_record(row, is_home=True)
            )
            team_history[row["away_team"]].append(
                make_team_match_record(row, is_home=False)
            )

    return pd.DataFrame(feature_rows)


def main():
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "processed" / "matches_clean.csv"
    output_path = project_root / "data" / "processed" / "matches_features.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find {input_path}. Run preprocess_data.py first."
        )

    df = pd.read_csv(input_path)

    required_columns = [
        "date", "season", "home_team", "away_team",
        "home_goals", "away_goals", "result",
        "home_shots", "away_shots",
        "home_shots_on_target", "away_shots_on_target",
        "home_corners", "away_corners",
        "home_fouls", "away_fouls",
        "home_yellow_cards", "away_yellow_cards",
        "home_red_cards", "away_red_cards",
    ]

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Parse dates and numeric values after loading the processed CSV.
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    numeric_columns = [
        column for column in required_columns
        if column not in {
            "date", "season", "home_team", "away_team", "result"
        }
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=["date", "season", "home_team", "away_team",
                "home_goals", "away_goals", "result"]
    )

    features_df = create_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print("Feature engineering complete!")
    print(f"Matches processed: {len(features_df)}")
    print(f"Output saved to: {output_path}")
    print(f"Features created: {len(features_df.columns) - 5}")

    print("\nMissing values in selected recent-statistic features:")
    columns_to_check = [
        "home_shots_avg", "away_shots_avg",
        "home_shots_on_target_avg", "away_shots_on_target_avg",
        "home_corners_avg", "away_corners_avg",
    ]
    print(features_df[columns_to_check].isna().sum())

    print(
        "\nNote: missing values are expected early in the dataset "
        "and where historical statistics were not recorded."
    )


if __name__ == "__main__":
    main()
