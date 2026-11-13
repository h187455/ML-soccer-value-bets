from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import football_api

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "football_model.joblib"

st.title("Premier League bet evaluator")

# ---- load model
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]

# ---- pick a fixture
teams = football_api.get_teams()
team_names = {team["id"]: team["name"] for team in teams}

fixtures = football_api.get_upcoming_fixtures()

if not fixtures:
    st.error("No upcoming fixtures returned by the API.")
    st.stop()

options = []

for fixture in fixtures:
    home_id, home_name, away_id, away_name = football_api.get_fixture_teams(
        fixture, teams
    )

    if home_id is None or away_id is None:
        continue

    options.append({
        "label": f'{fixture.get("date", "")[:10]}  {home_name} vs {away_name}',
        "home_id": home_id,
        "away_id": away_id,
    })

if not options:
    st.error(
        "Could not find the home/away teams in the fixtures. "
        "Here is what one fixture looks like:"
    )
    st.json(fixtures[0])
    st.stop()

choice = st.selectbox("Fixture", [option["label"] for option in options])
selected = next(option for option in options if option["label"] == choice)

# ---- odds
st.subheader("Bookmaker odds (decimal)")
col1, col2, col3 = st.columns(3)
odds_home = col1.number_input("Home win", min_value=1.01, value=2.00, step=0.05)
odds_draw = col2.number_input("Draw", min_value=1.01, value=3.50, step=0.05)
odds_away = col3.number_input("Away win", min_value=1.01, value=3.80, step=0.05)

# ---- evaluate
if st.button("Evaluate"):
    with st.spinner("Fetching data and calculating..."):
        features, matches_used = football_api.build_match_features(
            selected["home_id"], selected["away_id"]
        )

    X = pd.DataFrame([features]).reindex(columns=feature_columns).astype(float)

    probabilities = model.predict_proba(X)[0]
    probs = dict(zip(model.classes_, probabilities))

    odds = {"H": odds_home, "D": odds_draw, "A": odds_away}
    names = {"H": "Home win", "D": "Draw", "A": "Away win"}

    rows = []
    for outcome in ["H", "D", "A"]:
        expected_value = probs[outcome] * odds[outcome] - 1
        rows.append({
            "Outcome": names[outcome],
            "Odds": odds[outcome],
            "Bookmaker implied": 1 / odds[outcome],
            "Model probability": probs[outcome],
            "Expected value": expected_value,
        })

    results = pd.DataFrame(rows)

    st.subheader("Result")
    st.write(
        f"Elo: home {features['home_elo']:.0f}, away {features['away_elo']:.0f} "
        f"(built from {matches_used} matches from the API)"
    )

    st.dataframe(
        results.style.format({
            "Odds": "{:.2f}",
            "Bookmaker implied": "{:.1%}",
            "Model probability": "{:.1%}",
            "Expected value": "{:+.1%}",
        }),
        hide_index=True,
    )

    best = results.sort_values("Expected value", ascending=False).iloc[0]

    if best["Expected value"] > 0.05:
        st.success(
            f"Good bet: {best['Outcome']} at {best['Odds']:.2f} "
            f"(expected value {best['Expected value']:+.1%})"
        )
    else:
        st.error("No good bet here: the odds don't beat the model's probabilities.")

    st.caption("Expected value = model probability x odds - 1. Not financial advice.")

    with st.expander("Features sent to the model"):
        st.dataframe(X.T.rename(columns={0: "value"}))