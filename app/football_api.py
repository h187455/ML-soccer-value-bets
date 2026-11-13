import math
import time
from collections import defaultdict
from itertools import groupby

import requests
import streamlit as st

BASE_URL = "https://sportapi.ai/api"
LEAGUE_ID = 24
WINDOW_SIZE = 5

# Elo settings (same as the training feature script)
INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 100

# Statistics we average over the last matches
STAT_NAMES = [
    "shots",
    "shots_on_target",
    "corners",
    "fouls",
    "yellow_cards",
    "red_cards",
]

HEADERS = {
    "X-Api-Key": st.secrets["SPORTAPI_KEY"]
}


# ---------------------------------------------------------------- API calls
def api_request(endpoint, retries=3):
    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=60)
            response.raise_for_status()
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt == retries - 1:
                raise
            time.sleep(2)

    data = response.json()

    if not data.get("success", False):
        raise RuntimeError(data.get("error", "SportAPI request failed"))

    return data


def get_teams():
    data = api_request(f"leagues/{LEAGUE_ID}")
    return data.get("teams", [])


def get_upcoming_fixtures():
    data = api_request(f"leagues/{LEAGUE_ID}")
    fixtures = data.get("upcoming_fixtures", [])
    return [fixture for fixture in fixtures if fixture.get("id")]


def get_fixture_teams(fixture, teams):
    """
    Find (home_id, home_name, away_id, away_name) in a fixture, whatever the
    API calls the fields. Returns None for anything it can't find.
    """
    by_id = {str(team["id"]): team for team in teams}
    by_name = {team["name"].strip().casefold(): team for team in teams}

    result = []

    for side in ["home", "away"]:
        team = None

        candidates = [
            fixture.get(f"{side}_id"),
            fixture.get(f"{side}_team_id"),
            fixture.get(f"{side}_team"),
            fixture.get(side),
            fixture.get(f"{side}_name"),
            fixture.get(f"{side}_team_name"),
        ]

        for value in candidates:
            if isinstance(value, dict):
                value = value.get("id", value.get("name"))

            if value is None:
                continue

            if str(value) in by_id:
                team = by_id[str(value)]
            elif str(value).strip().casefold() in by_name:
                team = by_name[str(value).strip().casefold()]

            if team:
                break

        result += [team["id"], team["name"]] if team else [None, None]

    return tuple(result)


def get_completed_matches(team_id):
    """All completed Premier League matches the API returns for a team."""
    data = api_request(f"teams/{team_id}")
    matches = data.get("matches", [])

    return [
        match for match in matches
        if str(match.get("league_id")) == str(LEAGUE_ID)
        and match.get("status") in ["FT", "AET", "PENS"]
        and match.get("home_score") is not None
        and match.get("away_score") is not None
    ]


def get_team_matches(team_id):
    """The team's most recent completed matches (newest first)."""
    matches = get_completed_matches(team_id)
    matches.sort(key=lambda match: match.get("date", ""), reverse=True)
    return matches[:WINDOW_SIZE]


def get_match_stats(fixture_id):
    data = api_request(f"fixtures/{fixture_id}/stats")

    if not data.get("available"):
        return None

    return data.get("data")


# ------------------------------------------------------------- team form
def get_team_form(team_id):
    matches = get_team_matches(team_id)

    if not matches:
        raise ValueError(f"No recent Premier League matches for team {team_id}")

    history = []

    for match in matches:
        stats = get_match_stats(match["id"])
        is_home = match["home_id"] == team_id

        if is_home:
            goals_scored = match["home_score"]
            goals_conceded = match["away_score"]
            team_stats = stats.get("home", {}) if stats else {}
        else:
            goals_scored = match["away_score"]
            goals_conceded = match["home_score"]
            team_stats = stats.get("away", {}) if stats else {}

        if goals_scored > goals_conceded:
            points, win = 3, 1
        elif goals_scored == goals_conceded:
            points, win = 1, 0
        else:
            points, win = 0, 0

        history.append({
            "goals_scored": goals_scored,
            "goals_conceded": goals_conceded,
            "points": points,
            "win": win,
            "stats": team_stats,
        })

    return history


def average_stat(history, stat_name):
    values = [
        item["stats"].get(stat_name)
        for item in history
        if item["stats"].get(stat_name) is not None
    ]

    if not values:
        return math.nan

    return sum(values) / len(values)


def build_team_features(team_id, prefix):
    history = get_team_form(team_id)
    count = len(history)

    features = {
        f"{prefix}_matches_played": count,
        f"{prefix}_goals_scored_avg": sum(h["goals_scored"] for h in history) / count,
        f"{prefix}_goals_conceded_avg": sum(h["goals_conceded"] for h in history) / count,
        f"{prefix}_points_avg": sum(h["points"] for h in history) / count,
        f"{prefix}_win_rate": sum(h["win"] for h in history) / count,
    }

    for stat in STAT_NAMES:
        features[f"{prefix}_{stat}_avg"] = average_stat(history, stat)

    return features


# -------------------------------------------------------------------- Elo
def calculate_expected_score(home_elo, away_elo):
    home_rating = home_elo + HOME_ADVANTAGE
    return 1 / (1 + 10 ** ((away_elo - home_rating) / 400))


@st.cache_data(ttl=3600)
def calculate_all_elo():
    """
    Build Elo ratings for every team using all completed matches
    the API gives us. Matches on the same date use the same
    pre-date ratings, and ratings update after that date.
    """
    all_matches = {}

    for team in get_teams():
        for match in get_completed_matches(team["id"]):
            all_matches[match["id"]] = match  # dedupe by match id

    matches = sorted(all_matches.values(), key=lambda m: m.get("date", ""))

    elo = defaultdict(lambda: INITIAL_ELO)

    for _, day_matches in groupby(matches, key=lambda m: m["date"][:10]):
        changes = defaultdict(float)

        for match in day_matches:
            home_id = match["home_id"]
            away_id = match["away_id"]

            expected_home = calculate_expected_score(elo[home_id], elo[away_id])

            if match["home_score"] > match["away_score"]:
                actual_home = 1
            elif match["home_score"] == match["away_score"]:
                actual_home = 0.5
            else:
                actual_home = 0

            changes[home_id] += K_FACTOR * (actual_home - expected_home)
            changes[away_id] += K_FACTOR * ((1 - actual_home) - (1 - expected_home))

        for team_id, change in changes.items():
            elo[team_id] += change

    return dict(elo), len(matches)


# ----------------------------------------------------- full feature set
def build_match_features(home_id, away_id):
    home_features = build_team_features(home_id, "home")
    away_features = build_team_features(away_id, "away")

    features = {}
    features.update(home_features)
    features.update(away_features)

    elo, matches_used = calculate_all_elo()
    home_elo = elo.get(home_id, INITIAL_ELO)
    away_elo = elo.get(away_id, INITIAL_ELO)

    features["home_elo"] = home_elo
    features["away_elo"] = away_elo
    features["difference_elo"] = home_elo - away_elo

    # difference_<name> = home_<name> - away_<name>
    for key in home_features:
        name = key[len("home_"):]
        features[f"difference_{name}"] = home_features[key] - away_features[f"away_{name}"]

    return features, matches_used