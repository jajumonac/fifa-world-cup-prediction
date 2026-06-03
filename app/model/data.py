import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# Start dates of each FIFA World Cup (used for the recent-WC-window feature)
WC_START_DATES = pd.to_datetime([
    "1990-06-08", "1994-06-17", "1998-06-10", "2002-05-31",
    "2006-06-09", "2010-06-11", "2014-06-12", "2018-06-14", "2022-11-20",
])

GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Maps WC official team name → name used in the historical dataset
DATASET_NAME = {
    "Czechia": "Czech Republic",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Ivory Coast": "Côte d'Ivoire",
    "Curacao": "Curaçao",
}


def get_dataset_name(team: str) -> str:
    return DATASET_NAME.get(team, team)


def get_all_teams() -> list[str]:
    return [t for g in GROUPS.values() for t in g]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df
