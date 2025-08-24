# src/features/make_features.py
import os
import pandas as pd

RAW_PLAYERS = "data/processed/players.csv"
RAW_TEAMS = "data/processed/teams.csv"
RAW_POSITIONS = "data/processed/positions.csv"
OUT_CSV = "data/processed/model_dataset.csv"
OUT_PARQUET = "data/processed/model_dataset.parquet"

def load_data():
    players = pd.read_csv(RAW_PLAYERS)
    teams = pd.read_csv(RAW_TEAMS)[["id", "name"]].rename(columns={"id": "team_id", "name": "team_name"})
    positions = pd.read_csv(RAW_POSITIONS)[["id", "singular_name"]].rename(
        columns={"id": "pos_id", "singular_name": "position"}
    )
    # Merge team & position labels
    df = (
        players
        .merge(teams, left_on="team", right_on="team_id", how="left")
        .merge(positions, left_on="element_type", right_on="pos_id", how="left")
    )

    # Clean some columns to numeric if present
    to_num = [
        "value_season", "influence", "creativity", "threat", "ict_index",
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "expected_goals_conceded"
    ]
    for c in to_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Safe minutes divisor
    mins90 = (df["minutes"].clip(lower=1) / 90)

    # Basic contributions
    if "goals_scored" in df.columns and "assists" in df.columns:
        df["goal_contrib"] = df["goals_scored"] + df["assists"]
        df["goals_per90"] = df["goals_scored"] / mins90
        df["assists_per90"] = df["assists"] / mins90
        df["contrib_per90"] = df["goal_contrib"] / mins90

    # Efficiency-style features
    df["points_per90"] = df["total_points"] / mins90
    df["points_per_minute"] = df["total_points"] / df["minutes"].clip(lower=1)

    if "value_season" in df.columns:
        df["points_per_million"] = df["total_points"] / df["value_season"].replace(0, pd.NA)

    # Keep a clean set of columns for modeling (only the ones that exist)
    base_cols = [
        "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
        "influence", "creativity", "threat", "ict_index", "bonus", "bps",
        "value_season", "yellow_cards", "red_cards", "saves", "penalties_scored",
        "penalties_missed", "own_goals", "goals_conceded",
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "expected_goals_conceded",
        # engineered
        "goal_contrib", "goals_per90", "assists_per90", "contrib_per90",
        "points_per90", "points_per_minute", "points_per_million",
        # categorical labels
        "position", "team_name"
    ]
    keep = [c for c in base_cols if c in df.columns]
    df_model = df[keep].copy()

    # Fill missing numerics with 0, categoricals with "Unknown"
    num_cols = df_model.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in ["position", "team_name"] if c in df_model.columns]

    df_model[num_cols] = df_model[num_cols].fillna(0)
    for c in cat_cols:
        df_model[c] = df_model[c].fillna("Unknown")

    # One-hot encode categoricals (drop_first=True to avoid dummy trap)
    if cat_cols:
        df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

    return df_model

def main():
    os.makedirs("data/processed", exist_ok=True)

    print("Loading processed tables …")
    df = load_data()
    print(f" → Players loaded: {len(df):,}")

    print("Engineering features …")
    df_model = engineer_features(df)
    print(f" → Model dataset shape: {df_model.shape}")

    print("Saving outputs …")
    df_model.to_csv(OUT_CSV, index=False)
    try:
        df_model.to_parquet(OUT_PARQUET, index=False)
    except Exception:
        # Parquet optional if pyarrow/fastparquet isn’t installed
        pass

    print(f"Saved: {OUT_CSV}")
    if os.path.exists(OUT_PARQUET):
        print(f"Saved: {OUT_PARQUET}")

if __name__ == "__main__":
    main()
