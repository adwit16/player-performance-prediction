import os
import json
import pandas as pd

RAW_DATA_PATH = "data/raw/bootstrap_static.json"
PROCESSED_DATA_DIR = "data/processed"

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def process_and_save(data, key, filename):
    df = pd.DataFrame(data[key])
    csv_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(csv_path, index=False)
    print(f"Saved {key} data to {csv_path}")
    return df

def main():
    print("Loading raw FPL data...")
    data = load_json(RAW_DATA_PATH)

    print("Processing elements (players)...")
    elements_df = process_and_save(data, "elements", "players.csv")

    print("Processing teams...")
    teams_df = process_and_save(data, "teams", "teams.csv")

    print("Processing element_types(positions)...")
    positions_df = process_and_save(data, "element_types", "positions.csv")

    print("All data processed and saved to 'data/processed' directory.")

if __name__ == "__main__":
    main()