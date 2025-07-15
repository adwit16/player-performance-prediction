import requests
import os 
import json


DATA_DIR = "data/raw"
URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

def fetch_and_save_data():
    response = requests.get(URL)
    if response.status_code == 200:
        data = response.json()
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(os.path.join(DATA_DIR, "bootstrap_static.json"), "w") as f:
            json.dump(data, f)
        print("Data downloaded and saved to data/raw/bootstrap_static.json")
    else:
        print(f"Failed to fetch data:{response.status_code}")

if __name__ == "__main__":
    fetch_and_save_data()