import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "players.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "predicted_points_gw3.csv")


df_players = pd.read_csv(DATA_PATH)

# Drop unwanted columns
if "name" in df_players.columns:
    df_players = df_players.drop("name", axis=1)

# Drop unnamed columns (common from CSVs)
df_players = df_players.loc[:, ~df_players.columns.str.contains('^Unnamed')]

for col in df_players.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df_players[col] = le.fit_transform(df_players[col])

if 'total_points' in df_players.columns:
    X_pred = df_players.drop(columns=['total_points'])
else:
    X_pred = df_players.copy()

# Ensure columns match the trained model
rf_model = joblib.load(MODEL_PATH)
X_pred = X_pred[rf_model.feature_names_in_]

df_players['predicted_points'] = rf_model.predict(X_pred)


df_players.to_csv(OUTPUT_PATH, index=False)
print(f"Predictions saved to {OUTPUT_PATH}")
