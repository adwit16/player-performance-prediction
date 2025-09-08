import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load trained model
MODEL_PATH = "models/random_forest.pkl"
model = joblib.load(MODEL_PATH)

# Load player data
DATA_PATH = "data/processed/players.csv"
raw_df = pd.read_csv(DATA_PATH)

# Create a display_name column (prefer web_name, else combine first + second name)
if "web_name" in raw_df.columns:
    raw_df["player_name"] = raw_df["web_name"]
elif "first_name" in raw_df.columns and "second_name" in raw_df.columns:
    raw_df["player_name"] = raw_df["first_name"] + " " + raw_df["second_name"]
else:
    raw_df["player_name"] = "Unknown"

# Drop irrelevant columns for model input
cols_to_drop = ['first_name', 'second_name', 'web_name', 'photo', 'news',  'player_name']
df = raw_df.drop(columns=[col for col in cols_to_drop if col in raw_df.columns])

# Encode categorical columns
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Predict points
X = df.drop("total_points", axis=1)
df["predicted_points"] = model.predict(X)

undroped = raw_df[['player_name', 'id']]
df = pd.merge(df, undroped, on='id', how='inner')

# Merge back player_name for display
df["player_name"] = raw_df["player_name"]
if "team" in raw_df.columns:
    df["team"] = raw_df["team"]
if "element_type" in raw_df.columns:
    df["element_type"] = raw_df["element_type"]

# Sidebar options
st.sidebar.title("⚽ Build Your Fantasy Team")
budget = st.sidebar.slider("Select your budget (in millions)", 50, 120, 100)
position_filter = st.sidebar.multiselect(
    "Select positions", 
    options=raw_df["element_type"].unique() if "element_type" in raw_df.columns else []
)
team_filter = st.sidebar.selectbox(
    "Filter by Team (Optional)", ["All"] + list(raw_df["team"].unique()) if "team" in raw_df.columns else ["All"]
)

# Apply filters
filtered_df = df.copy()
if team_filter != "All" and "team" in df.columns:
    filtered_df = filtered_df[df["team"] == team_filter]
if position_filter and "element_type" in df.columns:
    filtered_df = filtered_df[filtered_df["element_type"].isin(position_filter)]

# Sort by predicted points
filtered_df = filtered_df.sort_values(by="predicted_points", ascending=False)

# Top N recommendations
st.subheader("🔥 Top Recommended Players")
top_n = st.sidebar.slider("Number of Players to Recommend", 5, 20, 10)

# print(filtered_df.columns
st.dataframe(filtered_df.head(top_n)[["player_name", "predicted_points", "now_cost", "team"]] if "now_cost" in df.columns else filtered_df.head(top_n)[["player_name", "predicted_points"]])

# Best XI suggestion under budget
# if "now_cost" in df.columns:
st.subheader("🏆 Suggested Best XI (under budget)")
best_team = filtered_df.copy()
best_team["cumulative_cost"] = best_team["now_cost"].cumsum()
best_team = best_team[best_team["cumulative_cost"] <= budget].head(11)
st.dataframe(best_team[["player_name", "predicted_points", "now_cost", "cumulative_cost", "team"]])
