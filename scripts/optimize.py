import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary


df = pd.read_csv('predicted_points_gw3.csv')  


required_cols = ['player_name', 'position', 'team', 'price', 'predicted_points']
df = df[required_cols]


prob = LpProblem("FPL_Optimization", LpMaximize)


players = df['player_name'].tolist()
x = LpVariable.dicts('player', players, 0, 1, cat=LpBinary)


prob += lpSum(df.loc[df['player_name'] == p, 'predicted_points'].values[0] * x[p] for p in players)


prob += lpSum(x[p] for p in players) == 15

# 2. Position constraints
positions = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
for pos, count in positions.items():
    prob += lpSum(x[p] for p in df[df['position'] == pos]['player_name']) == count

# 3. Max 3 players per team
for team in df['team'].unique():
    prob += lpSum(x[p] for p in df[df['team'] == team]['player_name']) <= 3

# 4. Budget constraint
prob += lpSum(df.loc[df['player_name'] == p, 'price'].values[0] * x[p] for p in players) <= 100


prob.solve()


selected_players = [p for p in players if x[p].value() == 1]
squad_df = df[df['player_name'].isin(selected_players)].copy()

# Sort by position for easy viewing
squad_df = squad_df.sort_values(by='position')

print("\nOptimal 15-player FPL Squad:\n")
print(squad_df[['player_name', 'position', 'team', 'price', 'predicted_points']])


# Example: choose top scoring players within valid formations
def select_starting_11(df):
    # Start with 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD
    # Simple rule: pick highest predicted points for each position
    starters = []
    
    # GK: pick top 1
    gk = df[df['position']=='GK'].sort_values(by='predicted_points', ascending=False).head(1)
    starters.append(gk)
    
    # DEF: pick top 3-5, pick 4 as example
    def_players = df[df['position']=='DEF'].sort_values(by='predicted_points', ascending=False).head(4)
    starters.append(def_players)
    
    # MID: pick top 3-5, pick 4 as example
    mid_players = df[df['position']=='MID'].sort_values(by='predicted_points', ascending=False).head(4)
    starters.append(mid_players)
    
    # FWD: pick top 1-3, pick 2 as example
    fwd_players = df[df['position']=='FWD'].sort_values(by='predicted_points', ascending=False).head(2)
    starters.append(fwd_players)
    
    starting_11 = pd.concat(starters)
    return starting_11

starting_11 = select_starting_11(squad_df)
bench = squad_df[~squad_df['player_name'].isin(starting_11['player_name'])]

print("\nStarting 11:\n")
print(starting_11[['player_name', 'position', 'team', 'price', 'predicted_points']])

print("\nBench:\n")
print(bench[['player_name', 'position', 'team', 'price', 'predicted_points']])
