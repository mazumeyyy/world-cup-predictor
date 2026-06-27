import os
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Resolve paths relative to this script's location
_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. PAGE SETUP & DESIGN
# ==========================================
st.set_page_config(page_title="2026 World Cup Predictor", page_icon="🏆", layout="centered")

st.title("FIFA World Cup 2026 Predictor")
st.markdown("### Powered by Machine Learning & Historical Data")
st.write("Select two teams below to simulate a high-stakes competitive match outcome.")

# ==========================================
# 2. DATA PROCESSING & TRAINING (LOCAL SYSTEM)
# ==========================================
@st.cache_data # Keeps the app blazing fast by loading data only once!
def load_and_train():
    # Load raw data from your data/ folder
    df_results = pd.read_csv(os.path.join(_DIR, 'results.csv'))
    df_shootouts = pd.read_csv(os.path.join(_DIR, 'shootouts.csv'))
    
    # Filter for major competitive tournaments
    comp_tournaments = ['FIFA World Cup', 'FIFA World Cup qualification', 
                        'UEFA European Championship', 'UEFA European Championship qualification', 'Copa América']
    df_filtered = df_results[df_results['tournament'].isin(comp_tournaments)].copy()
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    df_filtered = df_filtered.sort_values(by='date').reset_index(drop=True)
    
    # Calculate Winner column
    def get_winner(row):
        if row['home_score'] > row['away_score']: return row['home_team']
        elif row['away_score'] > row['home_score']: return row['away_team']
        return 'Draw'
    df_filtered['winner'] = df_filtered.apply(get_winner, axis=1)
    
    # Merge Shootout data
    df_shootouts['date'] = pd.to_datetime(df_shootouts['date'])
    df_merged = pd.merge(df_filtered, df_shootouts[['date', 'home_team', 'away_team', 'winner']], 
                         on=['date', 'home_team', 'away_team'], how='left', suffixes=('', '_shootout'))
    df_merged['winner'] = df_merged['winner_shootout'].fillna(df_merged['winner'])
    df_clean = df_merged.drop(columns=['winner_shootout'])
    
    # Calculate Rolling Points/Form
    def get_pts(row):
        if row['winner'] == 'Draw': return 1, 1
        elif row['winner'] == row['home_team']: return 3, 0
        return 0, 3
    pts = df_clean.apply(get_pts, axis=1)
    df_clean['home_points'] = [p[0] for p in pts]
    df_clean['away_points'] = [p[1] for p in pts]
    
    team_matches = []
    for _, row in df_clean.iterrows():
        team_matches.append({'date': row['date'], 'team': row['home_team'], 'points': row['home_points']})
        team_matches.append({'date': row['date'], 'team': row['away_team'], 'points': row['away_points']})
    df_teams = pd.DataFrame(team_matches).sort_values(by=['team', 'date'])
    df_teams['form_last_5'] = df_teams.groupby('team')['points'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    
    df_clean = pd.merge(df_clean, df_teams[['date', 'team', 'form_last_5']], left_on=['date', 'home_team'], right_on=['date', 'team'], how='left').rename(columns={'form_last_5': 'home_form'})
    df_clean.drop(columns=['team'], inplace=True)
    df_clean = pd.merge(df_clean, df_teams[['date', 'team', 'form_last_5']], left_on=['date', 'away_team'], right_on=['date', 'team'], how='left').rename(columns={'form_last_5': 'away_form'})
    df_clean.drop(columns=['team'], inplace=True)
    df_clean['home_form'] = df_clean['home_form'].fillna(1.0)
    df_clean['away_form'] = df_clean['away_form'].fillna(1.0)
    
    # Target Encoding
    def encode_target(row):
        if row['winner'] == row['home_team']: return 2
        elif row['winner'] == 'Draw': return 1
        return 0
    df_clean['target'] = df_clean.apply(encode_target, axis=1)
    
    # ML Train
    X = df_clean[['home_form', 'away_form']]
    y = df_clean['target']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Get distinct list of active unique countries for dropdown lists
    unique_countries = sorted(list(set(df_clean['home_team'].unique()) | set(df_clean['away_team'].unique())))
    
    return model, df_teams, unique_countries

# Run data setup pipeline
model, df_teams, countries = load_and_train()

# ==========================================
# 3. INTERACTIVE UI SIDE
# ==========================================
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("🏠 Select Home Team", countries, index=countries.index("Argentina") if "Argentina" in countries else 0)

with col2:
    away_team = st.selectbox("✈️ Select Away Team", countries, index=countries.index("France") if "France" in countries else 1)

if st.button("🚀 Run Match Simulation", use_container_width=True):
    if home_team == away_team:
        st.warning("A team cannot play against itself! Please select two different countries.")
    else:
        # Get latest form metrics
        try: home_f = df_teams[df_teams['team'] == home_team]['form_last_5'].iloc[-1]
        except: home_f = 1.0
            
        try: away_f = df_teams[df_teams['team'] == away_team]['form_last_5'].iloc[-1]
        except: away_f = 1.0
        
        # Predict
        features = pd.DataFrame([[home_f, away_f]], columns=['home_form', 'away_form'])
        probabilities = model.predict_proba(features)[0]
        
        # Display Interactive Output
        st.markdown("### Simulation Results")
        
        st.write(f"**{home_team} Wins:** {probabilities[2]*100:.1f}%")
        st.progress(float(probabilities[2]))
        
        st.write(f"**Draw:** {probabilities[1]*100:.1f}%")
        st.progress(float(probabilities[1]))
        
        st.write(f"**{away_team} Wins:** {probabilities[0]*100:.1f}%")
        st.progress(float(probabilities[0]))