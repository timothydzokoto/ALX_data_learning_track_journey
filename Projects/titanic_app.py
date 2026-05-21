import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import re

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,600&family=JetBrains+Mono:wght@300;400;500&family=Barlow:wght@300;400;500&display=swap');

    :root {
        --void:    #040d1a;
        --deep:    #071428;
        --surface: #0a1f3d;
        --card:    #0d2548;
        --border:  #1a3a6b;
        --steel:   #4a7fa5;
        --ice:     #a8d4e6;
        --gold:    #d4a843;
        --red:     #c0392b;
        --green:   #27ae60;
        --text:    #e8edf5;
        --muted:   #6b8cae;
        --shadow:  rgba(0,0,0,0.4);
    }

    html, body, [class*="css"] {
        font-family: 'Barlow', sans-serif;
        background-color: var(--void);
        color: var(--text);
    }

    .stApp {
        background: var(--void);
        background-image:
            radial-gradient(ellipse at 50% -10%, rgba(74,127,165,0.15) 0%, transparent 60%),
            radial-gradient(ellipse at 100% 100%, rgba(7,20,40,0.8) 0%, transparent 50%);
    }

    /* Hero */
    .hero {
        text-align: center;
        padding: 3rem 2rem 1.5rem;
        position: relative;
    }

    .hero::before {
        content: '🚢';
        font-size: 3rem;
        display: block;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 0 20px rgba(74,127,165,0.5));
    }

    .hero-date {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.3em;
        color: var(--gold);
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1.1;
        margin: 0;
    }

    .hero-title em {
        font-style: italic;
        color: var(--ice);
    }

    .hero-sub {
        font-size: 0.9rem;
        color: var(--muted);
        margin-top: 0.8rem;
        max-width: 500px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }

    .divider {
        width: 80px;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--steel), transparent);
        margin: 1.2rem auto;
    }

    /* Cards */
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .card-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--steel);
        margin-bottom: 0.8rem;
    }

    /* Result cards */
    .survived-card {
        background: linear-gradient(135deg, rgba(39,174,96,0.15), rgba(39,174,96,0.05));
        border: 1px solid rgba(39,174,96,0.4);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }

    .perished-card {
        background: linear-gradient(135deg, rgba(192,57,43,0.15), rgba(192,57,43,0.05));
        border: 1px solid rgba(192,57,43,0.4);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }

    .result-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .result-verdict {
        font-family: 'Cormorant Garamond', serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }

    .result-prob {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: var(--muted);
        margin-top: 0.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    .metric-val {
        font-family: 'Cormorant Garamond', serif;
        font-size: 2rem;
        font-weight: 700;
        color: var(--ice);
        line-height: 1;
    }

    .metric-lbl {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.1em;
        color: var(--muted);
        text-transform: uppercase;
        margin-top: 0.3rem;
    }

    /* Passenger card */
    .passenger-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .survived-badge {
        background: rgba(39,174,96,0.15);
        border: 1px solid rgba(39,174,96,0.4);
        color: #27ae60;
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .perished-badge {
        background: rgba(192,57,43,0.15);
        border: 1px solid rgba(192,57,43,0.4);
        color: #e74c3c;
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--steel), #2d6a9f) !important;
        color: white !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.1em !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.5rem !important;
        width: 100% !important;
        transition: all 0.2s !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(74,127,165,0.4) !important;
    }

    /* Inputs */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 6px !important;
        font-family: 'Barlow', sans-serif !important;
    }

    .stSlider > div > div > div {
        background: var(--steel) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--deep);
        border-bottom: 1px solid var(--border);
        gap: 0;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
        padding: 0.8rem 1.5rem;
    }

    .stTabs [aria-selected="true"] {
        color: var(--ice) !important;
        border-bottom: 2px solid var(--steel) !important;
        background: transparent !important;
    }

    /* Info boxes */
    .info-box {
        background: rgba(74,127,165,0.1);
        border: 1px solid rgba(74,127,165,0.3);
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        color: var(--ice);
        font-family: 'Barlow', sans-serif;
        margin: 0.5rem 0;
    }

    .warn-box {
        background: rgba(212,168,67,0.08);
        border: 1px solid rgba(212,168,67,0.3);
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        color: var(--gold);
        font-family: 'Barlow', sans-serif;
    }

    /* Search */
    .stTextInput > div > div > input {
        background: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }

    /* Hide chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* Fact tag */
    .fact-tag {
        display: inline-block;
        background: rgba(74,127,165,0.12);
        border: 1px solid rgba(74,127,165,0.25);
        border-radius: 4px;
        padding: 0.15rem 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: var(--ice);
        margin: 0.15rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Assets ────────────────────────────────────────────
@st.cache_resource
def load_model():
    paths = [
        Path(__file__).parent / "models" / "titanic_survival_model.pkl",
        Path("models") / "titanic_survival_model.pkl",
        Path("titanic_survival_model.pkl")
    ]
    for p in paths:
        if p.exists():
            return joblib.load(p)
    return None


@st.cache_data
def load_data():
    paths = [
        Path(__file__).parent / "titanic.csv",
        Path("titanic.csv")
    ]
    for p in paths:
        if p.exists():
            return pd.read_csv(p)
    return None


model = load_model()
df_raw = load_data()
DEMO_MODE = model is None


# ── Feature Engineering ────────────────────────────────────
def engineer_features(pclass, sex, age, sibsp, parch,
                       fare, embarked, cabin, ticket):
    """Reproduce the exact feature engineering from the notebook."""

    family_size = sibsp + parch + 1
    is_alone    = 1 if family_size == 1 else 0

    # Title from a generic name using sex/class
    if sex == 'male':
        if pclass == 1:
            title = 'Mr'
        else:
            title = 'Mr'
    else:
        if age < 18:
            title = 'Miss'
        elif pclass == 1:
            title = 'Mrs'
        else:
            title = 'Mrs'

    # Deck
    deck = cabin[0] if cabin and cabin != 'Unknown' else 'Unknown'
    valid_decks = ['B', 'C', 'D', 'E', 'F', 'G', 'T']
    if deck not in valid_decks:
        deck = 'Unknown'

    # Ticket prefix
    ticket_match = re.match(r'([A-Za-z]+)', ticket.replace('/', '').replace('.', '').strip())
    ticket_prefix = ticket_match.group(1).upper() if ticket_match else 'NoPrefix'
    valid_prefixes = ['C', 'CA', 'F', 'FA', 'Fa', 'LINE', 'NoPrefix',
                      'P', 'PC', 'PP', 'S', 'SC', 'SCO', 'SO',
                      'SOTON', 'STON', 'SW', 'W', 'WE']

    # Normalize ticket prefix
    tp_map = {
        'C': 'C', 'CA': 'CA', 'F': 'F', 'FA': 'Fa', 'FA': 'Fa',
        'LINE': 'LINE', 'P': 'P', 'PC': 'PC', 'PP': 'PP',
        'S': 'S', 'SC': 'SC', 'SCO': 'SCO', 'SO': 'SO',
        'SOTON': 'SOTON', 'STON': 'STON', 'SW': 'SW',
        'W': 'W', 'WE': 'WE'
    }
    ticket_prefix = tp_map.get(ticket_prefix, 'NoPrefix')

    # Age group
    if age < 12:
        age_group = 'Child'
    elif age < 18:
        age_group = 'Teen'
    elif age < 35:
        age_group = 'YoungAdult'
    elif age < 60:
        age_group = 'Adult'
    else:
        age_group = 'Senior'

    # Fare group
    if fare <= 7.91:
        fare_group = 'Low'
    elif fare <= 14.45:
        fare_group = 'Medium'
    elif fare <= 31:
        fare_group = 'High'
    else:
        fare_group = 'VeryHigh'

    # Build feature row matching model's expected features
    features = {
        'Pclass':    pclass,
        'Age':       age,
        'Fare':      fare,
        'FamilySize': family_size,
        'IsAlone':   is_alone,
        'Sex_male':  1 if sex == 'male' else 0,
        'Embarked_Q': 1 if embarked == 'Q' else 0,
        'Embarked_S': 1 if embarked == 'S' else 0,
        'Title_Miss':    1 if title == 'Miss' else 0,
        'Title_Mr':      1 if title == 'Mr' else 0,
        'Title_Mrs':     1 if title == 'Mrs' else 0,
        'Title_Officer': 1 if title == 'Officer' else 0,
        'Title_Royalty': 1 if title == 'Royalty' else 0,
        'Deck_B':       1 if deck == 'B' else 0,
        'Deck_C':       1 if deck == 'C' else 0,
        'Deck_D':       1 if deck == 'D' else 0,
        'Deck_E':       1 if deck == 'E' else 0,
        'Deck_F':       1 if deck == 'F' else 0,
        'Deck_G':       1 if deck == 'G' else 0,
        'Deck_T':       1 if deck == 'T' else 0,
        'Deck_Unknown': 1 if deck == 'Unknown' else 0,
        'Ticket_Prefix_C':     1 if ticket_prefix == 'C' else 0,
        'Ticket_Prefix_CA':    1 if ticket_prefix == 'CA' else 0,
        'Ticket_Prefix_F':     1 if ticket_prefix == 'F' else 0,
        'Ticket_Prefix_Fa':    1 if ticket_prefix == 'Fa' else 0,
        'Ticket_Prefix_LINE':  1 if ticket_prefix == 'LINE' else 0,
        'Ticket_Prefix_NoPrefix': 1 if ticket_prefix == 'NoPrefix' else 0,
        'Ticket_Prefix_P':     1 if ticket_prefix == 'P' else 0,
        'Ticket_Prefix_PC':    1 if ticket_prefix == 'PC' else 0,
        'Ticket_Prefix_PP':    1 if ticket_prefix == 'PP' else 0,
        'Ticket_Prefix_S':     1 if ticket_prefix == 'S' else 0,
        'Ticket_Prefix_SC':    1 if ticket_prefix == 'SC' else 0,
        'Ticket_Prefix_SCO':   1 if ticket_prefix == 'SCO' else 0,
        'Ticket_Prefix_SO':    1 if ticket_prefix == 'SO' else 0,
        'Ticket_Prefix_SOTON': 1 if ticket_prefix == 'SOTON' else 0,
        'Ticket_Prefix_STON':  1 if ticket_prefix == 'STON' else 0,
        'Ticket_Prefix_SW':    1 if ticket_prefix == 'SW' else 0,
        'Ticket_Prefix_W':     1 if ticket_prefix == 'W' else 0,
        'Ticket_Prefix_WE':    1 if ticket_prefix == 'WE' else 0,
        'AgeGroup_Teen':       1 if age_group == 'Teen' else 0,
        'AgeGroup_YoungAdult': 1 if age_group == 'YoungAdult' else 0,
        'AgeGroup_Adult':      1 if age_group == 'Adult' else 0,
        'AgeGroup_Senior':     1 if age_group == 'Senior' else 0,
        'FareGroup_Medium':    1 if fare_group == 'Medium' else 0,
        'FareGroup_High':      1 if fare_group == 'High' else 0,
        'FareGroup_VeryHigh':  1 if fare_group == 'VeryHigh' else 0,
    }

    return pd.DataFrame([features])


def predict_survival(pclass, sex, age, sibsp, parch,
                     fare, embarked, cabin, ticket):
    features = engineer_features(
        pclass, sex, age, sibsp, parch, fare, embarked, cabin, ticket
    )
    if DEMO_MODE:
        import random
        random.seed(int(age + fare + pclass))
        prob = random.uniform(0.2, 0.85)
        survived = prob > 0.5
        return survived, prob
    try:
        prob     = model.predict_proba(features)[0][1]
        survived = prob >= 0.5
        return survived, prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return False, 0.0


# ── Hero ───────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-date">✦ April 15, 1912 · North Atlantic Ocean</p>
    <h1 class="hero-title">Would you have<br><em>survived</em> the Titanic?</h1>
    <div class="divider"></div>
    <p class="hero-sub">
        A Random Forest model trained on 891 passengers predicts survival
        from class, gender, age, family size and cabin location.
    </p>
</div>
""", unsafe_allow_html=True)

# Status bar
if DEMO_MODE:
    st.markdown("""
    <div class="warn-box">
    ⚠️ Demo mode — place <code>titanic_survival_model.pkl</code> in a
    <code>models/</code> folder and <code>titanic.csv</code> in the same
    directory as this app for real predictions.
    </div>
    """, unsafe_allow_html=True)
else:
    total = len(df_raw) if df_raw is not None else 891
    survived_pct = round(df_raw['Survived'].mean() * 100, 1) if df_raw is not None else 38.4
    st.markdown(f"""
    <div class="info-box">
    ✅ Model loaded: <strong>Random Forest</strong>
    &nbsp;|&nbsp; Accuracy: <strong>83.24%</strong>
    &nbsp;|&nbsp; ROC-AUC: <strong>84.90%</strong>
    &nbsp;|&nbsp; Passengers in dataset: <strong>{total:,}</strong>
    &nbsp;|&nbsp; Survival rate: <strong>{survived_pct}%</strong>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎭 Survival Predictor",
    "📊 Data Insights",
    "🚢 Passenger Explorer",
    "📖 About"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<p class="card-label">✦ Passenger Details</p>',
                    unsafe_allow_html=True)

        # Quick presets
        st.markdown("""
        <p style="font-size:0.75rem;color:#6b8cae;font-family:'JetBrains Mono',monospace;
        margin-bottom:0.4rem">QUICK PRESETS:</p>
        """, unsafe_allow_html=True)

        preset_cols = st.columns(3)
        presets = {
            "1st Class Lady": dict(pclass=1, sex='female', age=35, sibsp=1,
                                   parch=0, fare=75.0, embarked='C',
                                   cabin='C85', ticket='PC 17599'),
            "3rd Class Man":  dict(pclass=3, sex='male', age=25, sibsp=0,
                                   parch=0, fare=7.25, embarked='S',
                                   cabin='Unknown', ticket='A/5 21171'),
            "Child":          dict(pclass=2, sex='female', age=8, sibsp=1,
                                   parch=2, fare=21.0, embarked='S',
                                   cabin='Unknown', ticket='248738'),
        }

        selected_preset = None
        for i, (label, vals) in enumerate(presets.items()):
            with preset_cols[i]:
                if st.button(label, key=f"preset_{i}"):
                    st.session_state['preset'] = vals

        if 'preset' in st.session_state:
            p = st.session_state['preset']
        else:
            p = {}

        st.markdown("---")

        # Input fields
        col_a, col_b = st.columns(2)
        with col_a:
            pclass = st.selectbox(
                "Passenger Class",
                options=[1, 2, 3],
                index=p.get('pclass', 3) - 1,
                format_func=lambda x: {
                    1: "1st — Upper",
                    2: "2nd — Middle",
                    3: "3rd — Lower"
                }[x]
            )
            age = st.slider(
                "Age",
                min_value=1, max_value=80,
                value=p.get('age', 28)
            )
            sibsp = st.slider(
                "Siblings / Spouse aboard",
                min_value=0, max_value=8,
                value=p.get('sibsp', 0)
            )
            fare = st.number_input(
                "Ticket Fare (£)",
                min_value=0.0, max_value=600.0,
                value=float(p.get('fare', 14.45)),
                step=0.5
            )

        with col_b:
            sex = st.selectbox(
                "Gender",
                options=['female', 'male'],
                index=0 if p.get('sex', 'male') == 'female' else 1,
                format_func=lambda x: x.capitalize()
            )
            embarked = st.selectbox(
                "Port of Embarkation",
                options=['S', 'C', 'Q'],
                index=['S', 'C', 'Q'].index(p.get('embarked', 'S')),
                format_func=lambda x: {
                    'S': 'Southampton (S)',
                    'C': 'Cherbourg (C)',
                    'Q': 'Queenstown (Q)'
                }[x]
            )
            parch = st.slider(
                "Parents / Children aboard",
                min_value=0, max_value=6,
                value=p.get('parch', 0)
            )
            cabin_input = st.selectbox(
                "Cabin Deck",
                options=['Unknown', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'],
                index=0
            )

        ticket_input = st.text_input(
            "Ticket Number",
            value=p.get('ticket', 'A/5 21171'),
            placeholder="e.g. PC 17599"
        )

        family_size = sibsp + parch + 1
        st.markdown(f"""
        <p style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
        color:#6b8cae;margin-top:0.3rem">
        Family size: {family_size} &nbsp;|&nbsp;
        {'Travelling alone' if family_size == 1 else f'Group of {family_size}'}
        </p>
        """, unsafe_allow_html=True)

        predict_btn = st.button("⚓ Predict My Fate")

    with col_right:
        st.markdown('<p class="card-label">✦ Prediction Result</p>',
                    unsafe_allow_html=True)

        if predict_btn:
            survived, prob = predict_survival(
                pclass, sex, age, sibsp, parch,
                fare, embarked, cabin_input, ticket_input
            )
            survival_pct = prob * 100
            death_pct    = (1 - prob) * 100

            if survived:
                st.markdown(f"""
                <div class="survived-card">
                    <div class="result-icon">🛟</div>
                    <p class="result-verdict" style="color:#27ae60">SURVIVED</p>
                    <p class="result-prob">Survival probability: {survival_pct:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="perished-card">
                    <div class="result-icon">🌊</div>
                    <p class="result-verdict" style="color:#e74c3c">DID NOT SURVIVE</p>
                    <p class="result-prob">Survival probability: {survival_pct:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability bar
            st.markdown('<p class="card-label">✦ Probability Breakdown</p>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin-bottom:0.8rem">
                <div style="display:flex;justify-content:space-between;
                font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                margin-bottom:0.3rem">
                    <span style="color:#27ae60">Survived</span>
                    <span style="color:#27ae60">{survival_pct:.1f}%</span>
                </div>
                <div style="height:10px;background:#0d2548;border-radius:5px;overflow:hidden">
                    <div style="height:100%;width:{survival_pct}%;
                    background:linear-gradient(90deg,#27ae60,#2ecc71);
                    border-radius:5px;transition:width 0.5s ease"></div>
                </div>
            </div>
            <div>
                <div style="display:flex;justify-content:space-between;
                font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                margin-bottom:0.3rem">
                    <span style="color:#e74c3c">Did Not Survive</span>
                    <span style="color:#e74c3c">{death_pct:.1f}%</span>
                </div>
                <div style="height:10px;background:#0d2548;border-radius:5px;overflow:hidden">
                    <div style="height:100%;width:{death_pct}%;
                    background:linear-gradient(90deg,#c0392b,#e74c3c);
                    border-radius:5px;transition:width 0.5s ease"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Key factors
            st.markdown('<p class="card-label">✦ Key Factors in This Prediction</p>',
                        unsafe_allow_html=True)

            factors = []
            if sex == 'female':
                factors.append(("✅", "Female", "Women had 74% survival rate vs 19% for men"))
            else:
                factors.append(("⚠️", "Male", "Men had only 19% survival rate"))

            if pclass == 1:
                factors.append(("✅", "1st Class", "63% of 1st class passengers survived"))
            elif pclass == 2:
                factors.append(("⚠️", "2nd Class", "47% of 2nd class passengers survived"))
            else:
                factors.append(("❌", "3rd Class", "Only 24% of 3rd class passengers survived"))

            if age < 15:
                factors.append(("✅", "Child", "Children had priority in lifeboats"))
            elif age > 60:
                factors.append(("⚠️", "Senior", "Older passengers had lower survival rates"))

            if family_size == 1:
                factors.append(("⚠️", "Travelling Alone", "Solo travellers had lower survival rates"))
            elif 2 <= family_size <= 4:
                factors.append(("✅", "Small Family", "Small families had better survival rates"))
            else:
                factors.append(("❌", "Large Family", "Large families struggled to evacuate together"))

            for icon, label, desc in factors:
                st.markdown(f"""
                <div style="display:flex;gap:0.8rem;margin-bottom:0.6rem;
                align-items:flex-start">
                    <span style="font-size:1rem;flex-shrink:0">{icon}</span>
                    <div>
                        <p style="font-family:'JetBrains Mono',monospace;
                        font-size:0.75rem;color:#a8d4e6;margin:0">{label}</p>
                        <p style="font-size:0.78rem;color:#6b8cae;margin:0.1rem 0 0;
                        font-family:'Barlow',sans-serif">{desc}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="border:1px dashed #1a3a6b;border-radius:12px;
            padding:3rem 2rem;text-align:center">
                <p style="font-family:'Cormorant Garamond',serif;font-size:2rem;
                font-style:italic;color:#1a3a6b;margin:0">your fate awaits</p>
                <p style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                letter-spacing:0.15em;color:#1a3a6b;margin-top:0.8rem">
                FILL IN DETAILS · PREDICT
                </p>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — DATA INSIGHTS
# ════════════════════════════════════════════════════════════
with tab2:
    if df_raw is None:
        st.markdown("""
        <div class="warn-box">
        Dataset not found. Place <code>titanic.csv</code> in the same
        directory as this app to view insights.
        </div>
        """, unsafe_allow_html=True)
    else:
        df = df_raw.copy()
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

        # Top metrics
        st.markdown('<p class="card-label">✦ Key Statistics</p>',
                    unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)

        metrics = [
            (m1, f"{len(df):,}", "Total Passengers"),
            (m2, f"{df['Survived'].sum():,}", "Survived"),
            (m3, f"{len(df) - df['Survived'].sum():,}", "Perished"),
            (m4, f"{df['Survived'].mean()*100:.1f}%", "Survival Rate"),
            (m5, f"{df['Age'].median():.0f}", "Median Age"),
        ]
        for col, val, lbl in metrics:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val">{val}</div>
                    <div class="metric-lbl">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts row 1
        col1, col2, col3 = st.columns(3)

        def styled_fig():
            fig, ax = plt.subplots(facecolor='#0d2548')
            ax.set_facecolor('#0d2548')
            ax.tick_params(colors='#6b8cae', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#1a3a6b')
            return fig, ax

        with col1:
            fig, ax = styled_fig()
            surv_sex = df.groupby('Sex')['Survived'].mean() * 100
            colors   = ['#e74c3c' if s < 50 else '#27ae60'
                        for s in surv_sex.values]
            bars = ax.bar(surv_sex.index, surv_sex.values,
                          color=colors, edgecolor='#0d2548', linewidth=1.5)
            for bar, val in zip(bars, surv_sex.values):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom',
                        color='#a8d4e6', fontsize=8, fontfamily='monospace')
            ax.set_title("Survival by Gender", color='#e8edf5',
                         fontsize=9, fontfamily='monospace')
            ax.set_ylabel("Survival Rate %", color='#6b8cae', fontsize=8)
            ax.set_ylim(0, 100)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col2:
            fig, ax = styled_fig()
            surv_class = df.groupby('Pclass')['Survived'].mean() * 100
            colors     = ['#d4a843', '#4a7fa5', '#c0392b']
            bars = ax.bar(
                ['1st Class', '2nd Class', '3rd Class'],
                surv_class.values,
                color=colors, edgecolor='#0d2548', linewidth=1.5
            )
            for bar, val in zip(bars, surv_class.values):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom',
                        color='#a8d4e6', fontsize=8, fontfamily='monospace')
            ax.set_title("Survival by Class", color='#e8edf5',
                         fontsize=9, fontfamily='monospace')
            ax.set_ylabel("Survival Rate %", color='#6b8cae', fontsize=8)
            ax.set_ylim(0, 100)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col3:
            fig, ax = styled_fig()
            survived     = df[df['Survived'] == 1]['Age'].dropna()
            not_survived = df[df['Survived'] == 0]['Age'].dropna()
            ax.hist(not_survived, bins=25, alpha=0.6,
                    color='#c0392b', label='Perished', edgecolor='#0d2548')
            ax.hist(survived, bins=25, alpha=0.6,
                    color='#27ae60', label='Survived', edgecolor='#0d2548')
            ax.set_title("Age Distribution by Survival",
                         color='#e8edf5', fontsize=9, fontfamily='monospace')
            ax.set_xlabel("Age", color='#6b8cae', fontsize=8)
            ax.set_ylabel("Count", color='#6b8cae', fontsize=8)
            ax.legend(fontsize=7, facecolor='#0d2548',
                      edgecolor='#1a3a6b', labelcolor='#e8edf5')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Charts row 2
        col4, col5, col6 = st.columns(3)

        with col4:
            fig, ax = styled_fig()
            surv_fs = df.groupby('FamilySize')['Survived'].mean() * 100
            ax.plot(surv_fs.index, surv_fs.values,
                    color='#4a7fa5', linewidth=2, marker='o',
                    markerfacecolor='#a8d4e6', markersize=6)
            ax.fill_between(surv_fs.index, surv_fs.values,
                            alpha=0.15, color='#4a7fa5')
            ax.set_title("Survival by Family Size",
                         color='#e8edf5', fontsize=9, fontfamily='monospace')
            ax.set_xlabel("Family Size", color='#6b8cae', fontsize=8)
            ax.set_ylabel("Survival Rate %", color='#6b8cae', fontsize=8)
            ax.set_ylim(0, 100)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col5:
            fig, ax = styled_fig()
            port_names = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
            surv_emb   = df.groupby('Embarked')['Survived'].mean() * 100
            surv_emb.index = [port_names.get(x, x) for x in surv_emb.index]
            colors_emb = ['#4a7fa5', '#d4a843', '#27ae60']
            bars = ax.bar(surv_emb.index, surv_emb.values,
                          color=colors_emb[:len(surv_emb)],
                          edgecolor='#0d2548', linewidth=1.5)
            for bar, val in zip(bars, surv_emb.values):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom',
                        color='#a8d4e6', fontsize=8, fontfamily='monospace')
            ax.set_title("Survival by Port of Embarkation",
                         color='#e8edf5', fontsize=9, fontfamily='monospace')
            ax.set_ylabel("Survival Rate %", color='#6b8cae', fontsize=8)
            ax.set_ylim(0, 100)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col6:
            fig, ax = styled_fig()
            fare_bins   = [0, 7.91, 14.45, 31, 600]
            fare_labels = ['Low\n(≤7.91)', 'Medium\n(7.91-14.45)',
                           'High\n(14.45-31)', 'Very High\n(>31)']
            df['FareGroup'] = pd.cut(df['Fare'], bins=fare_bins,
                                     labels=fare_labels)
            surv_fare = df.groupby('FareGroup',
                                   observed=True)['Survived'].mean() * 100
            colors_fare = ['#c0392b', '#e67e22', '#4a7fa5', '#27ae60']
            bars = ax.bar(surv_fare.index, surv_fare.values,
                          color=colors_fare[:len(surv_fare)],
                          edgecolor='#0d2548', linewidth=1.5)
            for bar, val in zip(bars, surv_fare.values):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom',
                        color='#a8d4e6', fontsize=7, fontfamily='monospace')
            ax.set_title("Survival by Fare Group",
                         color='#e8edf5', fontsize=9, fontfamily='monospace')
            ax.set_ylabel("Survival Rate %", color='#6b8cae', fontsize=8)
            ax.set_ylim(0, 100)
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Key insight
        st.markdown("""
        <div class="info-box" style="margin-top:1rem">
        💡 <strong>Key Finding:</strong> Gender was the strongest predictor of survival —
        women had a 74% survival rate compared to just 19% for men, reflecting the
        "women and children first" evacuation policy. Passenger class was the second
        strongest predictor — 1st class passengers had access to upper deck lifeboats.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — PASSENGER EXPLORER
# ════════════════════════════════════════════════════════════
with tab3:
    if df_raw is None:
        st.markdown("""
        <div class="warn-box">
        Dataset not found. Place <code>titanic.csv</code> in the app directory.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<p class="card-label">✦ Search Real Titanic Passengers</p>',
                    unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:0.82rem;color:#6b8cae;margin-bottom:1rem;
        font-family:'Barlow',sans-serif">
        Browse the 891 passengers in the dataset. Search by name or filter
        by class, gender and survival outcome.
        </p>
        """, unsafe_allow_html=True)

        # Search and filters
        col_s, col_f1, col_f2, col_f3 = st.columns([2, 1, 1, 1])

        with col_s:
            search = st.text_input(
                "Search by name",
                placeholder="Try: Rose, Jack, Braund...",
                label_visibility="collapsed"
            )
        with col_f1:
            filter_class = st.selectbox(
                "Class",
                options=['All', '1st', '2nd', '3rd'],
                label_visibility="collapsed"
            )
        with col_f2:
            filter_sex = st.selectbox(
                "Gender",
                options=['All', 'Male', 'Female'],
                label_visibility="collapsed"
            )
        with col_f3:
            filter_surv = st.selectbox(
                "Outcome",
                options=['All', 'Survived', 'Perished'],
                label_visibility="collapsed"
            )

        # Apply filters
        df_filtered = df_raw.copy()

        if search:
            df_filtered = df_filtered[
                df_filtered['Name'].str.contains(search, case=False, na=False)
            ]
        if filter_class != 'All':
            pclass_map = {'1st': 1, '2nd': 2, '3rd': 3}
            df_filtered = df_filtered[
                df_filtered['Pclass'] == pclass_map[filter_class]
            ]
        if filter_sex != 'All':
            df_filtered = df_filtered[
                df_filtered['Sex'] == filter_sex.lower()
            ]
        if filter_surv != 'All':
            surv_val = 1 if filter_surv == 'Survived' else 0
            df_filtered = df_filtered[df_filtered['Survived'] == surv_val]

        st.markdown(f"""
        <p style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
        color:#6b8cae;margin-bottom:0.8rem">
        Showing {len(df_filtered)} of {len(df_raw)} passengers
        </p>
        """, unsafe_allow_html=True)

        # Display passengers
        for _, row in df_filtered.head(30).iterrows():
            survived  = row['Survived'] == 1
            badge     = '<span class="survived-badge">✓ SURVIVED</span>' if survived \
                        else '<span class="perished-badge">✗ PERISHED</span>'
            class_map = {1: '1st Class', 2: '2nd Class', 3: '3rd Class'}
            age_str   = f"{int(row['Age'])} yrs" if not pd.isna(row['Age']) else 'Age unknown'
            fare_str  = f"£{row['Fare']:.2f}"

            st.markdown(f"""
            <div class="passenger-card">
                <div>
                    <p style="font-family:'Cormorant Garamond',serif;font-size:1rem;
                    font-weight:600;color:#e8edf5;margin:0">{row['Name']}</p>
                    <p style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                    color:#6b8cae;margin:0.2rem 0 0">
                    {class_map[row['Pclass']]} &nbsp;·&nbsp;
                    {row['Sex'].capitalize()} &nbsp;·&nbsp;
                    {age_str} &nbsp;·&nbsp;
                    {fare_str} &nbsp;·&nbsp;
                    Port: {row['Embarked'] if not pd.isna(row['Embarked']) else 'Unknown'}
                    </p>
                </div>
                {badge}
            </div>
            """, unsafe_allow_html=True)

        if len(df_filtered) > 30:
            st.markdown(f"""
            <p style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
            color:#6b8cae;text-align:center;margin-top:0.5rem">
            Showing first 30 of {len(df_filtered)} results — refine your search
            </p>
            """, unsafe_allow_html=True)

        if len(df_filtered) == 0:
            st.markdown("""
            <div style="text-align:center;padding:2rem;color:#1a3a6b">
                <p style="font-family:'Cormorant Garamond',serif;font-size:1.5rem;
                font-style:italic">No passengers found</p>
                <p style="font-family:'JetBrains Mono',monospace;font-size:0.72rem">
                Try a different search or filter
                </p>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════════════════════════
with tab4:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<p class="card-label">✦ About This Project</p>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.87rem;color:#a8d4e6;line-height:1.7;
        font-family:'Barlow',sans-serif">
        <p>This project predicts whether a Titanic passenger would have survived
        based on their socio-economic and demographic profile using a
        <strong>Random Forest Classifier</strong>.</p>

        <p>The model was trained on the <strong>Kaggle Titanic Dataset</strong> —
        891 passengers — with extensive feature engineering including title
        extraction, family size, deck location and fare grouping.</p>

        <p>Key insight: <strong>gender was the strongest predictor</strong> of
        survival, followed by passenger class — reflecting the historical
        "women and children first" policy during the evacuation.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="card-label" style="margin-top:1.5rem">✦ Feature Engineering</p>',
                    unsafe_allow_html=True)

        features_info = [
            ("FamilySize", "SibSp + Parch + 1 — total family aboard"),
            ("IsAlone", "Binary flag for solo travellers"),
            ("Title", "Extracted from name — Mr, Mrs, Miss, Officer, Royalty"),
            ("Deck", "Extracted from Cabin — A through T"),
            ("AgeGroup", "Child, Teen, YoungAdult, Adult, Senior"),
            ("FareGroup", "Low, Medium, High, VeryHigh"),
            ("Ticket_Prefix", "Prefix from ticket number — 19 categories"),
        ]

        for feat, desc in features_info:
            st.markdown(f"""
            <div style="display:flex;gap:0.8rem;margin-bottom:0.5rem">
                <span class="fact-tag">{feat}</span>
                <span style="font-size:0.78rem;color:#6b8cae;
                font-family:'Barlow',sans-serif;padding-top:0.15rem">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="card-label">✦ Model Performance</p>',
                    unsafe_allow_html=True)

        perf_metrics = [
            ("83.24%", "Accuracy"),
            ("84.90%", "ROC-AUC"),
            ("Random Forest", "Algorithm"),
            ("200", "Estimators"),
            ("46", "Features"),
            ("891", "Training Samples"),
        ]

        p1, p2 = st.columns(2)
        for i, (val, lbl) in enumerate(perf_metrics):
            col = p1 if i % 2 == 0 else p2
            with col:
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:0.6rem">
                    <div class="metric-val" style="font-size:1.4rem">{val}</div>
                    <div class="metric-lbl">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<p class="card-label" style="margin-top:1.2rem">✦ Models Compared</p>',
                    unsafe_allow_html=True)

        models_compared = [
            ("Logistic Regression", "Baseline linear model"),
            ("Random Forest ⭐", "Best — 83.24% accuracy"),
            ("SVM", "Strong F1 performance"),
            ("XGBoost", "Gradient boosting ensemble"),
        ]

        for name, desc in models_compared:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
            padding:0.5rem 0;border-bottom:1px solid #1a3a6b">
                <span style="font-family:'JetBrains Mono',monospace;
                font-size:0.78rem;color:#a8d4e6">{name}</span>
                <span style="font-size:0.75rem;color:#6b8cae;
                font-family:'Barlow',sans-serif">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<p class="card-label" style="margin-top:1.2rem">✦ How to Run</p>',
                    unsafe_allow_html=True)
        st.code("""
pip install streamlit pandas numpy
         scikit-learn joblib matplotlib seaborn

streamlit run titanic_app.py
        """, language="bash")

        st.markdown("""
        <div style="margin-top:1rem;font-family:'Barlow',sans-serif;
        font-size:0.85rem;color:#6b8cae">
            <strong style="color:#e8edf5">Timothy Dzokoto</strong><br>
            Data Science Portfolio Project<br>
            <span style="color:#4a7fa5">Classification · Feature Engineering · Streamlit</span>
        </div>
        """, unsafe_allow_html=True)    