import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="MBTI Personality Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

    :root {
        --ink:      #0a0a0f;
        --paper:    #f5f0e8;
        --cream:    #ede8dc;
        --gold:     #c9a84c;
        --gold2:    #e8c97a;
        --rust:     #c4622d;
        --sage:     #4a7c6f;
        --slate:    #2d3561;
        --mist:     #8b9cb0;
        --border:   #d4cfc4;
        --shadow:   rgba(10,10,15,0.12);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--paper);
        color: var(--ink);
    }

    .stApp {
        background: var(--paper);
        background-image:
            radial-gradient(ellipse at 20% 0%, rgba(201,168,76,0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 100%, rgba(74,124,111,0.06) 0%, transparent 50%);
    }

    /* Hero */
    .hero {
        text-align: center;
        padding: 3rem 2rem 2rem;
        position: relative;
    }

    .hero-eyebrow {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: var(--gold);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.8rem;
        font-weight: 900;
        line-height: 1.05;
        color: var(--ink);
        margin: 0;
    }

    .hero-title em {
        font-style: italic;
        color: var(--gold);
    }

    .hero-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        color: var(--mist);
        margin-top: 1rem;
        max-width: 520px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }

    .divider {
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, var(--gold), transparent);
        margin: 1.5rem auto;
    }

    /* Cards */
    .card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.8rem;
        box-shadow: 0 2px 20px var(--shadow);
        margin-bottom: 1.2rem;
    }

    .card-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--gold);
        margin-bottom: 0.6rem;
    }

    /* Result card */
    .result-hero {
        background: var(--ink);
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .result-hero::before {
        content: '';
        position: absolute;
        top: -40%;
        left: -20%;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(201,168,76,0.15) 0%, transparent 70%);
        pointer-events: none;
    }

    .result-type {
        font-family: 'Playfair Display', serif;
        font-size: 5rem;
        font-weight: 900;
        color: var(--gold);
        line-height: 1;
        letter-spacing: 0.05em;
    }

    .result-name {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        font-style: italic;
        color: var(--paper);
        margin-top: 0.3rem;
    }

    .result-desc {
        font-size: 0.85rem;
        color: var(--mist);
        margin-top: 0.5rem;
        font-family: 'DM Sans', sans-serif;
    }

    /* Dimension bars */
    .dim-row {
        display: flex;
        align-items: center;
        margin: 0.6rem 0;
        gap: 0.8rem;
    }

    .dim-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        width: 80px;
        color: var(--mist);
        flex-shrink: 0;
    }

    .dim-bar-track {
        flex: 1;
        height: 8px;
        background: var(--cream);
        border-radius: 4px;
        overflow: hidden;
    }

    .dim-bar-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, var(--gold), var(--gold2));
        transition: width 0.6s ease;
    }

    .dim-value {
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        color: var(--gold);
        width: 45px;
        text-align: right;
        flex-shrink: 0;
    }

    .dim-letter {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        font-weight: 700;
        width: 20px;
        flex-shrink: 0;
    }

    /* Personality badges */
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.8rem;
    }

    .badge {
        background: var(--cream);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        font-family: 'DM Mono', monospace;
        color: var(--ink);
    }

    .badge-gold {
        background: rgba(201,168,76,0.12);
        border-color: rgba(201,168,76,0.4);
        color: var(--gold);
    }

    /* Textarea */
    .stTextArea textarea {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        background: white !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--ink) !important;
        padding: 1rem !important;
        line-height: 1.6 !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 2px rgba(201,168,76,0.15) !important;
    }

    /* Button */
    .stButton > button {
        background: var(--ink) !important;
        color: var(--gold) !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        border: 1px solid var(--ink) !important;
        border-radius: 8px !important;
        padding: 0.7rem 2rem !important;
        width: 100% !important;
        transition: all 0.2s !important;
    }

    .stButton > button:hover {
        background: var(--gold) !important;
        color: var(--ink) !important;
        border-color: var(--gold) !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        border-bottom: 1px solid var(--border);
        gap: 0;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--mist);
        padding: 0.7rem 1.5rem;
        border-bottom: 2px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        color: var(--gold) !important;
        border-bottom: 2px solid var(--gold) !important;
        background: transparent !important;
    }

    /* Info box */
    .info-box {
        background: rgba(201,168,76,0.08);
        border: 1px solid rgba(201,168,76,0.3);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: #8b6914;
        font-family: 'DM Sans', sans-serif;
        margin: 0.5rem 0;
    }

    .warn-box {
        background: rgba(196,98,45,0.06);
        border: 1px solid rgba(196,98,45,0.25);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: var(--rust);
        font-family: 'DM Sans', sans-serif;
    }

    /* Hide streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* Metric */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }

    .metric-item {
        flex: 1;
        text-align: center;
        padding: 1rem;
        background: var(--cream);
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    .metric-val {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--gold);
    }

    .metric-lbl {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        color: var(--mist);
        text-transform: uppercase;
        margin-top: 0.2rem;
    }

    /* Example chips */
    .example-chip {
        display: inline-block;
        background: white;
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        font-size: 0.75rem;
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
        margin: 0.2rem;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)


# ── MBTI Data ──────────────────────────────────────────────
MBTI_DATA = {
    'INTJ': {
        'name': 'The Architect',
        'desc': 'Strategic, independent and decisive thinkers who see the world as a place for improvement.',
        'traits': ['Strategic', 'Independent', 'Decisive', 'Private', 'Analytical'],
        'famous': ['Elon Musk', 'Stephen Hawking', 'Nikola Tesla'],
        'color': '#2d3561'
    },
    'INTP': {
        'name': 'The Logician',
        'desc': 'Innovative inventors with an unquenchable thirst for knowledge.',
        'traits': ['Analytical', 'Objective', 'Reserved', 'Flexible', 'Curious'],
        'famous': ['Albert Einstein', 'Bill Gates', 'Charles Darwin'],
        'color': '#3a5a8c'
    },
    'ENTJ': {
        'name': 'The Commander',
        'desc': 'Bold, imaginative and strong-willed leaders who always find a way.',
        'traits': ['Bold', 'Charismatic', 'Efficient', 'Confident', 'Strategic'],
        'famous': ['Steve Jobs', 'Gordon Ramsay', 'Margaret Thatcher'],
        'color': '#c4622d'
    },
    'ENTP': {
        'name': 'The Debater',
        'desc': 'Smart and curious thinkers who cannot resist an intellectual challenge.',
        'traits': ['Innovative', 'Argumentative', 'Charismatic', 'Energetic'],
        'famous': ['Mark Twain', 'Celine Dion', 'Leonardo da Vinci'],
        'color': '#c4622d'
    },
    'INFJ': {
        'name': 'The Advocate',
        'desc': 'Quiet and mystical, yet very inspiring and tireless idealists.',
        'traits': ['Insightful', 'Principled', 'Passionate', 'Altruistic'],
        'famous': ['Martin Luther King Jr.', 'Nelson Mandela', 'Oprah'],
        'color': '#4a7c6f'
    },
    'INFP': {
        'name': 'The Mediator',
        'desc': 'Poetic, kind and altruistic people always eager to help a good cause.',
        'traits': ['Empathetic', 'Creative', 'Idealistic', 'Loyal', 'Open-minded'],
        'famous': ['J.R.R. Tolkien', 'William Shakespeare', 'Princess Diana'],
        'color': '#4a7c6f'
    },
    'ENFJ': {
        'name': 'The Protagonist',
        'desc': 'Charismatic and inspiring leaders able to mesmerize their listeners.',
        'traits': ['Charismatic', 'Empathetic', 'Reliable', 'Natural Leader'],
        'famous': ['Barack Obama', 'Oprah Winfrey', 'Maya Angelou'],
        'color': '#c9a84c'
    },
    'ENFP': {
        'name': 'The Campaigner',
        'desc': 'Enthusiastic, creative and sociable free spirits who can always find a reason to smile.',
        'traits': ['Enthusiastic', 'Creative', 'Sociable', 'Optimistic'],
        'famous': ['Robin Williams', 'Walt Disney', 'Ellen DeGeneres'],
        'color': '#c9a84c'
    },
    'ISTJ': {
        'name': 'The Logistician',
        'desc': 'Practical and fact-minded individuals whose reliability cannot be doubted.',
        'traits': ['Responsible', 'Thorough', 'Dependable', 'Traditional'],
        'famous': ['Queen Elizabeth II', 'Warren Buffett', 'Jeff Bezos'],
        'color': '#5c6bc0'
    },
    'ISFJ': {
        'name': 'The Defender',
        'desc': 'Dedicated and warm protectors always ready to defend their loved ones.',
        'traits': ['Supportive', 'Reliable', 'Patient', 'Observant'],
        'famous': ['Beyoncé', 'Kate Middleton', 'Mother Teresa'],
        'color': '#5c6bc0'
    },
    'ESTJ': {
        'name': 'The Executive',
        'desc': 'Excellent administrators who are unsurpassed at managing things and people.',
        'traits': ['Organized', 'Loyal', 'Patient', 'Reliable', 'Honest'],
        'famous': ['Judge Judy', 'Michelle Obama', 'Henry Ford'],
        'color': '#8d6e63'
    },
    'ESFJ': {
        'name': 'The Consul',
        'desc': 'Extraordinarily caring, social and popular people always eager to help.',
        'traits': ['Caring', 'Loyal', 'Sensitive', 'Warm', 'Practical'],
        'famous': ['Taylor Swift', 'Bill Clinton', 'Jennifer Garner'],
        'color': '#8d6e63'
    },
    'ISTP': {
        'name': 'The Virtuoso',
        'desc': 'Bold and practical experimenters, masters of all kinds of tools.',
        'traits': ['Optimistic', 'Creative', 'Direct', 'Practical', 'Observant'],
        'famous': ['Clint Eastwood', 'Michael Jordan', 'Bruce Lee'],
        'color': '#546e7a'
    },
    'ISFP': {
        'name': 'The Adventurer',
        'desc': 'Flexible and charming artists always ready to explore new things.',
        'traits': ['Charming', 'Sensitive', 'Imaginative', 'Curious'],
        'famous': ['Michael Jackson', 'Marilyn Monroe', 'Frida Kahlo'],
        'color': '#546e7a'
    },
    'ESTP': {
        'name': 'The Entrepreneur',
        'desc': 'Smart, energetic and very perceptive people who truly enjoy living on the edge.',
        'traits': ['Bold', 'Rational', 'Direct', 'Perceptive', 'Sociable'],
        'famous': ['Donald Trump', 'Madonna', 'Ernest Hemingway'],
        'color': '#bf360c'
    },
    'ESFP': {
        'name': 'The Entertainer',
        'desc': 'Spontaneous, energetic and enthusiastic people with life being never boring around them.',
        'traits': ['Spontaneous', 'Energetic', 'Enthusiastic', 'Playful'],
        'famous': ['Adele', 'Jamie Oliver', 'Marilyn Monroe'],
        'color': '#bf360c'
    }
}

DIM_DESCRIPTIONS = {
    'Mind':    {'I': ('Introvert', 'Recharges through solitude and reflection'),
                'E': ('Extrovert', 'Energized by social interaction and external world')},
    'Energy':  {'N': ('Intuitive', 'Focuses on patterns, meaning and future possibilities'),
                'S': ('Sensing', 'Focuses on concrete facts, details and present reality')},
    'Nature':  {'T': ('Thinking', 'Makes decisions based on logic and objective analysis'),
                'F': ('Feeling', 'Makes decisions based on values and emotional impact')},
    'Tactics': {'J': ('Judging', 'Prefers structure, planning and decisive action'),
                'P': ('Perceiving', 'Prefers flexibility, spontaneity and open options')}
}


# ── Load Models ────────────────────────────────────────────
@st.cache_resource
def load_models():
    models_dir = Path(__file__).parent / "models"
    if not models_dir.exists():
        models_dir = Path("models")
    if not models_dir.exists():
        return None

    assets = {}
    try:
        assets['vectorizer'] = joblib.load(models_dir / 'tfidf_vectorizer.pkl')
        for dim in ['Mind', 'Energy', 'Nature', 'Tactics']:
            assets[dim] = joblib.load(models_dir / f'mbti_{dim.lower()}_model.pkl')

        # Load metadata if exists
        meta_path = models_dir / 'model_metadata.json'
        if meta_path.exists():
            with open(meta_path) as f:
                assets['metadata'] = json.load(f)

        return assets
    except Exception as e:
        return None


assets = load_models()
DEMO_MODE = assets is None


# ── NLP Helpers ────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    for pkg in ['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'punkt_tab']:
        try:
            nltk.download(pkg, quiet=True)
        except:
            pass

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    NLTK_READY = True
except:
    NLTK_READY = False

MBTI_TYPES = [
    'infp', 'infj', 'intp', 'intj', 'entp', 'enfp',
    'istp', 'isfp', 'entj', 'istj', 'enfj', 'isfj',
    'estp', 'esfp', 'estj', 'esfj'
]


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    for mbti in MBTI_TYPES:
        text = re.sub(mbti, '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    if NLTK_READY:
        try:
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
            return ' '.join(tokens)
        except:
            pass
    return text


def predict_personality(text):
    if DEMO_MODE or not assets:
        # Demo predictions
        import random
        random.seed(len(text))
        dims = {'Mind': 'I', 'Energy': 'N', 'Nature': 'F', 'Tactics': 'P'}
        confs = {d: random.uniform(55, 85) for d in dims}
        mbti = ''.join(dims.values())
        return mbti, dims, confs

    cleaned  = clean_text(text)
    features = assets['vectorizer'].transform([cleaned])
    dims     = {}
    confs    = {}

    for dim in ['Mind', 'Energy', 'Nature', 'Tactics']:
        model = assets[dim]
        pred  = model.predict(features)[0]
        dims[dim] = pred

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confs[dim] = round(float(max(proba)) * 100, 1)
        elif hasattr(model, 'decision_function'):
            score = abs(float(model.decision_function(features)[0]))
            confs[dim] = round(min(score / 2 * 100, 99.0), 1)
        else:
            confs[dim] = 70.0

    mbti = ''.join(dims.values())
    return mbti, dims, confs


# ── Hero ───────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-eyebrow">✦ NLP · Machine Learning · Personality Science</p>
    <h1 class="hero-title">Can your words reveal<br><em>who you are?</em></h1>
    <div class="divider"></div>
    <p class="hero-sub">
        An NLP model trained on 8,675 social media users predicts your
        Myers-Briggs personality type from the way you write.
    </p>
</div>
""", unsafe_allow_html=True)

# Model status
if DEMO_MODE:
    st.markdown("""
    <div class="warn-box">
    ⚠️ Running in demo mode — add your trained model files to a <code>models/</code>
    folder to enable real predictions.
    </div>
    """, unsafe_allow_html=True)
else:
    meta = assets.get('metadata', {})
    full_acc = meta.get('full_type_accuracy', 'N/A')
    st.markdown(f"""
    <div class="info-box">
    ✅ Models loaded — Full MBTI reconstruction accuracy: <strong>{full_acc}</strong>
    &nbsp;|&nbsp; Training samples: <strong>{meta.get('training_samples', '8,675')}</strong>
    &nbsp;|&nbsp; Status: <strong>Ready</strong>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Tabs ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Explore Types", "📖 About"])

# ════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown('<p class="card-label">✦ Your Text Input</p>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:0.85rem;color:#8b9cb0;margin-bottom:0.8rem;font-family:'DM Sans',sans-serif">
        Write or paste anything — journal entries, messages, social media posts,
        stream of consciousness. The more text the better (aim for 100+ words).
        </p>
        """, unsafe_allow_html=True)

        user_text = st.text_area(
            label="",
            placeholder="Start writing here... Share your thoughts, feelings, ideas, opinions. Write naturally as you would in a private journal or to a close friend.",
            height=220,
            label_visibility="collapsed"
        )

        # Example texts
        st.markdown("""
        <p style="font-size:0.72rem;font-family:'DM Mono',monospace;color:#8b9cb0;
        letter-spacing:0.1em;text-transform:uppercase;margin:0.8rem 0 0.3rem">
        Try an example:
        </p>
        """, unsafe_allow_html=True)

        examples = {
            "Introvert Thinker": "I spend most of my time thinking about complex systems and how things connect. I prefer to analyze problems from every angle before making a decision. Social events drain me and I find much more value in deep one-on-one conversations than large gatherings. Logic and consistency matter deeply to me.",
            "Extrovert Feeler": "I love being around people and feeding off their energy! Every day is an opportunity to connect and make someone smile. I make decisions with my heart first — if something doesn't feel right I won't do it even if it makes logical sense. Life is about relationships and experiences.",
            "Creative Idealist": "My mind is always racing with possibilities and what-ifs. I see patterns and connections others miss. I dream big and believe deeply that the world can be better. Sometimes I struggle with the mundane details of everyday life because my imagination pulls me toward bigger ideas and deeper meanings."
        }

        ex_cols = st.columns(3)
        for i, (label, text) in enumerate(examples.items()):
            with ex_cols[i]:
                if st.button(label, key=f"ex_{i}"):
                    st.session_state['example_text'] = text
                    st.rerun()

        if 'example_text' in st.session_state and not user_text:
            user_text = st.session_state['example_text']

        # Word count
        word_count = len(user_text.split()) if user_text.strip() else 0
        wc_color = '#4a7c6f' if word_count >= 50 else '#c4622d' if word_count > 0 else '#8b9cb0'
        st.markdown(f"""
        <p style="font-family:'DM Mono',monospace;font-size:0.72rem;
        color:{wc_color};text-align:right;margin-top:0.3rem">
        {word_count} words {'✓ Good length' if word_count >= 50 else '— aim for 50+ words' if word_count > 0 else ''}
        </p>
        """, unsafe_allow_html=True)

        predict_btn = st.button("✦ Reveal My Personality Type")

    with col_right:
        st.markdown('<p class="card-label">✦ Your Result</p>', unsafe_allow_html=True)

        if predict_btn and user_text.strip():
            if word_count < 20:
                st.markdown("""
                <div class="warn-box">
                Please write at least 20 words for a meaningful prediction.
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner("Analyzing your writing..."):
                    mbti, dims, confs = predict_personality(user_text)
                    data = MBTI_DATA.get(mbti, {
                        'name': 'Unknown Type',
                        'desc': '',
                        'traits': [],
                        'famous': [],
                        'color': '#c9a84c'
                    })

                # Result hero card
                st.markdown(f"""
                <div class="result-hero">
                    <p style="font-family:'DM Mono',monospace;font-size:0.65rem;
                    letter-spacing:0.25em;color:#8b9cb0;text-transform:uppercase;
                    margin-bottom:0.5rem">Your personality type</p>
                    <div class="result-type">{mbti}</div>
                    <div class="result-name">{data['name']}</div>
                    <div class="result-desc">{data['desc']}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Dimension breakdown
                st.markdown('<p class="card-label">✦ Dimension Breakdown</p>', unsafe_allow_html=True)

                for dim, letter in dims.items():
                    label, desc = DIM_DESCRIPTIONS[dim][letter]
                    conf = confs.get(dim, 70)

                    st.markdown(f"""
                    <div class="dim-row">
                        <span class="dim-letter" style="color:{data['color']}">{letter}</span>
                        <span class="dim-label">{label}</span>
                        <div class="dim-bar-track">
                            <div class="dim-bar-fill" style="width:{conf}%"></div>
                        </div>
                        <span class="dim-value">{conf:.0f}%</span>
                    </div>
                    <p style="font-size:0.72rem;color:#8b9cb0;margin:-0.3rem 0 0.5rem 28px;
                    font-family:'DM Sans',sans-serif">{desc}</p>
                    """, unsafe_allow_html=True)

                # Traits
                st.markdown('<p class="card-label" style="margin-top:1rem">✦ Key Traits</p>', unsafe_allow_html=True)
                traits_html = ''.join([
                    f'<span class="badge badge-gold">{t}</span>'
                    for t in data.get('traits', [])
                ])
                st.markdown(f'<div class="badge-row">{traits_html}</div>', unsafe_allow_html=True)

                # Famous people
                if data.get('famous'):
                    st.markdown('<p class="card-label" style="margin-top:1rem">✦ Famous {}</p>'.format(mbti), unsafe_allow_html=True)
                    famous_html = ''.join([
                        f'<span class="badge">{p}</span>'
                        for p in data['famous']
                    ])
                    st.markdown(f'<div class="badge-row">{famous_html}</div>', unsafe_allow_html=True)

        elif predict_btn and not user_text.strip():
            st.markdown("""
            <div class="warn-box">Please enter some text before predicting.</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                border: 1px dashed #d4cfc4;
                border-radius: 12px;
                padding: 3rem 2rem;
                text-align: center;
                color: #8b9cb0;
            ">
                <p style="font-family:'Playfair Display',serif;font-size:1.8rem;
                font-style:italic;color:#d4cfc4;margin:0">your type awaits</p>
                <p style="font-family:'DM Mono',monospace;font-size:0.7rem;
                letter-spacing:0.15em;margin-top:0.8rem">
                WRITE · SUBMIT · DISCOVER
                </p>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — EXPLORE TYPES
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="card-label">✦ All 16 MBTI Types</p>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:0.85rem;color:#8b9cb0;margin-bottom:1.5rem;font-family:'DM Sans',sans-serif">
    Explore all personality types. Each type is a unique combination of 4 dimensions.
    </p>
    """, unsafe_allow_html=True)

    # Type grid
    groups = {
        'Analysts (NT)':   ['INTJ', 'INTP', 'ENTJ', 'ENTP'],
        'Diplomats (NF)':  ['INFJ', 'INFP', 'ENFJ', 'ENFP'],
        'Sentinels (SJ)':  ['ISTJ', 'ISFJ', 'ESTJ', 'ESFJ'],
        'Explorers (SP)':  ['ISTP', 'ISFP', 'ESTP', 'ESFP']
    }

    group_colors = {
        'Analysts (NT)':  '#2d3561',
        'Diplomats (NF)': '#4a7c6f',
        'Sentinels (SJ)': '#5c6bc0',
        'Explorers (SP)': '#bf360c'
    }

    selected_type = st.selectbox(
        "Select a type to explore:",
        options=list(MBTI_DATA.keys()),
        format_func=lambda x: f"{x} — {MBTI_DATA[x]['name']}"
    )

    if selected_type:
        data = MBTI_DATA[selected_type]
        col1, col2 = st.columns([1, 1.5], gap="large")

        with col1:
            st.markdown(f"""
            <div class="result-hero" style="padding:2rem">
                <div class="result-type" style="font-size:3.5rem">{selected_type}</div>
                <div class="result-name">{data['name']}</div>
                <div class="result-desc" style="margin-top:0.8rem">{data['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="card-label">✦ Key Traits</p>', unsafe_allow_html=True)
            traits_html = ''.join([f'<span class="badge badge-gold">{t}</span>' for t in data['traits']])
            st.markdown(f'<div class="badge-row">{traits_html}</div>', unsafe_allow_html=True)

            st.markdown('<p class="card-label" style="margin-top:1rem">✦ Famous People</p>', unsafe_allow_html=True)
            famous_html = ''.join([f'<span class="badge">{p}</span>' for p in data['famous']])
            st.markdown(f'<div class="badge-row">{famous_html}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<p class="card-label">✦ Dimension Profile</p>', unsafe_allow_html=True)

            for i, (dim, letter) in enumerate(zip(
                ['Mind', 'Energy', 'Nature', 'Tactics'],
                list(selected_type)
            )):
                opposite = {'I': 'E', 'E': 'I', 'N': 'S', 'S': 'N',
                           'T': 'F', 'F': 'T', 'J': 'P', 'P': 'J'}[letter]
                label, desc = DIM_DESCRIPTIONS[dim][letter]
                opp_label = DIM_DESCRIPTIONS[dim][opposite][0]

                st.markdown(f"""
                <div style="margin-bottom:1.2rem">
                    <div style="display:flex;justify-content:space-between;
                    font-family:'DM Mono',monospace;font-size:0.72rem;
                    margin-bottom:0.3rem">
                        <span style="color:{data['color']};font-weight:500">
                            {letter} · {label}
                        </span>
                        <span style="color:#d4cfc4">{opposite} · {opp_label}</span>
                    </div>
                    <div style="height:6px;background:#ede8dc;border-radius:3px;overflow:hidden">
                        <div style="height:100%;width:75%;border-radius:3px;
                        background:linear-gradient(90deg,{data['color']},{data['color']}88)">
                        </div>
                    </div>
                    <p style="font-size:0.75rem;color:#8b9cb0;margin-top:0.3rem;
                    font-family:'DM Sans',sans-serif">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

            # Group badge
            for group, types in groups.items():
                if selected_type in types:
                    color = group_colors[group]
                    st.markdown(f"""
                    <div style="margin-top:1rem;padding:0.8rem 1rem;
                    background:rgba(0,0,0,0.03);border-radius:8px;
                    border-left:3px solid {color}">
                        <p style="font-family:'DM Mono',monospace;font-size:0.7rem;
                        letter-spacing:0.1em;color:{color};text-transform:uppercase;
                        margin:0">{group}</p>
                        <p style="font-size:0.8rem;color:#8b9cb0;margin:0.2rem 0 0;
                        font-family:'DM Sans',sans-serif">
                        Types in this group share {group.split('(')[1].rstrip(')')} preferences
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    break

    st.markdown("<br>", unsafe_allow_html=True)

    # Type distribution chart
    st.markdown('<p class="card-label">✦ Population Distribution</p>', unsafe_allow_html=True)

    # Approximate real world population percentages
    pop_data = {
        'ISFJ': 13.8, 'ESFJ': 12.3, 'ISTJ': 11.6, 'ISFP': 8.8,
        'ESTJ': 8.7,  'ESFP': 8.5,  'ENFP': 8.1,  'ISTP': 5.4,
        'INFP': 4.4,  'ESTP': 4.3,  'INTP': 3.3,  'ENTP': 3.2,
        'ENFJ': 2.5,  'INTJ': 2.1,  'ENTJ': 1.8,  'INFJ': 1.5
    }

    fig, ax = plt.subplots(figsize=(12, 4), facecolor='#f5f0e8')
    ax.set_facecolor('#f5f0e8')

    colors_bar = [MBTI_DATA[t]['color'] for t in pop_data.keys()]
    bars = ax.bar(
        list(pop_data.keys()),
        list(pop_data.values()),
        color=colors_bar,
        edgecolor='#f5f0e8',
        linewidth=1.5,
        alpha=0.85
    )

    # Highlight selected
    if selected_type in pop_data:
        idx = list(pop_data.keys()).index(selected_type)
        bars[idx].set_edgecolor('#c9a84c')
        bars[idx].set_linewidth(2.5)
        bars[idx].set_alpha(1.0)

    ax.tick_params(colors='#8b9cb0', labelsize=8)
    ax.spines['bottom'].set_color('#d4cfc4')
    ax.spines['left'].set_color('#d4cfc4')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("Population %", color='#8b9cb0', fontsize=8)
    ax.set_title("Estimated Global Population by MBTI Type",
                color='#0a0a0f', fontsize=10,
                fontfamily='monospace', pad=10)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<p class="card-label">✦ About This Project</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.87rem;color:#2d3561;line-height:1.7;
        font-family:'DM Sans',sans-serif">
        <p>This project explores whether <strong>personality can be predicted from text</strong>
        using Natural Language Processing and Machine Learning.</p>

        <p>Trained on the <strong>Kaggle MBTI Dataset</strong> — 8,675 users from
        PersonalityCafe forum with their last 50 posts — the model learns linguistic
        patterns associated with each of the 4 MBTI dimensions.</p>

        <p>Rather than treating this as a 16-class problem, we decompose it into
        <strong>4 independent binary classifiers</strong> — one per personality dimension —
        then combine predictions to reconstruct the full type.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="card-label" style="margin-top:1.5rem">✦ ML Pipeline</p>', unsafe_allow_html=True)
        steps = [
            ("1", "Text Preprocessing", "Lowercase, remove URLs, MBTI mentions, stopwords, lemmatize"),
            ("2", "Feature Extraction", "TF-IDF with 5,000 features and bigrams (1,2)"),
            ("3", "Train/Test Split", "80/20 stratified split per dimension"),
            ("4", "Model Training", "4 algorithms compared per dimension"),
            ("5", "Evaluation", "Accuracy, F1 weighted, ROC-AUC, confusion matrix"),
            ("6", "Hyperparameter Tuning", "RandomizedSearchCV on best model"),
            ("7", "Deployment", "Streamlit app with real-time prediction"),
        ]

        for num, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex;gap:0.8rem;margin-bottom:0.8rem;align-items:flex-start">
                <div style="background:#c9a84c;color:white;border-radius:50%;
                width:22px;height:22px;display:flex;align-items:center;
                justify-content:center;font-size:0.65rem;font-family:'DM Mono',monospace;
                flex-shrink:0;margin-top:0.1rem">{num}</div>
                <div>
                    <p style="font-family:'DM Mono',monospace;font-size:0.75rem;
                    color:#0a0a0f;margin:0">{title}</p>
                    <p style="font-size:0.78rem;color:#8b9cb0;margin:0.1rem 0 0;
                    font-family:'DM Sans',sans-serif">{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="card-label">✦ Model Performance</p>', unsafe_allow_html=True)

        if not DEMO_MODE and 'metadata' in assets:
            meta = assets['metadata']
            perf = meta.get('performance', {})

            for dim, metrics in perf.items():
                improvement = metrics['accuracy'] - metrics['baseline']
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                padding:0.6rem 0;border-bottom:1px solid #ede8dc;
                align-items:center">
                    <div>
                        <span style="font-family:'DM Mono',monospace;font-size:0.8rem;
                        color:#0a0a0f">{dim}</span>
                        <span style="font-size:0.72rem;color:#8b9cb0;
                        font-family:'DM Sans',sans-serif;margin-left:0.5rem">
                        {meta['best_models'].get(dim, '')}
                        </span>
                    </div>
                    <div style="text-align:right">
                        <span style="font-family:'DM Mono',monospace;font-size:0.85rem;
                        color:#c9a84c">{metrics['accuracy']:.3f}</span>
                        <span style="font-size:0.7rem;color:#4a7c6f;margin-left:0.4rem">
                        +{improvement:.3f}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:1rem;padding:1rem;background:#ede8dc;
            border-radius:8px;text-align:center">
                <p style="font-family:'Playfair Display',serif;font-size:2rem;
                font-weight:700;color:#c9a84c;margin:0">
                {meta.get('full_type_accuracy', 'N/A')}
                </p>
                <p style="font-family:'DM Mono',monospace;font-size:0.68rem;
                letter-spacing:0.1em;color:#8b9cb0;text-transform:uppercase;margin:0.2rem 0 0">
                Full MBTI Reconstruction Accuracy
                </p>
                <p style="font-size:0.75rem;color:#8b9cb0;margin:0.3rem 0 0;
                font-family:'DM Sans',sans-serif">
                vs 6.25% random baseline for 16 types
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warn-box">
            Load trained model files to display real performance metrics.
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<p class="card-label" style="margin-top:1.5rem">✦ How to Run</p>', unsafe_allow_html=True)
        st.code("""
pip install streamlit pandas numpy scikit-learn
         joblib matplotlib seaborn nltk wordcloud

streamlit run mbti_app.py
        """, language="bash")

        st.markdown('<p class="card-label" style="margin-top:1rem">✦ Built By</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'DM Sans',sans-serif;font-size:0.85rem;color:#8b9cb0">
            <strong style="color:#0a0a0f">Timothy Dzokoto</strong><br>
            Data Science Portfolio Project<br>
            <span style="color:#c9a84c">NLP · Personality Science · Machine Learning</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="card-label" style="margin-top:1rem">✦ Dataset</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.8rem;color:#8b9cb0;font-family:'DM Sans',sans-serif">
        Kaggle MBTI Myers-Briggs Personality Type Dataset<br>
        8,675 users · PersonalityCafe Forum<br>
        <a href="https://www.kaggle.com/datasets/datasnaek/mbti-type"
        style="color:#c9a84c">kaggle.com/datasets/datasnaek/mbti-type</a>
        </div>
        """, unsafe_allow_html=True)