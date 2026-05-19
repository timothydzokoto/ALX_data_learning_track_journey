import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="🌍 Climate Predictor",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

    :root {
        --void: #060810;
        --surface: #0d1117;
        --card: #111827;
        --border: #1f2937;
        --accent: #00ff9d;
        --accent2: #ff6b35;
        --accent3: #3b82f6;
        --text: #e2e8f0;
        --muted: #6b7280;
    }

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
        background-color: var(--void);
        color: var(--text);
    }

    .stApp {
        background: var(--void);
    }

    /* Header */
    .hero {
        background: linear-gradient(135deg, #060810 0%, #0d1117 50%, #0a1628 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(0,255,157,0.06) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00ff9d, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.1;
    }

    .hero-sub {
        color: var(--muted);
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        letter-spacing: 0.05em;
    }

    .badge {
        display: inline-block;
        background: rgba(0,255,157,0.1);
        border: 1px solid rgba(0,255,157,0.3);
        color: var(--accent);
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        margin-right: 0.4rem;
        margin-top: 0.8rem;
    }

    /* Metric cards */
    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent), var(--accent3));
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--accent);
        font-family: 'Space Mono', monospace;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.75rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.3rem;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, rgba(0,255,157,0.05), rgba(59,130,246,0.05));
        border: 1px solid rgba(0,255,157,0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .result-value {
        font-size: 3.5rem;
        font-weight: 800;
        font-family: 'Space Mono', monospace;
        color: var(--accent);
        line-height: 1;
    }

    .result-unit {
        font-size: 1rem;
        color: var(--muted);
        font-family: 'Space Mono', monospace;
        margin-top: 0.5rem;
    }

    .result-label {
        font-size: 0.8rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.5rem;
    }

    /* Warning/info boxes */
    .info-box {
        background: rgba(59,130,246,0.08);
        border: 1px solid rgba(59,130,246,0.25);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #93c5fd;
        font-family: 'Space Mono', monospace;
        margin: 0.5rem 0;
    }

    .warn-box {
        background: rgba(255,107,53,0.08);
        border: 1px solid rgba(255,107,53,0.25);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #fdba74;
        font-family: 'Space Mono', monospace;
        margin: 0.5rem 0;
    }

    /* Section headers */
    .section-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.2em;
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid var(--border);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent), #00cc7a) !important;
        color: var(--void) !important;
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        letter-spacing: 0.05em !important;
        width: 100% !important;
        transition: all 0.2s !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(0,255,157,0.3) !important;
    }

    /* Sliders and inputs */
    .stSlider > div > div > div {
        background: var(--accent) !important;
    }

    /* Divider */
    hr {
        border-color: var(--border) !important;
        margin: 1.5rem 0 !important;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--card);
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
        border: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: var(--muted);
        border-radius: 6px;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(0,255,157,0.1) !important;
        color: var(--accent) !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model Assets ──────────────────────────────────────
@st.cache_resource
def load_assets():
    """Load model, scaler and feature columns from disk."""
    assets = {}
    models_dir = Path("models")

    if not models_dir.exists():
        return None

    # Find model file
    model_files = list(models_dir.glob("tuned_model_*.pkl")) + list(models_dir.glob("best_model_*.pkl"))
    if model_files:
        assets['model'] = joblib.load(model_files[0])
        assets['model_name'] = model_files[0].stem.replace('tuned_model_', '').replace('best_model_', '').replace('_', ' ').title()
    else:
        return None

    # Load scaler
    scaler_path = models_dir / "scaler.pkl"
    if scaler_path.exists():
        assets['scaler'] = joblib.load(scaler_path)
    else:
        return None

    # Load feature columns
    features_path = models_dir / "feature_columns.pkl"
    if features_path.exists():
        assets['features'] = joblib.load(features_path)
    else:
        return None

    return assets


assets = load_assets()

# ── Hero Header ────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">🌍 Climate Predictor</p>
    <p class="hero-sub">TEMPERATURE CHANGE FROM GREENHOUSE GAS EMISSIONS</p>
    <span class="badge">ML POWERED</span>
    <span class="badge">REGRESSION MODEL</span>
    <span class="badge">OWID CO2 DATASET</span>
</div>
""", unsafe_allow_html=True)

# ── Model Status ───────────────────────────────────────────
if assets is None:
    st.markdown("""
    <div class="warn-box">
    ⚠️ MODEL FILES NOT FOUND — Place your model files in a <code>models/</code> folder 
    in the same directory as this app. Required: tuned_model_*.pkl, scaler.pkl, feature_columns.pkl
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    📁 Expected structure:<br>
    &nbsp;&nbsp;models/<br>
    &nbsp;&nbsp;&nbsp;&nbsp;tuned_model_random_forest.pkl<br>
    &nbsp;&nbsp;&nbsp;&nbsp;scaler.pkl<br>
    &nbsp;&nbsp;&nbsp;&nbsp;feature_columns.pkl<br>
    app.py
    </div>
    """, unsafe_allow_html=True)

    st.info("Running in **DEMO MODE** — predictions use a simulated model for demonstration purposes.")
    DEMO_MODE = True
else:
    DEMO_MODE = False
    st.markdown(f"""
    <div class="info-box">
    ✅ Model loaded: <strong>{assets['model_name']}</strong> &nbsp;|&nbsp; 
    Features: <strong>{len(assets['features'])}</strong> &nbsp;|&nbsp;
    Status: <strong>READY</strong>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar — Input Controls ───────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">⚙️ Input Parameters</p>', unsafe_allow_html=True)
    st.markdown("Adjust the sliders to set emission values for prediction.")
    st.markdown("---")

    st.markdown('<p class="section-label">📅 Time</p>', unsafe_allow_html=True)
    year = st.slider("Year", min_value=1900, max_value=2023, value=2020, step=1)
    decade = (year // 10) * 10

    st.markdown("---")
    st.markdown('<p class="section-label">💨 CO2 Emissions</p>', unsafe_allow_html=True)
    co2 = st.slider("Total CO2 (Mt)", 0.0, 12000.0, 500.0, step=10.0)
    co2_per_capita = st.slider("CO2 Per Capita (t)", 0.0, 40.0, 4.5, step=0.1)
    coal_co2 = st.slider("Coal CO2 (Mt)", 0.0, 8000.0, 150.0, step=10.0)
    oil_co2 = st.slider("Oil CO2 (Mt)", 0.0, 4000.0, 120.0, step=10.0)
    gas_co2 = st.slider("Gas CO2 (Mt)", 0.0, 3000.0, 80.0, step=10.0)

    st.markdown("---")
    st.markdown('<p class="section-label">🌿 Other GHG</p>', unsafe_allow_html=True)
    methane = st.slider("Methane (Mt CO2eq)", 0.0, 3000.0, 200.0, step=10.0)
    nitrous_oxide = st.slider("Nitrous Oxide (Mt CO2eq)", 0.0, 1000.0, 80.0, step=5.0)
    ghg_per_capita = st.slider("GHG Per Capita (t CO2eq)", 0.0, 80.0, 10.0, step=0.5)

    st.markdown("---")
    st.markdown('<p class="section-label">🏭 Cumulative Emissions</p>', unsafe_allow_html=True)
    cumulative_co2 = st.slider("Cumulative CO2 (Gt)", 0.0, 500.0, 20.0, step=1.0)

    st.markdown("---")
    st.markdown('<p class="section-label">⚡ Energy</p>', unsafe_allow_html=True)
    energy_per_capita = st.slider("Energy Per Capita (kWh)", 0.0, 100000.0, 30000.0, step=500.0)
    population = st.slider("Population (millions)", 0.1, 1500.0, 50.0, step=1.0)

    st.markdown("---")
    predict_btn = st.button("🔮 PREDICT TEMPERATURE CHANGE")


# ── Build Input DataFrame ──────────────────────────────────
def build_input(features):
    """Build input dataframe matching training features."""
    input_map = {
        'year': year,
        'decade': decade,
        'co2': co2,
        'co2_per_capita': co2_per_capita,
        'coal_co2': coal_co2,
        'oil_co2': oil_co2,
        'gas_co2': gas_co2,
        'methane': methane,
        'nitrous_oxide': nitrous_oxide,
        'ghg_per_capita': ghg_per_capita,
        'cumulative_co2': cumulative_co2,
        'energy_per_capita': energy_per_capita,
        'population': population * 1e6,
        'cement_co2': co2 * 0.04,
        'flaring_co2': co2 * 0.02,
        'land_use_change_co2': co2 * 0.08,
        'co2_growth_abs': co2 * 0.02,
        'co2_growth_prct': 2.0,
        'co2_per_gdp': co2_per_capita / 10,
        'co2_per_unit_energy': co2 / max(energy_per_capita, 1),
        'cumulative_coal_co2': coal_co2 * year * 0.1,
        'cumulative_oil_co2': oil_co2 * year * 0.08,
        'cumulative_gas_co2': gas_co2 * year * 0.06,
        'share_global_co2': min(co2 / 37000 * 100, 100),
        'total_ghg': co2 + methane + nitrous_oxide,
        'total_ghg_excluding_lucf': co2 + methane,
        'ghg_excluding_lucf_per_capita': ghg_per_capita * 0.9,
        'methane_per_capita': methane / max(population, 1),
        'nitrous_oxide_per_capita': nitrous_oxide / max(population, 1),
        'primary_energy_consumption': energy_per_capita * population / 1e6,
        'share_of_temperature_change_from_ghg': min(co2 / 37000 * 100, 100),
    }

    row = {}
    for feat in features:
        if feat in input_map:
            row[feat] = input_map[feat]
        else:
            row[feat] = 0.0

    return pd.DataFrame([row])


# ── Demo prediction ────────────────────────────────────────
def demo_predict():
    """Simulate a prediction for demo mode."""
    base = (co2 / 37000) * 0.5
    year_factor = (year - 1850) / 170 * 0.3
    ghg_factor = (methane + nitrous_oxide) / 10000 * 0.1
    noise = np.random.normal(0, 0.002)
    return round(base + year_factor + ghg_factor + noise, 6)


# ── Risk Level ─────────────────────────────────────────────
def get_risk(val):
    if val < 0.01:
        return "MINIMAL", "#00ff9d", "🟢"
    elif val < 0.05:
        return "LOW", "#86efac", "🟡"
    elif val < 0.15:
        return "MODERATE", "#fbbf24", "🟠"
    elif val < 0.3:
        return "HIGH", "#f97316", "🔴"
    else:
        return "CRITICAL", "#ef4444", "🚨"


# ── Main Content ───────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 PREDICTION", "📊 ANALYSIS", "📖 ABOUT"])

with tab1:
    if predict_btn or True:  # always show result area
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown('<p class="section-label">🎯 Prediction Result</p>', unsafe_allow_html=True)

            if predict_btn:
                with st.spinner("Running model..."):
                    if DEMO_MODE:
                        prediction = demo_predict()
                    else:
                        try:
                            input_df = build_input(assets['features'])
                            input_scaled = assets['scaler'].transform(input_df)
                            prediction = assets['model'].predict(input_scaled)[0]
                        except Exception as e:
                            st.error(f"Prediction error: {e}")
                            prediction = demo_predict()

                    risk_label, risk_color, risk_icon = get_risk(abs(prediction))

                    st.markdown(f"""
                    <div class="result-card">
                        <p class="result-label">PREDICTED TEMPERATURE CHANGE</p>
                        <p class="result-value">{prediction:+.4f}</p>
                        <p class="result-unit">degrees Celsius (°C)</p>
                        <br>
                        <span style="
                            background: rgba(255,255,255,0.05);
                            border: 1px solid {risk_color}40;
                            color: {risk_color};
                            font-family: 'Space Mono', monospace;
                            font-size: 0.75rem;
                            padding: 0.3rem 0.8rem;
                            border-radius: 20px;
                            letter-spacing: 0.1em;
                        ">{risk_icon} RISK LEVEL: {risk_label}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Input summary
                    st.markdown('<p class="section-label" style="margin-top:1rem">📋 Input Summary</p>', unsafe_allow_html=True)
                    summary_data = {
                        "Parameter": ["Year", "Total CO2", "Methane", "Nitrous Oxide", "GHG Per Capita", "Cumulative CO2"],
                        "Value": [
                            str(year),
                            f"{co2:,.1f} Mt",
                            f"{methane:,.1f} Mt",
                            f"{nitrous_oxide:,.1f} Mt",
                            f"{ghg_per_capita:.1f} t",
                            f"{cumulative_co2:.1f} Gt"
                        ]
                    }
                    st.dataframe(
                        pd.DataFrame(summary_data),
                        hide_index=True,
                        use_container_width=True
                    )

            else:
                st.markdown("""
                <div style="
                    background: var(--card);
                    border: 1px dashed var(--border);
                    border-radius: 12px;
                    padding: 3rem;
                    text-align: center;
                    color: var(--muted);
                    font-family: 'Space Mono', monospace;
                    font-size: 0.8rem;
                ">
                    ← SET PARAMETERS IN SIDEBAR<br><br>
                    THEN CLICK PREDICT
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown('<p class="section-label">📊 Emission Breakdown</p>', unsafe_allow_html=True)

            # Donut chart of emission sources
            fig, ax = plt.subplots(figsize=(5, 4), facecolor='#060810')
            ax.set_facecolor('#060810')

            sizes = [coal_co2, oil_co2, gas_co2, methane * 0.3, nitrous_oxide * 0.3]
            labels = ['Coal', 'Oil', 'Gas', 'Methane', 'N₂O']
            colors = ['#ef4444', '#f97316', '#fbbf24', '#00ff9d', '#3b82f6']
            sizes = [max(s, 0.01) for s in sizes]

            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct='%1.0f%%',
                startangle=90,
                wedgeprops=dict(width=0.55, edgecolor='#060810', linewidth=2),
                textprops={'color': '#e2e8f0', 'fontsize': 9, 'fontfamily': 'monospace'}
            )

            for at in autotexts:
                at.set_color('#060810')
                at.set_fontsize(8)
                at.set_fontweight('bold')

            ax.set_title("Emission Sources", color='#6b7280', fontsize=9,
                        fontfamily='monospace', pad=10)

            centre_circle = plt.Circle((0, 0), 0.45, fc='#060810')
            ax.add_patch(centre_circle)
            ax.text(0, 0, f'{co2:.0f}\nMt CO₂', ha='center', va='center',
                   color='#00ff9d', fontsize=8, fontfamily='monospace', fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Risk gauge
            st.markdown('<p class="section-label" style="margin-top:1rem">🌡️ Emission Level</p>', unsafe_allow_html=True)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size:1.3rem">{co2_per_capita:.1f}</div>
                    <div class="metric-label">t CO₂/person</div>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size:1.3rem">{ghg_per_capita:.1f}</div>
                    <div class="metric-label">t GHG/person</div>
                </div>
                """, unsafe_allow_html=True)
            with col_c:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size:1.3rem">{decade}</div>
                    <div class="metric-label">Decade</div>
                </div>
                """, unsafe_allow_html=True)


with tab2:
    st.markdown('<p class="section-label">📈 Emission Trends Simulator</p>', unsafe_allow_html=True)
    st.markdown("See how emissions grow over time based on your current input values.")

    col1, col2 = st.columns(2)

    with col1:
        # CO2 trend over decades
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#060810')
        ax.set_facecolor('#111827')

        decades = list(range(1900, 2031, 10))
        growth_factor = [max(0, (d - 1850) / 170) for d in decades]
        simulated_co2 = [co2 * g for g in growth_factor]

        ax.fill_between(decades, simulated_co2, alpha=0.2, color='#00ff9d')
        ax.plot(decades, simulated_co2, color='#00ff9d', linewidth=2)
        ax.axvline(x=year, color='#ff6b35', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Year {year}')

        ax.set_facecolor('#111827')
        ax.tick_params(colors='#6b7280', labelsize=8)
        ax.spines['bottom'].set_color('#1f2937')
        ax.spines['left'].set_color('#1f2937')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title("Simulated CO₂ Trend", color='#e2e8f0', fontsize=10, fontfamily='monospace')
        ax.set_xlabel("Year", color='#6b7280', fontsize=8)
        ax.set_ylabel("CO₂ (Mt)", color='#6b7280', fontsize=8)
        ax.legend(fontsize=8, facecolor='#111827', edgecolor='#1f2937', labelcolor='#e2e8f0')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        # GHG composition bar
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#060810')
        ax.set_facecolor('#111827')

        categories = ['CO₂', 'Methane', 'N₂O', 'Other']
        values = [co2, methane, nitrous_oxide, co2 * 0.05]
        colors = ['#3b82f6', '#00ff9d', '#fbbf24', '#6b7280']

        bars = ax.bar(categories, values, color=colors, edgecolor='#060810', linewidth=1.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   f'{val:.0f}', ha='center', va='bottom',
                   color='#e2e8f0', fontsize=8, fontfamily='monospace')

        ax.set_facecolor('#111827')
        ax.tick_params(colors='#6b7280', labelsize=8)
        ax.spines['bottom'].set_color('#1f2937')
        ax.spines['left'].set_color('#1f2937')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title("GHG Composition (Mt)", color='#e2e8f0', fontsize=10, fontfamily='monospace')
        ax.set_ylabel("Emissions (Mt)", color='#6b7280', fontsize=8)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown('<p class="section-label">🌍 Global Context</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    global_co2_2020 = 37000

    with col1:
        share = min(co2 / global_co2_2020 * 100, 100)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{share:.2f}%</div>
            <div class="metric-label">Global CO₂ Share</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        world_avg_co2_pc = 4.7
        compare = co2_per_capita / world_avg_co2_pc
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{compare:.1f}x</div>
            <div class="metric-label">vs World Average</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{year}</div>
            <div class="metric-label">Selected Year</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        total_ghg = co2 + methane + nitrous_oxide
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_ghg:,.0f}</div>
            <div class="metric-label">Total GHG (Mt)</div>
        </div>
        """, unsafe_allow_html=True)


with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-label">📖 About This App</p>', unsafe_allow_html=True)
        st.markdown("""
        This app uses a machine learning regression model trained on the
        **Our World in Data CO2 and Greenhouse Gas Emissions** dataset to predict
        how much a country's emissions contribute to global temperature change.

        **Dataset:** 50,411 rows × 79 columns spanning 1750–2022

        **Target Variable:** `temperature_change_from_ghg` — the estimated
        global mean surface temperature change attributable to a country's
        cumulative greenhouse gas emissions.
        """)

        st.markdown('<p class="section-label" style="margin-top:1.5rem">🔄 Pipeline Steps</p>', unsafe_allow_html=True)
        steps = [
            "1. Data Loading & EDA",
            "2. Missing Value Handling",
            "3. Feature Selection",
            "4. Feature Engineering",
            "5. Train / Test Split (80/20)",
            "6. StandardScaler Normalization",
            "7. Model Training (5 algorithms)",
            "8. Evaluation (R², MAE, RMSE)",
            "9. Hyperparameter Tuning",
            "10. Model Serialization (joblib)"
        ]
        for step in steps:
            st.markdown(f"<p style='font-family:monospace;font-size:0.8rem;color:#6b7280;margin:0.2rem 0'>{step}</p>",
                       unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-label">📦 Model Info</p>', unsafe_allow_html=True)

        if not DEMO_MODE:
            model_info = {
                "Model": assets['model_name'],
                "Features": len(assets['features']),
                "Status": "Production Ready"
            }
            for k, v in model_info.items():
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:0.4rem 0;
                border-bottom:1px solid var(--border);font-size:0.8rem;">
                    <span style="color:var(--muted);font-family:monospace">{k}</span>
                    <span style="color:var(--accent);font-family:monospace">{v}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warn-box">
            Running in DEMO MODE.<br>
            Load your trained model files to enable real predictions.
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<p class="section-label" style="margin-top:1.5rem">🚀 How to Run</p>', unsafe_allow_html=True)
        st.code("""
# Install dependencies
pip install streamlit pandas numpy 
    scikit-learn joblib matplotlib seaborn

# Run the app
streamlit run app.py
        """, language="bash")

        st.markdown('<p class="section-label" style="margin-top:1rem">👤 Built By</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:monospace;font-size:0.8rem;color:var(--muted)">
        Data Science Portfolio Project<br>
        <span style="color:var(--accent)">GES ICT Teacher → Data Scientist</span><br>
        Built with Python · Scikit-learn · Streamlit
        </div>
        """, unsafe_allow_html=True)