import base64
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error

from src.config import BASE_DIR, MODEL_PATH, RAW_DATA_PATH
from src.data_preprocessor.data_loader import DataLoader
from src.modelling.model_manager import ModelManager

st.set_page_config(page_title="Nairobi Air Quality Predictor", page_icon="🌫️", layout="wide")

TARGET, FEATURE = "P2", "P2.L1"

# US EPA PM2.5 categories (µg/m³): (upper limit, label, colour, advice)
CATEGORIES = [
    (9.0, "Good", "#2ecc71", "Air quality is great. Enjoy the outdoors."),
    (35.4, "Moderate", "#f1c40f", "Acceptable. Very sensitive people may want to limit long outdoor exertion."),
    (55.4, "Unhealthy for Sensitive Groups", "#e67e22", "Children, older adults and people with asthma should reduce outdoor activity."),
    (125.4, "Unhealthy", "#e74c3c", "Everyone should reduce prolonged outdoor exertion."),
    (225.4, "Very Unhealthy", "#8e44ad", "Avoid outdoor activity if possible."),
    (float("inf"), "Hazardous", "#7f1d1d", "Stay indoors and keep windows closed."),
]


def categorize(value):
    for limit, label, colour, advice in CATEGORIES:
        if value <= limit:
            return label, colour, advice


# ---------- Data and model ----------
@st.cache_data
def get_data():
    loader = DataLoader(RAW_DATA_PATH)
    return loader.wrangle(loader.load_data())


@st.cache_resource
def get_model():
    obj = ModelManager(MODEL_PATH).load_model()
    return obj[0] if isinstance(obj, tuple) else obj  # handles the old (model, mae) file


def predict(model, values):
    return model.predict(pd.DataFrame({FEATURE: values}))


def forecast(model, last_value, steps):
    """Recursive forecast: each prediction becomes the next step's input."""
    out, x = [], float(last_value)
    for _ in range(steps):
        x = float(predict(model, [x])[0])
        out.append(x)
    return out


df, model = get_data(), get_model()
latest_time, latest_value = df.index[-1], float(df[TARGET].iloc[-1])
label, colour, advice = categorize(latest_value)

# ---------- Hero banner ----------
def hero_background(colour):
    photo = Path(BASE_DIR) / "assets" / "hero.jpg"
    if photo.exists():
        data = base64.b64encode(photo.read_bytes()).decode()
        return f"url(data:image/jpeg;base64,{data})"
    blocks = [(20, 60, 90), (90, 40, 140), (140, 70, 110), (230, 50, 170), (290, 80, 120),
              (390, 45, 190), (445, 60, 100), (620, 70, 150), (700, 50, 210), (760, 65, 130),
              (840, 55, 175), (900, 80, 105), (990, 60, 160), (1060, 70, 125), (1130, 50, 95)]
    far = "".join(f'<rect x="{x}" y="{330 - h}" width="{w}" height="{h + 70}" fill="#0a1830" opacity="0.55"/>' for x, w, h in blocks)
    near = "".join(f'<rect x="{x + 25}" y="{350 - h * 0.7}" width="{w}" height="{h * 0.7 + 60}" fill="#050d1a"/>' for x, w, h in blocks[::2])
    kicc = ('<rect x="540" y="130" width="34" height="270" fill="#050d1a"/>'
            '<ellipse cx="557" cy="128" rx="28" ry="10" fill="#050d1a"/>'
            '<rect x="556" y="85" width="3" height="45" fill="#050d1a"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 400" preserveAspectRatio="xMidYMid slice">
    <defs>
      <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#0b1d3a"/><stop offset="0.65" stop-color="{colour}" stop-opacity="0.6"/>
        <stop offset="1" stop-color="#0b1d3a"/></linearGradient>
      <linearGradient id="haze" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="{colour}" stop-opacity="0"/><stop offset="1" stop-color="{colour}" stop-opacity="0.45"/></linearGradient>
    </defs>
    <rect width="1200" height="400" fill="url(#sky)"/>
    <circle cx="900" cy="150" r="60" fill="#fff6d6" opacity="0.35"/>
    {far}<rect y="200" width="1200" height="200" fill="url(#haze)"/>{near}{kicc}
    </svg>'''
    return f"url(data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()})"


st.markdown(f"""
<style>
.stApp {{ background: #0b1220; }}
.hero {{ background-image: linear-gradient(180deg, rgba(11,18,32,.1), rgba(11,18,32,.75)), {hero_background(colour)};
        background-size: cover; background-position: center; border-radius: 20px; padding: 56px 40px 40px; margin-bottom: 24px; }}
.hero h1 {{ color:#fff; font-size: 2.8rem; margin:0; }}
.hero p {{ color:#d6e2f5; font-size:1.1rem; margin: 6px 0 22px; }}
.badge {{ display:inline-block; background:{colour}; color:#0b1220; font-weight:700; padding:8px 18px; border-radius:999px; }}
.card {{ background:#131c2e; border:1px solid #223050; border-radius:16px; padding:18px 22px; }}
.card h4 {{ color:#8fa6c9; margin:0 0 6px; font-size:.85rem; font-weight:500; text-transform:uppercase; letter-spacing:.06em; }}
.card .v {{ color:#fff; font-size:1.9rem; font-weight:700; }}
h2, h3, .stTabs button {{ color:#e6efff !important; }}
</style>
<div class="hero">
  <h1>🌫️ Nairobi Air Quality Predictor</h1>
  <p>PM2.5 readings, model predictions and a short-term forecast for Nairobi.</p>
  <span class="badge">{latest_value:.1f} µg/m³ · {label}</span>
  <p style="margin-top:14px">{advice}</p>
</div>
""", unsafe_allow_html=True)


def card(col, title, value):
    col.markdown(f'<div class="card"><h4>{title}</h4><div class="v">{value}</div></div>', unsafe_allow_html=True)


# ---------- Predictions on the full data ----------
pred_all = pd.Series(predict(model, df[FEATURE]), index=df.index)
mae = mean_absolute_error(df[TARGET], pred_all)

c1, c2, c3, c4 = st.columns(4)
card(c1, "Latest reading", f"{latest_value:.1f}")
card(c2, "Average", f"{df[TARGET].mean():.1f}")
card(c3, "Peak", f"{df[TARGET].max():.1f}")
card(c4, "Model error (MAE)", f"±{mae:.1f}")
st.write("")

tab1, tab2, tab3 = st.tabs(["📈 Readings", "🎯 Predictions vs actual", "🔮 Forecast"])

LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              yaxis_title="PM2.5 (µg/m³)", margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
              legend=dict(orientation="h", y=1.1))

with tab1:
    days = st.slider("Days to show", 1, 90, 14, key="d1")
    view = df[TARGET].loc[df.index[-1] - pd.Timedelta(days=days):]
    fig = go.Figure(go.Scatter(x=view.index, y=view, name="PM2.5", line=dict(color="#4ea8ff", width=2),
                               fill="tozeroy", fillcolor="rgba(78,168,255,.12)"))
    fig.add_hline(y=15, line_dash="dash", line_color="#2ecc71", annotation_text="WHO 24h guideline (15)")
    st.plotly_chart(fig.update_layout(**LAYOUT), use_container_width=True)

with tab2:
    days = st.slider("Days to show", 1, 30, 5, key="d2")
    cutoff = df.index[-1] - pd.Timedelta(days=days)
    fig = go.Figure()
    fig.add_scatter(x=df.loc[cutoff:].index, y=df.loc[cutoff:, TARGET], name="Actual", line=dict(color="#4ea8ff", width=2))
    fig.add_scatter(x=pred_all.loc[cutoff:].index, y=pred_all.loc[cutoff:], name="Model prediction",
                    line=dict(color="#ff9f43", width=2, dash="dot"))
    st.plotly_chart(fig.update_layout(**LAYOUT), use_container_width=True)
    st.caption(f"On average the model's one-step predictions are off by about {mae:.1f} µg/m³.")

with tab3:
    steps = st.slider("Hours ahead", 1, 24, 6)
    step = df.index.to_series().diff().median()
    future_idx = [latest_time + step * (i + 1) for i in range(steps)]
    future = forecast(model, latest_value, steps)
    recent = df[TARGET].iloc[-48:]

    fig = go.Figure()
    fig.add_scatter(x=recent.index, y=recent, name="Recent actual", line=dict(color="#4ea8ff", width=2))
    fig.add_scatter(x=[latest_time] + future_idx, y=[latest_value] + future, name="Forecast",
                    line=dict(color="#ff9f43", width=3, dash="dash"), mode="lines+markers")
    fig.add_scatter(x=future_idx + future_idx[::-1], y=[v + mae for v in future] + [v - mae for v in future][::-1],
                    fill="toself", fillcolor="rgba(255,159,67,.15)", line=dict(width=0), name="± typical error", hoverinfo="skip")
    st.plotly_chart(fig.update_layout(**LAYOUT), use_container_width=True)

    end_label, _, _ = categorize(future[-1])
    st.info(f"Expected in {steps} step(s): **{future[-1]:.1f} µg/m³** ({end_label}).")
    st.caption("This model predicts each reading from the one before it, so forecasts drift toward the long-term "
               "average and can't anticipate sudden pollution spikes. Treat it as a rough short-term guide.")