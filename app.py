#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
APP.PY - DiabetesSense: Aplikasi Web ML untuk Deteksi Diabetes
=============================================================================
Mata Kuliah  : Pembelajaran Mesin
Semester     : Genap 2025/2026
Universitas  : Universitas Dian Nuswantoro (UDINUS) Semarang

Kelompok:
  - Fahmi Fatmawati Azzahra  (NIM: A11.2024.15831)
  - Nathaniela Febry Nathasa (NIM: A11.2024.15850)

Cara Menjalankan:
  1. Jalankan training dulu: python train.py
  2. Jalankan app       : streamlit run app.py
=============================================================================
"""

import os
import sys
import json
import warnings
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# GLOBAL PLOTLY TEMPLATE — teks gelap untuk background putih
# ─────────────────────────────────────────────────────────────
_FONT_DARK = dict(family="Inter, sans-serif", color="#111111", size=12)
pio.templates["light_app"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_FONT_DARK,
        title=dict(font=dict(color="#111111", size=14, family="Inter, sans-serif")),
        xaxis=dict(
            color="#111111", gridcolor="#e5e7eb", linecolor="#bbbbbb",
            tickfont=dict(color="#111111"),
            title=dict(font=dict(color="#111111")),
        ),
        yaxis=dict(
            color="#111111", gridcolor="#e5e7eb", linecolor="#bbbbbb",
            tickfont=dict(color="#111111"),
            title=dict(font=dict(color="#111111")),
        ),
        legend=dict(
            font=dict(color="#111111"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#dddddd",
        ),
        coloraxis=dict(colorbar=dict(
            tickfont=dict(color="#111111"),
            title=dict(font=dict(color="#111111")),
        )),
        annotationdefaults=dict(font=dict(color="#111111")),
    )
)
pio.templates.default = "light_app"




# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title   = "DiabetesSense – ML Classifier",
    page_icon    = "🩺",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)
st.markdown("""
<style>
/* ══════════════════════════════════════════════════════════════
   FONTS
══════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, * { font-family: 'Inter', sans-serif !important; }

/* ══════════════════════════════════════════════════════════════
   APP BACKGROUND
══════════════════════════════════════════════════════════════ */
.stApp { background: linear-gradient(140deg, #eef2f7 0%, #e3eaf5 100%) !important; }

/* ══════════════════════════════════════════════════════════════
   SIDEBAR — sembunyikan tombol toggle (sidebar selalu tampil)
══════════════════════════════════════════════════════════════ */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* Sidebar background */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1246 45%, #0d2b6b 100%) !important;
}
/* Sidebar text — putih */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong { color: white !important; }
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stRadio label p,
[data-testid="stSidebar"] .stRadio span { color: rgba(255,255,255,0.9) !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2) !important; }

/* ══════════════════════════════════════════════════════════════
   MAIN CONTENT — SEMUA TEKS HITAM/GELAP
   Menggunakan semua selector Streamlit yang dikenal
══════════════════════════════════════════════════════════════ */

/* Container utama */
[data-testid="stMain"] { color: #111111 !important; }
[data-testid="stMainBlockContainer"] { color: #111111 !important; }
.main .block-container { color: #111111 !important; }

/* Markdown headings (st.markdown("### ...")) */
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h4,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h5,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h6 { color: #1a1a2e !important; }

[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] span { color: #374151 !important; }

/* Heading langsung (st.header, st.subheader) */
[data-testid="stHeadingWithActionElements"] *,
[data-testid="stHeading"] * { color: #1a1a2e !important; }

/* st.metric */
[data-testid="stMetricValue"]  { color: #1a1a2e !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"]  { color: #6B7280 !important; }
[data-testid="stMetricDelta"]  { color: #27AE60 !important; }

/* Widget labels */
[data-testid="stWidgetLabel"]  { color: #1a1a2e !important; }
[data-testid="stWidgetLabel"] p { color: #1a1a2e !important; }
.stSlider > label, .stSelectbox > label,
.stNumberInput > label, .stCheckbox > label { color: #1a1a2e !important; }

/* Radio di main (bukan sidebar) */
[data-testid="stMain"] .stRadio label,
[data-testid="stMain"] .stRadio label p { color: #1a1a2e !important; }

/* Dataframe / tabel */
.stDataFrame td, .stDataFrame th { color: #111111 !important; }

/* Tab buttons */
[data-testid="stTabs"] [role="tab"] { color: #1a1a2e !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #2E86AB !important; }
.stTabs [data-baseweb="tab"] { color: #1a1a2e !important; }

/* Code blocks */
[data-testid="stCode"] * { color: #111111 !important; }

/* ══════════════════════════════════════════════════════════════
   COMPONENT STYLES — elemen dengan background gelap tetap putih
══════════════════════════════════════════════════════════════ */

/* Hero card (background navy → teks putih) */
.hero-card {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1246 50%, #0d2b6b 100%);
    padding: 2.4rem 2rem; border-radius: 18px;
    text-align: center; margin-bottom: 2rem;
    box-shadow: 0 12px 40px rgba(10,15,70,0.3);
}
.hero-card, .hero-card *,
.hero-card h1, .hero-card h2, .hero-card h3,
.hero-card p, .hero-card span {
    color: white !important;
    -webkit-text-fill-color: white !important;
}
.hero-card h1 { font-size: 2.6rem; font-weight: 800; margin: 0; }
.hero-card p  { font-size: 1.05rem; opacity: 0.82; margin: 0.4rem 0 0; }

/* Metric card */
.metric-card {
    background: white; padding: 1.4rem 1.2rem; border-radius: 14px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.08); border-left: 5px solid #2E86AB;
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 10px 28px rgba(0,0,0,0.14); }
.metric-card h3 { font-size: 2rem; font-weight: 800; color: #2E86AB !important; margin: 0; }
.metric-card p  { color: #6B7280 !important; margin: .2rem 0 0; font-size: .85rem; }

/* Team card */
.team-card {
    background: white; padding: 1.4rem 1rem; border-radius: 14px;
    text-align: center; box-shadow: 0 4px 18px rgba(0,0,0,0.08); transition: transform .2s;
}
.team-card:hover { transform: translateY(-4px); }

/* Section header */
.section-header {
    font-size: 1.35rem; font-weight: 700; color: #1a1a2e !important;
    padding-bottom: .45rem; border-bottom: 3px solid #2E86AB; margin-bottom: 1.4rem;
}

/* Prediction cards (background merah/hijau → teks putih) */
.pred-positive {
    background: linear-gradient(135deg, #e53935 0%, #b71c1c 100%);
    padding: 2.2rem; border-radius: 18px; text-align: center;
    box-shadow: 0 10px 35px rgba(229,57,53,.35);
    animation: pulse .9s ease-in-out infinite alternate;
}
.pred-negative {
    background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
    padding: 2.2rem; border-radius: 18px; text-align: center;
    box-shadow: 0 10px 35px rgba(46,125,50,.35);
}
.pred-positive *, .pred-positive h2, .pred-positive p, .pred-positive span,
.pred-negative *, .pred-negative h2, .pred-negative p, .pred-negative span {
    color: white !important;
    -webkit-text-fill-color: white !important;
}
@keyframes pulse {
    from { box-shadow: 0 10px 35px rgba(229,57,53,.35); }
    to   { box-shadow: 0 10px 55px rgba(229,57,53,.60); }
}

/* Insight box */
.insight-box {
    background: white; padding: 1.25rem; border-radius: 12px;
    border-left: 5px solid #F18F01; box-shadow: 0 3px 12px rgba(0,0,0,.07);
    margin-bottom: .9rem;
}
.insight-box h4 { color: #F18F01 !important; margin: 0 0 .4rem; font-size: .95rem; font-weight: 700; }
.insight-box p  { color: #374151 !important; margin: 0; font-size: .88rem; line-height: 1.65; }

/* Step circle */
.step-circle {
    width: 38px; height: 38px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 1rem; color: white !important; flex-shrink: 0;
}

/* ── Code blocks — background terang, teks hitam ── */
[data-testid="stCode"],
[data-testid="stCode"] pre,
[data-testid="stCode"] code,
[data-testid="stCodeBlock"],
[data-testid="stCodeBlock"] pre,
[data-testid="stCodeBlock"] code,
[class*="CodeBlock"],
[class*="stCodeBlock"] {
    background: #f4f6fa !important;
    color: #111111 !important;
    border: 1px solid #d1d5db !important;
}

/* ── Hide Streamlit default chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────

# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / 'data'
MODEL_DIR   = BASE_DIR / 'models'
FIGURES_DIR = BASE_DIR / 'reports' / 'figures'

COLUMNS = [
    'Pregnancies','Glucose','BloodPressure','SkinThickness',
    'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'
]
ZERO_INVALID = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

COLORS = {
    'primary'  : '#2E86AB',
    'secondary': '#A23B72',
    'accent'   : '#F18F01',
    'danger'   : '#C73E1D',
    'success'  : '#27AE60',
    'neutral'  : '#6B7280',
}
MODEL_COLORS = {
    'Logistic Regression': '#2E86AB',
    'Random Forest'      : '#A23B72',
    'Gradient Boosting'  : '#F18F01',
}

# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# CHART HELPER — paksa semua teks chart hitam #111111
# ─────────────────────────────────────────────────────────────
def show_chart(fig, **kwargs):
    """Render Plotly chart dengan semua teks hitam di background putih."""
    BLACK = "#111111"
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=BLACK, family="Inter, sans-serif", size=12),
        title_font=dict(color=BLACK, family="Inter, sans-serif", size=14),
        legend=dict(
            font=dict(color=BLACK),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cccccc",
        ),
    )
    # Paksa semua sumbu (termasuk subplot) hitam
    fig.update_xaxes(
        tickfont=dict(color=BLACK, size=11),
        title_font=dict(color=BLACK, size=12),
        color=BLACK,
        linecolor="#aaaaaa",
        gridcolor="#eeeeee",
    )
    fig.update_yaxes(
        tickfont=dict(color=BLACK, size=11),
        title_font=dict(color=BLACK, size=12),
        color=BLACK,
        linecolor="#aaaaaa",
        gridcolor="#eeeeee",
    )
    # Paksa colorbar hitam (heatmap / scatter color)
    for trace in fig.data:
        if hasattr(trace, 'colorbar') and trace.colorbar is not None:
            trace.update(colorbar=dict(
                tickfont=dict(color=BLACK),
                title=dict(font=dict(color=BLACK)),
            ))
    # Subplot titles (annotation)
    for ann in fig.layout.annotations:
        ann.font.color = BLACK
    st.plotly_chart(fig, **kwargs)


# ─────────────────────────────────────────────────────────────
# DATA & MODEL LOADERS (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw_data():
    path = DATA_DIR / 'diabetes.csv'
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        url = ("https://raw.githubusercontent.com/jbrownlee/"
               "Datasets/master/pima-indians-diabetes.data.csv")
        urllib.request.urlretrieve(url, path)
        df = pd.read_csv(path, header=None, names=COLUMNS)
        df.to_csv(path, index=False)
    else:
        try:
            df = pd.read_csv(path)
            if df.shape[1] == 9 and list(df.columns) == COLUMNS:
                pass
            else:
                df = pd.read_csv(path, header=None, names=COLUMNS)
        except Exception:
            df = pd.read_csv(path, header=None, names=COLUMNS)
    return df

@st.cache_data(show_spinner=False)
def load_results():
    p = MODEL_DIR / 'results.json'
    return json.loads(p.read_text()) if p.exists() else None

@st.cache_data(show_spinner=False)
def load_feature_importances():
    p = MODEL_DIR / 'feature_importances.json'
    return json.loads(p.read_text()) if p.exists() else None

@st.cache_resource(show_spinner=False)
def load_models():
    out = {}
    for name, fname in [
        ('Logistic Regression', 'logistic_regression.pkl'),
        ('Random Forest',       'random_forest.pkl'),
        ('Gradient Boosting',   'gradient_boosting.pkl'),
        ('Best Model',          'best_model.pkl'),
    ]:
        p = MODEL_DIR / fname
        if p.exists():
            out[name] = joblib.load(p)
    return out

@st.cache_resource(show_spinner=False)
def load_scaler():
    p = MODEL_DIR / 'scaler.pkl'
    return joblib.load(p) if p.exists() else None

def models_ready():
    return (MODEL_DIR / 'best_model.pkl').exists() and \
           (MODEL_DIR / 'results.json').exists()

# ─────────────────────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────────────────────
def build_input(pregnancies, glucose, bp, skin, insulin, bmi, dpf, age,
                scaler, feature_names):
    """Buat array input untuk inferensi model."""
    d = {
        'Pregnancies'            : pregnancies,
        'Glucose'                : glucose,
        'BloodPressure'          : bp,
        'SkinThickness'          : skin,
        'Insulin'                : insulin,
        'BMI'                    : bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age'                    : age,
        # Feature engineering (harus sama dengan train.py)
        'BMI_Category'           : (0 if bmi < 18.5 else 1 if bmi < 25 else
                                    2 if bmi < 30 else 3),
        'Glucose_Category'       : (0 if glucose < 100 else 1 if glucose < 126 else 2),
        'Age_Group'              : (0 if age < 35 else 1 if age < 50 else 2),
        'Insulin_Glucose_Ratio'  : insulin / (glucose + 1),
    }
    df_in = pd.DataFrame([d])
    if feature_names:
        df_in = df_in[feature_names]
    return scaler.transform(df_in)

# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1.2rem 0 .5rem;">
          <div style="font-size:3.2rem;">🩺</div>
          <h2 style="color:white;margin:0;font-size:1.4rem;font-weight:800;">DiabetesSense</h2>
          <p style="color:rgba(255,255,255,.6);font-size:.78rem;margin:.2rem 0 0;">ML Classifier · UDINUS 2026</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        page = st.radio(
            "Navigasi",
            ["🏠 Beranda",
             "📊 EDA Dashboard",
             "🤖 Prediksi Diabetes",
             "📈 Evaluasi Model",
             "💡 Interpretasi & Bisnis",
             "📚 Dokumentasi"],
            label_visibility="collapsed"
        )
        st.markdown("---")

        # Status panel
        if models_ready():
            res = load_results()
            if res:
                bm   = res.get('best_model','–')
                bf1  = res['results'][bm]['test']['f1']
                bauc = res['results'][bm]['test']['roc_auc']
                st.markdown(f"""
                <div style="background:rgba(255,255,255,.1);padding:1rem;border-radius:10px;">
                  <p style="color:rgba(255,255,255,.65);font-size:.72rem;margin:0 0 .4rem;">⭐ Model Terbaik</p>
                  <p style="color:white;font-weight:700;font-size:.88rem;margin:0;">{bm}</p>
                  <p style="color:#90caf9;font-size:.78rem;margin:.2rem 0 0;">
                    F1 = {bf1:.4f} &nbsp;|&nbsp; AUC = {bauc:.4f}
                  </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Model belum dilatih.\nJalankan `python train.py`")

        st.markdown("---")
        st.markdown("""
        <div style="text-align:center;padding:.4rem 0;">
          <p style="color:rgba(255,255,255,.45);font-size:.69rem;margin:0;line-height:1.7;">
            Dibuat oleh:<br>
            <strong style="color:rgba(255,255,255,.75);">Fahmi & Nathaniela</strong><br>
            Teknik Informatika · UDINUS<br>
            Semester Genap 2025/2026
          </p>
        </div>
        """, unsafe_allow_html=True)
    return page

# ═══════════════════════════════════════════════════════════════
# PAGE 1 – BERANDA
# ═══════════════════════════════════════════════════════════════
def page_beranda():
    st.markdown("""
    <div class="hero-card">
      <h1>🩺 DiabetesSense</h1>
      <p>Sistem Prediksi Penyakit Diabetes Berbasis Machine Learning</p>
      <p style="opacity:.6;font-size:.88rem;">
        UAS Pembelajaran Mesin &nbsp;·&nbsp; Semester Genap 2025/2026 &nbsp;·&nbsp; UDINUS Semarang
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tim ──
    st.markdown('<div class="section-header">👥 Tim Pengembang</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="team-card">
          <div style="font-size:3.2rem;margin-bottom:.6rem;">👩‍🎓</div>
          <h3 style="color:#1a1a2e;font-size:1.05rem;margin:0 0 .25rem;">Fahmi Fatmawati Azzahra</h3>
          <p style="color:#2E86AB;font-weight:700;margin:0;font-size:.95rem;">NIM: A11.2024.15831</p>
          <p style="color:#6B7280;font-size:.82rem;margin:.2rem 0 0;">Teknik Informatika · UDINUS Semarang</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="team-card">
          <div style="font-size:3.2rem;margin-bottom:.6rem;">👩‍🎓</div>
          <h3 style="color:#1a1a2e;font-size:1.05rem;margin:0 0 .25rem;">Nathaniela Febry Nathasa</h3>
          <p style="color:#A23B72;font-weight:700;margin:0;font-size:.95rem;">NIM: A11.2024.15850</p>
          <p style="color:#6B7280;font-size:.82rem;margin:.2rem 0 0;">Teknik Informatika · UDINUS Semarang</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stats cards ──
    df = load_raw_data()
    n0 = (df['Outcome']==0).sum()
    n1 = (df['Outcome']==1).sum()

    st.markdown('<div class="section-header">📊 Statistik Dataset</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    data_cards = [
        (df.shape[0], "Total Sampel",       "#2E86AB"),
        (df.shape[1]-1,"Jumlah Fitur",      "#A23B72"),
        (n1,            "Kasus Diabetes",   "#C73E1D"),
        (n0,            "Non-Diabetes",     "#27AE60"),
    ]
    for col, (val, lbl, clr) in zip(cols, data_cards):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{clr};">
              <h3 style="color:{clr};">{val}</h3>
              <p>{lbl}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tentang proyek ──
    st.markdown('<div class="section-header">🎯 Tentang Proyek</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
        <div style="background:white;padding:1.5rem;border-radius:14px;box-shadow:0 4px 18px rgba(0,0,0,.08);">
          <h4 style="color:#1a1a2e;margin-top:0;">📝 Latar Belakang Masalah</h4>
          <p style="color:#374151;line-height:1.82;font-size:.93rem;text-align:justify;">
            Diabetes mellitus merupakan salah satu penyakit kronis dengan prevalensi tertinggi di dunia.
            Berdasarkan data <em>International Diabetes Federation</em> (IDF) edisi 2021, terdapat sekitar
            <strong>537 juta</strong> orang dewasa yang hidup dengan diabetes, dan angka ini diproyeksikan
            meningkat hingga <strong>783 juta</strong> pada 2045. Di Indonesia, diabetes masuk dalam
            10 besar penyebab kematian tertinggi berdasarkan data Kementerian Kesehatan RI.
          </p>
          <p style="color:#374151;line-height:1.82;font-size:.93rem;text-align:justify;">
            Deteksi dini sangat krusial untuk mencegah komplikasi serius seperti penyakit kardiovaskular,
            gagal ginjal, neuropati, dan retinopati diabetik. Namun, banyak penderita tidak menyadari
            kondisinya karena gejala tahap awal yang tidak spesifik dan minimnya akses ke fasilitas
            pemeriksaan kesehatan — terutama di daerah terpencil.
          </p>
          <p style="color:#374151;line-height:1.82;font-size:.93rem;text-align:justify;">
            Proyek ini membangun sistem prediksi berbasis <em>Machine Learning</em> yang mampu
            mengklasifikasikan risiko diabetes berdasarkan indikator kesehatan dasar (kadar glukosa,
            BMI, tekanan darah, usia, dll.) menggunakan <strong>Pima Indians Diabetes Dataset</strong>
            dari National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), yang terdiri
            dari 768 rekam medis pasien perempuan berusia ≥21 tahun keturunan Pima Indian.
          </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background:white;padding:1.4rem;border-radius:14px;box-shadow:0 4px 18px rgba(0,0,0,.08);margin-bottom:1rem;">
          <h4 style="color:#1a1a2e;margin-top:0;">🎯 Tujuan & Metrik Sukses</h4>
          <ul style="color:#374151;font-size:.88rem;line-height:2;padding-left:1.2rem;">
            <li>Membangun model prediksi diabetes akurasi tinggi</li>
            <li>Mengidentifikasi faktor risiko utama</li>
            <li>Mendukung skrining awal oleh tenaga medis</li>
            <li>Memberikan interpretasi yang dapat dipahami non-teknis</li>
          </ul>
          <hr style="border-color:#e5e7eb;margin:.8rem 0;">
          <h5 style="color:#1a1a2e;margin:.5rem 0 .4rem;">📏 Target Metrik</h5>
          <ul style="color:#374151;font-size:.88rem;line-height:2;padding-left:1.2rem;">
            <li><strong style="color:#2E86AB;">Accuracy ≥ 75%</strong></li>
            <li><strong style="color:#A23B72;">F1-Score ≥ 0.70</strong></li>
            <li><strong style="color:#F18F01;">AUC-ROC ≥ 0.80</strong></li>
            <li><strong style="color:#C73E1D;">Recall ≥ 0.75</strong> (min. false negative)</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    # ── Alur kerja ──
    st.markdown('<div class="section-header">🔄 Pipeline Machine Learning</div>', unsafe_allow_html=True)
    steps = [
        ("1","#2E86AB","📥 Akuisisi Data","Pima Indians Dataset\n768 sampel · 8 fitur"),
        ("2","#A23B72","🔍 EDA","Distribusi · Korelasi\nOutlier · Missing Values"),
        ("3","#F18F01","⚙️ Preprocessing","Imputasi · Scaling\nFeature Engineering · Split"),
        ("4","#C73E1D","🤖 Pemodelan","LR · RF · GB\nGridSearchCV Tuning"),
        ("5","#27AE60","📊 Evaluasi","Accuracy · F1 · AUC\nConfusion Matrix · ROC"),
        ("6","#6B7280","🚀 Deployment","Streamlit App\nInteraktif & Real-time"),
    ]
    cols = st.columns(6)
    for col, (num, clr, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style="background:white;padding:1rem .7rem;border-radius:12px;text-align:center;
                        box-shadow:0 4px 15px rgba(0,0,0,.07);min-height:135px;">
              <div class="step-circle" style="background:{clr};margin-bottom:.5rem;">{num}</div>
              <h4 style="color:{clr};font-size:.82rem;margin:.2rem 0 .25rem;line-height:1.3;">{title}</h4>
              <p style="color:#6B7280;font-size:.74rem;margin:0;line-height:1.45;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 2 – EDA DASHBOARD
# ═══════════════════════════════════════════════════════════════
def page_eda():
    st.markdown("""
    <div class="hero-card" style="padding:1.4rem 2rem;">
      <h1 style="font-size:1.9rem;">📊 EDA Dashboard</h1>
      <p style="margin:0;">Exploratory Data Analysis – Pima Indians Diabetes Dataset</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_raw_data()
    n0 = (df['Outcome']==0).sum()
    n1 = (df['Outcome']==1).sum()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Overview",
        "📈 Distribusi & Univariat",
        "🔗 Korelasi & Multivariat",
        "🎯 Kualitas Data",
        "💡 5 Key Insights",
    ])

    # ── Tab 1: Overview ──────────────────────────────────────
    with tab1:
        st.markdown("### 📋 Ringkasan Dataset")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Baris",   df.shape[0])
            st.metric("Total Kolom",   df.shape[1])
        with c2:
            st.metric("Diabetes (1)",  n1, f"{n1/len(df)*100:.1f}%")
            st.metric("Non-DM (0)",    n0, f"{n0/len(df)*100:.1f}%")
        with c3:
            st.metric("NaN Eksplisit", df.isnull().sum().sum())
            st.metric("Duplikat",      df.duplicated().sum())

        n_show = st.slider("Jumlah baris preview:", 5, 50, 10)
        st.markdown("#### Preview Data")
        st.dataframe(df.head(n_show), use_container_width=True)

        st.markdown("#### Statistik Deskriptif")
        st.dataframe(df.describe().round(3), use_container_width=True)

        st.markdown("#### Informasi Tipe Data & Nilai 0")
        info_rows = []
        zero_invalid_set = set(ZERO_INVALID)
        for col in df.columns:
            info_rows.append({
                "Kolom"       : col,
                "Dtype"       : str(df[col].dtype),
                "Non-Null"    : df[col].notna().sum(),
                "Nilai 0"     : (df[col] == 0).sum(),
                "0 Tdk Valid?": "⚠️ Ya" if col in zero_invalid_set else "—",
                "Min"         : round(df[col].min(), 2),
                "Max"         : round(df[col].max(), 2),
                "Mean"        : round(df[col].mean(), 3),
                "Std"         : round(df[col].std(), 3),
            })
        st.dataframe(pd.DataFrame(info_rows), use_container_width=True, hide_index=True)

    # ── Tab 2: Distribusi ────────────────────────────────────
    with tab2:
        st.markdown("### Distribusi Kelas Target")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure([go.Bar(
                x=["Non-Diabetes (0)", "Diabetes (1)"],
                y=[n0, n1],
                marker_color=[COLORS['primary'], COLORS['danger']],
                text=[f"{n0} ({n0/len(df)*100:.1f}%)", f"{n1} ({n1/len(df)*100:.1f}%)"],
                textposition="outside", textfont=dict(size=12),
            )])
            fig.update_layout(title="Bar Chart Distribusi Kelas",
                              yaxis_title="Jumlah Sampel", height=340,
                              plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
            show_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure([go.Pie(
                labels=["Non-Diabetes","Diabetes"],
                values=[n0, n1], hole=0.42,
                marker_colors=[COLORS['primary'], COLORS['danger']],
                textinfo="label+percent", textfont=dict(size=12),
            )])
            fig.update_layout(title="Pie Chart Proporsi Kelas", height=340,
                              plot_bgcolor="white", paper_bgcolor="white")
            show_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Distribusi Fitur")

        feat_list  = [c for c in df.columns if c != 'Outcome']
        sel_feat   = st.selectbox("Pilih fitur:", feat_list, index=1)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for ov, lbl, clr in [(0,"Non-Diabetes",COLORS['primary']),
                                  (1,"Diabetes",    COLORS['danger'])]:
                fig.add_trace(go.Histogram(
                    x=df[df['Outcome']==ov][sel_feat], name=lbl,
                    opacity=0.68, marker_color=clr, nbinsx=24,
                ))
            fig.update_layout(
                barmode="overlay", title=f"Histogram {sel_feat}",
                xaxis_title=sel_feat, yaxis_title="Frekuensi",
                height=350, plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(x=0.68, y=0.97),
            )
            show_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            for ov, lbl, clr in [(0,"Non-Diabetes",COLORS['primary']),
                                  (1,"Diabetes",    COLORS['danger'])]:
                fig.add_trace(go.Box(
                    y=df[df['Outcome']==ov][sel_feat],
                    name=lbl, marker_color=clr, boxmean="sd",
                ))
            fig.update_layout(
                title=f"Box Plot {sel_feat}", yaxis_title=sel_feat,
                height=350, plot_bgcolor="white", paper_bgcolor="white",
            )
            show_chart(fig, use_container_width=True)

        # All features subplots
        st.markdown("#### Semua Fitur (Grid)")
        ncols_g = 4
        nrows_g = -(-len(feat_list) // ncols_g)
        fig = make_subplots(rows=nrows_g, cols=ncols_g,
                            subplot_titles=feat_list, shared_yaxes=False)
        for i, col in enumerate(feat_list):
            r, c_pos = divmod(i, ncols_g)
            for ov, lbl, clr in [(0,"Non-DM",COLORS['primary']),
                                  (1,"Diabetes",COLORS['danger'])]:
                fig.add_trace(go.Histogram(
                    x=df[df['Outcome']==ov][col], name=lbl,
                    opacity=0.66, marker_color=clr, showlegend=(i==0), nbinsx=20,
                ), row=r+1, col=c_pos+1)
        fig.update_layout(barmode="overlay", height=230*nrows_g,
                          title_text="Semua Fitur berdasarkan Kelas Target",
                          plot_bgcolor="white", paper_bgcolor="white")
        show_chart(fig, use_container_width=True)

    # ── Tab 3: Korelasi ──────────────────────────────────────
    with tab3:
        st.markdown("### 🔗 Correlation Heatmap")
        corr = df.corr().round(3)
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
            colorscale="RdBu_r", zmid=0,
            text=corr.values, texttemplate="%{text:.2f}",
            textfont=dict(size=10),
            colorbar=dict(title="r"),
        ))
        fig.update_layout(title="Correlation Matrix – Pima Indians Dataset",
                          height=520, plot_bgcolor="white", paper_bgcolor="white")
        show_chart(fig, use_container_width=True)

        st.markdown("#### Korelasi dengan Target (Outcome)")
        tc = df.corr()['Outcome'].drop('Outcome').sort_values(ascending=False)
        fig = go.Figure(go.Bar(
            x=tc.index, y=tc.values,
            marker_color=[COLORS['danger'] if v > 0 else COLORS['primary'] for v in tc.values],
            text=tc.round(3).values, textposition="outside",
        ))
        fig.update_layout(title="Korelasi Fitur vs Outcome",
                          xaxis_title="Fitur", yaxis_title="Pearson r",
                          height=360, plot_bgcolor="white", paper_bgcolor="white")
        show_chart(fig, use_container_width=True)

        st.markdown("#### Scatter Plot Multivariat")
        c1, c2 = st.columns(2)
        with c1:
            xf = st.selectbox("Fitur X:", feat_list, index=1)
        with c2:
            yf = st.selectbox("Fitur Y:", feat_list, index=5)
        fig = px.scatter(df, x=xf, y=yf, color="Outcome",
                         color_discrete_map={0:COLORS['primary'], 1:COLORS['danger']},
                         opacity=0.65, title=f"Scatter: {xf} vs {yf}")
        fig.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white")
        show_chart(fig, use_container_width=True)

    # ── Tab 4: Kualitas Data ─────────────────────────────────
    with tab4:
        st.markdown("### 🎯 Analisis Kualitas Data")

        # Missing values implicit
        st.markdown("#### Nilai 0 Tidak Valid (Implicit Missing Values)")
        st.info("Nilai 0 pada Glucose, BloodPressure, SkinThickness, Insulin, BMI tidak mungkin secara medis → dianggap data hilang.")
        zd = [{"Fitur": c, "Jumlah 0": (df[c]==0).sum(),
                "Persentase (%)": round((df[c]==0).mean()*100,2)} for c in ZERO_INVALID]
        zdf = pd.DataFrame(zd)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(zdf, use_container_width=True, hide_index=True)
        with c2:
            fig = go.Figure(go.Bar(
                x=zdf["Fitur"], y=zdf["Persentase (%)"],
                marker_color=COLORS['accent'],
                text=zdf["Persentase (%)"].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
            ))
            fig.update_layout(title="Persentase Nilai 0 Tidak Valid",
                              height=310, plot_bgcolor="white", paper_bgcolor="white")
            show_chart(fig, use_container_width=True)

        # Outlier IQR
        st.markdown("---")
        st.markdown("#### Deteksi Outlier (Metode IQR)")
        out_rows = []
        for col in COLUMNS[:-1]:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
            n_out = ((df[col] < lo) | (df[col] > hi)).sum()
            out_rows.append({"Fitur": col, "Q1": round(q1,2), "Q3": round(q3,2),
                             "IQR": round(iqr,2), "Lower": round(lo,2), "Upper": round(hi,2),
                             "# Outlier": n_out, "% Outlier": round(n_out/len(df)*100,2)})
        st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)

        # Duplicates
        n_dup = df.duplicated().sum()
        if n_dup == 0:
            st.success("✅ Tidak ditemukan baris duplikat dalam dataset.")
        else:
            st.warning(f"⚠️ Ditemukan {n_dup} baris duplikat.")

    # ── Tab 5: 5 Key Insights ───────────────────────────────
    with tab5:
        st.markdown("### 💡 5 Key Insights dari Data")
        dm = df[df['Outcome']==1]
        nd = df[df['Outcome']==0]

        gl_dm  = dm['Glucose'].mean()
        gl_nd  = nd['Glucose'].mean()
        gl_dif = (gl_dm - gl_nd) / gl_nd * 100
        bmi_dm = dm['BMI'].mean()
        bmi_nd = nd['BMI'].mean()
        age_dm = dm['Age'].mean()
        age_nd = nd['Age'].mean()
        ins_zero = (df['Insulin']==0).mean()*100
        hi_gluc  = (dm['Glucose']>140).mean()*100

        insights = [
            ("#C73E1D","🩸","Insight 1: Glukosa adalah Prediktor Terkuat",
             f"Penderita diabetes memiliki rata-rata glukosa <strong>{gl_dm:.1f} mg/dL</strong> "
             f"vs non-diabetes <strong>{gl_nd:.1f} mg/dL</strong> — selisih <strong>{gl_dif:.1f}%</strong>. "
             f"Korelasi dengan Outcome = {df['Glucose'].corr(df['Outcome']):.2f}. "
             f"{hi_gluc:.1f}% penderita diabetes memiliki glukosa > 140 mg/dL."),

            ("#A23B72","⚖️","Insight 2: BMI Tinggi Berkorelasi dengan Diabetes",
             f"Rata-rata BMI diabetes ({bmi_dm:.1f}) jauh lebih tinggi dari non-diabetes ({bmi_nd:.1f}). "
             f"Ini mengkonfirmasi bahwa obesitas (BMI ≥ 30) merupakan faktor risiko utama. "
             f"Korelasi Pearson BMI–Outcome = {df['BMI'].corr(df['Outcome']):.2f}."),

            ("#2E86AB","🎂","Insight 3: Usia Berpengaruh pada Risiko",
             f"Rata-rata usia penderita diabetes ({age_dm:.1f} tahun) lebih tinggi dari non-diabetes "
             f"({age_nd:.1f} tahun). Risiko meningkat signifikan setelah usia 40 tahun. "
             f"Korelasi Usia–Outcome = {df['Age'].corr(df['Outcome']):.2f}."),

            ("#F18F01","💉","Insight 4: Data Insulin Paling Banyak Hilang",
             f"{ins_zero:.1f}% nilai Insulin bernilai 0 (tidak valid medis) — tertinggi di antara semua fitur. "
             f"Ini menunjukkan banyak data tidak terukur/tercatat. "
             f"Diperlukan imputasi median per kelas untuk mengatasi masalah ini."),

            ("#6B7280","⚖️","Insight 5: Dataset Tidak Seimbang (Class Imbalance)",
             f"Rasio kelas: Non-Diabetes {n0/len(df)*100:.1f}% vs Diabetes {n1/len(df)*100:.1f}%. "
             f"Ketidakseimbangan ini berpotensi membuat model bias ke kelas mayoritas. "
             f"Ditangani dengan parameter class_weight='balanced' pada model dan pemilihan F1-Score "
             f"sebagai metrik utama daripada Accuracy semata."),
        ]

        for clr, icon, title, content in insights:
            st.markdown(f"""
            <div style="background:white;padding:1.35rem;border-radius:13px;
                        border-left:5px solid {clr};box-shadow:0 4px 16px rgba(0,0,0,.07);
                        margin-bottom:.9rem;">
              <h4 style="color:{clr};margin-top:0;font-size:.98rem;">{icon} {title}</h4>
              <p style="color:#374151;line-height:1.72;margin:0;font-size:.9rem;">{content}</p>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 3 – PREDIKSI DIABETES
# ═══════════════════════════════════════════════════════════════
def page_prediksi():
    st.markdown("""
    <div class="hero-card" style="padding:1.4rem 2rem;">
      <h1 style="font-size:1.9rem;">🤖 Prediksi Diabetes</h1>
      <p style="margin:0;">Input data kesehatan pasien → Dapatkan prediksi real-time dari model terbaik</p>
    </div>
    """, unsafe_allow_html=True)

    if not models_ready():
        st.error("⚠️ Model belum dilatih. Jalankan terlebih dahulu:")
        st.code("python train.py", language="bash")
        return

    models   = load_models()
    scaler   = load_scaler()
    results  = load_results()
    if not models or scaler is None:
        st.error("Gagal memuat model. Pastikan train.py sudah dijalankan.")
        return

    feat_names = results.get('feature_names') if results else None
    best_name  = results.get('best_model','Random Forest') if results else 'Random Forest'

    c1, c2 = st.columns([2.2, 1])

    with c1:
        st.markdown("### 📋 Form Input Data Pasien")
        with st.form("pred_form"):
            fa, fb = st.columns(2)
            with fa:
                pregnancies = st.slider("🤰 Jumlah Kehamilan",      0,  17,  3)
                glucose     = st.slider("🩸 Glukosa (mg/dL)",       44, 199, 120)
                bp          = st.slider("💓 Tekanan Darah (mmHg)",  24, 122,  70)
                skin        = st.slider("📏 Kulit Tricep (mm)",      7,  99,  23)
            with fb:
                insulin = st.slider("💉 Insulin (mu U/ml)",  14, 846,  79)
                bmi     = st.slider("⚖️ BMI (kg/m²)",       18.2, 67.1, 32.0, 0.1)
                dpf     = st.slider("🧬 Diabetes Pedigree", 0.078, 2.42, 0.471, 0.001)
                age     = st.slider("🎂 Usia (tahun)",       21,  81,  33)

            available_models = [n for n in models if n != 'Best Model']
            idx_best = available_models.index(best_name) if best_name in available_models else 0
            sel_model = st.selectbox("🤖 Pilih Model:", available_models, index=idx_best)

            submit = st.form_submit_button("🔍 PREDIKSI SEKARANG!", use_container_width=True)

    with c2:
        st.markdown("""
        <div style="background:white;padding:1.25rem;border-radius:13px;box-shadow:0 4px 18px rgba(0,0,0,.08);">
          <h4 style="color:#1a1a2e;margin-top:0;font-size:.95rem;">📊 Nilai Referensi Normal</h4>
          <table style="width:100%;font-size:.82rem;border-collapse:collapse;">
            <tr style="border-bottom:1px solid #e5e7eb;"><td style="padding:.35rem 0;color:#6B7280;">Glukosa Normal</td><td style="font-weight:700;color:#065f46;">&lt; 100 mg/dL</td></tr>
            <tr style="border-bottom:1px solid #e5e7eb;"><td style="padding:.35rem 0;color:#6B7280;">Glukosa Pre-DM</td><td style="font-weight:700;color:#92400e;">100–125 mg/dL</td></tr>
            <tr style="border-bottom:1px solid #e5e7eb;"><td style="padding:.35rem 0;color:#6B7280;">Glukosa Diabetes</td><td style="font-weight:700;color:#991b1b;">&gt; 125 mg/dL</td></tr>
            <tr style="border-bottom:1px solid #e5e7eb;"><td style="padding:.35rem 0;color:#6B7280;">BMI Normal</td><td style="font-weight:700;color:#065f46;">18.5–24.9</td></tr>
            <tr style="border-bottom:1px solid #e5e7eb;"><td style="padding:.35rem 0;color:#6B7280;">BMI Overweight</td><td style="font-weight:700;color:#92400e;">25.0–29.9</td></tr>
            <tr style="border-bottom:1px solid #e5e7eb;"><td style="padding:.35rem 0;color:#6B7280;">BMI Obese</td><td style="font-weight:700;color:#991b1b;">≥ 30.0</td></tr>
            <tr style="border-bottom:1px solid #e5e7eb;"><td style="padding:.35rem 0;color:#6B7280;">TD Diastolik Normal</td><td style="font-weight:700;color:#065f46;">&lt; 80 mmHg</td></tr>
            <tr><td style="padding:.35rem 0;color:#6B7280;">TD Diastolik Tinggi</td><td style="font-weight:700;color:#991b1b;">≥ 90 mmHg</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    if submit:
        model = models.get(sel_model)
        if model and scaler:
            X_in   = build_input(pregnancies, glucose, bp, skin, insulin,
                                  bmi, dpf, age, scaler, feat_names)
            pred   = model.predict(X_in)[0]
            prob   = model.predict_proba(X_in)[0]
            dm_prob = float(prob[1])

            st.markdown("---")
            st.markdown("### 🎯 Hasil Prediksi")

            _, cc, _ = st.columns([1, 2, 1])
            with cc:
                if pred == 1:
                    rlv = "TINGGI" if dm_prob >= 0.7 else "SEDANG"
                    st.markdown(f"""
                    <div class="pred-positive">
                      <div style="font-size:3rem;margin-bottom:.5rem;">⚠️</div>
                      <h2 style="color:white;margin:0 0 .4rem;font-size:1.6rem;">TERDETEKSI DIABETES</h2>
                      <p style="font-size:1.05rem;opacity:.92;margin:0;">
                        Probabilitas Diabetes: <strong>{dm_prob*100:.1f}%</strong>
                      </p>
                      <p style="font-size:.88rem;opacity:.8;margin:.4rem 0 0;">
                        Tingkat Risiko: <strong>{rlv}</strong> &nbsp;·&nbsp; Model: {sel_model}
                      </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="pred-negative">
                      <div style="font-size:3rem;margin-bottom:.5rem;">✅</div>
                      <h2 style="color:white;margin:0 0 .4rem;font-size:1.6rem;">TIDAK TERDETEKSI DIABETES</h2>
                      <p style="font-size:1.05rem;opacity:.92;margin:0;">
                        Probabilitas Diabetes: <strong>{dm_prob*100:.1f}%</strong>
                      </p>
                      <p style="font-size:.88rem;opacity:.8;margin:.4rem 0 0;">
                        Tingkat Risiko: <strong>RENDAH</strong> &nbsp;·&nbsp; Model: {sel_model}
                      </p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            g1, g2 = st.columns(2)

            with g1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=dm_prob * 100,
                    title={"text": "Probabilitas Diabetes (%)", "font": {"size": 15}},
                    gauge={
                        "axis"     : {"range": [0, 100]},
                        "bar"      : {"color": COLORS['danger'] if pred==1 else COLORS['success']},
                        "steps"    : [
                            {"range": [0, 40],  "color": "#d1fae5"},
                            {"range": [40, 60], "color": "#fef3c7"},
                            {"range": [60, 100],"color": "#fee2e2"},
                        ],
                        "threshold": {"line": {"color":"red","width":3},
                                      "thickness": 0.78, "value": 50},
                    },
                ))
                fig.update_layout(height=300, paper_bgcolor="white")
                show_chart(fig, use_container_width=True)

            with g2:
                # Risk factor summary
                def risk_tag(val, lo_thr, hi_thr, lo_lbl="NORMAL", hi_lbl="HIGH"):
                    if val >= hi_thr: return hi_lbl, COLORS['danger']
                    if val >= lo_thr: return "MEDIUM", COLORS['accent']
                    return lo_lbl, COLORS['success']

                risks = [
                    ("🩸 Glukosa",     glucose, *risk_tag(glucose, 100, 126)),
                    ("⚖️ BMI",         bmi,     *risk_tag(bmi, 25, 30)),
                    ("🎂 Usia",        age,     *risk_tag(age, 35, 50)),
                    ("💓 Tek. Darah",  bp,      *risk_tag(bp, 80, 90)),
                ]
                st.markdown('<div style="background:white;padding:1.2rem;border-radius:13px;box-shadow:0 4px 18px rgba(0,0,0,.08);">'
                            '<h4 style="color:#1a1a2e;margin-top:0;font-size:.93rem;">⚡ Analisis Faktor Risiko</h4>',
                            unsafe_allow_html=True)
                for fname, fval, flevel, fclr in risks:
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                padding:.38rem 0;border-bottom:1px solid #f3f4f6;">
                      <span style="font-size:.88rem;color:#374151;">{fname}: <strong>{fval}</strong></span>
                      <span style="background:{fclr}22;color:{fclr};padding:.12rem .6rem;
                                   border-radius:9999px;font-size:.74rem;font-weight:700;">{flevel}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Semua model
            st.markdown("---")
            st.markdown("#### 🔄 Prediksi dari Semua Model")
            rows = []
            for mn, mm in models.items():
                if mn == 'Best Model': continue
                Xi  = build_input(pregnancies, glucose, bp, skin, insulin,
                                  bmi, dpf, age, scaler, feat_names)
                mp  = mm.predict(Xi)[0]
                mpr = mm.predict_proba(Xi)[0][1]
                rows.append({"Model": mn,
                             "Prediksi": "🔴 Diabetes" if mp==1 else "🟢 Tidak Diabetes",
                             "Prob. Diabetes (%)": f"{mpr*100:.2f}%",
                             "⭐ Terbaik?": "⭐" if mn == best_name else ""})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("""
            <div style="background:#fef3c7;padding:.9rem 1.1rem;border-radius:10px;
                        border-left:4px solid #F18F01;margin-top:.8rem;">
              <p style="color:#92400e;margin:0;font-size:.84rem;">
                ⚠️ <strong>Disclaimer Medis:</strong> Prediksi ini bersifat indikatif dan
                TIDAK menggantikan diagnosis medis dari dokter profesional. Selalu berkonsultasi
                dengan tenaga kesehatan berkompeten untuk penanganan yang tepat.
              </p>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 4 – EVALUASI MODEL
# ═══════════════════════════════════════════════════════════════
def page_evaluasi():
    st.markdown("""
    <div class="hero-card" style="padding:1.4rem 2rem;">
      <h1 style="font-size:1.9rem;">📈 Evaluasi Model</h1>
      <p style="margin:0;">Perbandingan komprehensif performa semua model Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    if not models_ready():
        st.error("⚠️ Model belum dilatih. Jalankan `python train.py`.")
        return

    res = load_results()
    if not res:
        st.error("Hasil evaluasi tidak ditemukan."); return

    best_m   = res.get('best_model','')
    m_names  = [k for k in res['results'] if k not in ('best_model','feature_names','dataset_info')]

    # ── Summary cards ──
    st.markdown(f"### ⭐ Performa Model Terbaik: {best_m} (Test Set)")
    bt = res['results'][best_m]['test']
    c_cards = [
        ("Accuracy", bt['accuracy'],  "#2E86AB"),
        ("Precision",bt['precision'], "#A23B72"),
        ("Recall",   bt['recall'],    "#F18F01"),
        ("F1-Score", bt['f1'],        "#C73E1D"),
        ("AUC-ROC",  bt['roc_auc'],   "#6B7280"),
    ]
    cols = st.columns(5)
    for col, (lbl, val, clr) in zip(cols, c_cards):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{clr};text-align:center;">
              <h3 style="color:{clr};font-size:1.55rem;">{val:.4f}</h3>
              <p style="font-size:.8rem;">{lbl}</p>
              <small style="color:{clr};font-weight:700;">{best_m}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabel perbandingan ──
    st.markdown("### 📊 Tabel Perbandingan Performa Semua Model")
    rows = []
    for mn in m_names:
        for sp in ['train','val','test']:
            mr = res['results'][mn][sp]
            rows.append({
                "Model"    : mn,
                "Split"    : sp.capitalize(),
                "Accuracy" : f"{mr['accuracy']:.4f}",
                "Precision": f"{mr['precision']:.4f}",
                "Recall"   : f"{mr['recall']:.4f}",
                "F1-Score" : f"{mr['f1']:.4f}",
                "AUC-ROC"  : f"{mr['roc_auc']:.4f}",
                "Terbaik?" : "⭐" if mn == best_m else "",
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Bar chart ──
    st.markdown("### 📊 Grafik Perbandingan (Test Set)")
    metrics_k = ['accuracy','precision','recall','f1','roc_auc']
    metrics_l = ['Accuracy','Precision','Recall','F1-Score','AUC-ROC']
    fig = go.Figure()
    for mn, clr in MODEL_COLORS.items():
        if mn not in res['results']: continue
        vals = [res['results'][mn]['test'][k] for k in metrics_k]
        fig.add_trace(go.Bar(
            name=mn, x=metrics_l, y=vals, marker_color=clr, opacity=0.85,
            text=[f"{v:.3f}" for v in vals], textposition="outside",
        ))
    fig.update_layout(barmode="group", yaxis_range=[0, 1.12],
                      height=450, plot_bgcolor="white", paper_bgcolor="white",
                      legend=dict(x=0.75, y=0.98))
    show_chart(fig, use_container_width=True)

    # ── Radar chart ──
    st.markdown("### 🕸️ Radar Chart")
    fig = go.Figure()
    cats = metrics_l + [metrics_l[0]]
    for mn, clr in MODEL_COLORS.items():
        if mn not in res['results']: continue
        vals = [res['results'][mn]['test'][k] for k in metrics_k]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats, fill='toself', name=mn,
            line_color=clr, fillcolor=clr, opacity=0.28,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        height=450, plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(x=0.82, y=1.0),
    )
    show_chart(fig, use_container_width=True)

    # ── Confusion Matrices (tabs) ──
    st.markdown("### 🎯 Confusion Matrix")
    tabs_cm = st.tabs(m_names)
    for tab, mn in zip(tabs_cm, m_names):
        with tab:
            cm = np.array(res['results'][mn]['test']['confusion_matrix'])
            tn, fp, fn, tp = cm.ravel()
            c1, c2 = st.columns([1.1, 1])
            with c1:
                z_flip = cm[::-1]
                fig = go.Figure(go.Heatmap(
                    z=z_flip, colorscale="Blues",
                    x=["Pred: Non-DM","Pred: Diabetes"],
                    y=["Act: Diabetes","Act: Non-DM"],
                    text=[[f"<b>{v}</b>" for v in row] for row in z_flip],
                    texttemplate="%{text}", textfont=dict(size=22), showscale=True,
                ))
                f1v = res['results'][mn]['test']['f1']
                fig.update_layout(
                    title=f"Confusion Matrix – {mn}<br>F1-Score = {f1v:.4f}",
                    height=400, plot_bgcolor="white", paper_bgcolor="white",
                )
                show_chart(fig, use_container_width=True)

            with c2:
                sens = tp / (tp+fn) if (tp+fn) > 0 else 0
                spec = tn / (tn+fp) if (tn+fp) > 0 else 0
                ppv  = tp / (tp+fp) if (tp+fp) > 0 else 0
                npv  = tn / (tn+fn) if (tn+fn) > 0 else 0

                for lbl, val, bg, tc in [
                    (f"True Positive (TP) = {tp}","Diabetes benar diprediksi Diabetes","#d1fae5","#065f46"),
                    (f"True Negative (TN) = {tn}","Non-DM benar diprediksi Non-DM","#d1fae5","#065f46"),
                    (f"False Positive (FP) = {fp}","Non-DM salah → Diabetes (tipe I)","#fee2e2","#991b1b"),
                    (f"False Negative (FN) = {fn}","Diabetes salah → Non-DM (tipe II)","#fef3c7","#92400e"),
                ]:
                    st.markdown(f"""
                    <div style="background:{bg};padding:.55rem .8rem;border-radius:8px;
                                margin-bottom:.45rem;">
                      <p style="margin:0;font-weight:700;color:{tc};font-size:.88rem;">{lbl}</p>
                      <p style="margin:0;font-size:.8rem;color:{tc};">{val}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("**Metrik Turunan:**")
                der = pd.DataFrame([
                    {"Metrik":"Sensitivity (Recall)","Nilai":f"{sens:.4f}"},
                    {"Metrik":"Specificity",         "Nilai":f"{spec:.4f}"},
                    {"Metrik":"PPV (Precision)",      "Nilai":f"{ppv:.4f}"},
                    {"Metrik":"NPV",                  "Nilai":f"{npv:.4f}"},
                ])
                st.dataframe(der, use_container_width=True, hide_index=True)

    # ── ROC Curves ──
    st.markdown("### 📉 ROC Curves – Perbandingan")
    fig = go.Figure()
    for mn, clr in MODEL_COLORS.items():
        if mn not in res['results']: continue
        yt = np.array(res['results'][mn]['test_y_true'])
        yp = np.array(res['results'][mn]['test_y_prob'])
        fpr, tpr, _ = roc_curve(yt, yp)
        auc_v = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{mn} (AUC={auc_v:.4f})",
                                  line=dict(color=clr, width=2.5)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Classifier',
                              line=dict(color='gray', width=1.5, dash='dash')))
    fig.update_layout(title='ROC Curves – Semua Model',
                      xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                      xaxis_range=[0,1], yaxis_range=[0,1.05],
                      height=500, plot_bgcolor='white', paper_bgcolor='white',
                      legend=dict(x=0.5, y=0.1))
    show_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 5 – INTERPRETASI & BISNIS
# ═══════════════════════════════════════════════════════════════
def page_interpretasi():
    st.markdown("""
    <div class="hero-card" style="padding:1.4rem 2rem;">
      <h1 style="font-size:1.9rem;">💡 Interpretasi & Insights Bisnis</h1>
      <p style="margin:0;">Analisis feature importance, justifikasi model, dan rekomendasi strategis</p>
    </div>
    """, unsafe_allow_html=True)

    if not models_ready():
        st.error("⚠️ Jalankan `python train.py` terlebih dahulu!"); return

    fi  = load_feature_importances()
    res = load_results()
    if not fi or not res:
        st.error("Data tidak ditemukan."); return

    best_m = res.get('best_model','')
    bt     = res['results'][best_m]['test']

    # ── Feature Importance per model ──
    st.markdown("### 🔬 Feature Importance Analysis")
    t1, t2, t3 = st.tabs(["Logistic Regression","Random Forest","Gradient Boosting"])

    def fi_bar(fi_data, key, color, title, xlabel):
        feat = fi_data[key]['feature_names']
        if key == 'logistic_regression':
            vals = fi_data[key]['coefficients']
            clrs = [COLORS['danger'] if v > 0 else COLORS['primary'] for v in vals]
        else:
            vals = fi_data[key]['importances']
            clrs = color
        sorted_idx = np.argsort(np.abs(vals))
        fig = go.Figure(go.Bar(
            y=np.array(feat)[sorted_idx],
            x=np.array(vals)[sorted_idx],
            orientation='h',
            marker_color=(np.array(clrs)[sorted_idx].tolist()
                          if isinstance(clrs, list) else clrs),
            text=[f"{v:.4f}" for v in np.array(vals)[sorted_idx]],
            textposition='outside',
        ))
        fig.update_layout(title=title, xaxis_title=xlabel,
                          height=420, plot_bgcolor='white', paper_bgcolor='white')
        return fig

    with t1:
        lr_fig = fi_bar(fi,'logistic_regression',None,
            "Logistic Regression – Koefisien Fitur","Koefisien")
        show_chart(lr_fig, use_container_width=True)
        lr_vals = fi.get('logistic_regression', {}).get('coefficients', [])
        has_neg = any(v < 0 for v in lr_vals)
        if has_neg:
            st.info("🔴 **Merah** = koefisien positif (meningkatkan risiko diabetes) · "
                    "🔵 **Biru** = koefisien negatif (mengurangi risiko diabetes).")
        else:
            st.info("ℹ️ Semua koefisien bernilai positif — seluruh fitur berkontribusi "
                    "meningkatkan risiko diabetes dalam model Logistic Regression ini. "
                    "Semakin besar nilai koefisien, semakin kuat pengaruhnya.")

    with t2:
        show_chart(fi_bar(fi,'random_forest',COLORS['secondary'],
            "Random Forest – Feature Importances","Importance Score"),
            use_container_width=True)
    with t3:
        show_chart(fi_bar(fi,'gradient_boosting',COLORS['accent'],
            "Gradient Boosting – Feature Importances","Importance Score"),
            use_container_width=True)

    # ── Performa best model ──
    st.markdown("---")
    st.markdown(f"### ⭐ Justifikasi Model Terbaik: {best_m}")
    c1, c2 = st.columns([1.8, 1])
    with c1:
        st.markdown(f"""
        <div style="background:white;padding:1.4rem;border-radius:13px;box-shadow:0 4px 18px rgba(0,0,0,.08);">
          <h4 style="color:#1a1a2e;margin-top:0;">📊 Metrik Performa {best_m} – Test Set</h4>
          <table style="width:100%;border-collapse:collapse;">
            <tr style="background:#f8f9fa;">
              <th style="padding:.45rem;text-align:left;font-size:.88rem;">Metrik</th>
              <th style="padding:.45rem;text-align:center;font-size:.88rem;">Nilai</th>
              <th style="padding:.45rem;text-align:left;font-size:.88rem;">Interpretasi</th>
            </tr>
            <tr style="border-bottom:1px solid #e5e7eb;">
              <td style="padding:.45rem;font-size:.88rem;">Accuracy</td>
              <td style="padding:.45rem;text-align:center;font-weight:800;color:{COLORS['primary']};font-size:.95rem;">{bt['accuracy']:.4f}</td>
              <td style="padding:.45rem;font-size:.82rem;color:#6B7280;">{bt['accuracy']*100:.1f}% total prediksi benar</td>
            </tr>
            <tr style="border-bottom:1px solid #e5e7eb;">
              <td style="padding:.45rem;font-size:.88rem;">Precision</td>
              <td style="padding:.45rem;text-align:center;font-weight:800;color:{COLORS['secondary']};font-size:.95rem;">{bt['precision']:.4f}</td>
              <td style="padding:.45rem;font-size:.82rem;color:#6B7280;">{bt['precision']*100:.1f}% prediksi diabetes benar-benar positif</td>
            </tr>
            <tr style="border-bottom:1px solid #e5e7eb;">
              <td style="padding:.45rem;font-size:.88rem;">Recall</td>
              <td style="padding:.45rem;text-align:center;font-weight:800;color:{COLORS['accent']};font-size:.95rem;">{bt['recall']:.4f}</td>
              <td style="padding:.45rem;font-size:.82rem;color:#6B7280;">{bt['recall']*100:.1f}% kasus diabetes berhasil terdeteksi</td>
            </tr>
            <tr style="border-bottom:1px solid #e5e7eb;">
              <td style="padding:.45rem;font-size:.88rem;">F1-Score</td>
              <td style="padding:.45rem;text-align:center;font-weight:800;color:{COLORS['danger']};font-size:.95rem;">{bt['f1']:.4f}</td>
              <td style="padding:.45rem;font-size:.82rem;color:#6B7280;">Harmonic mean Precision & Recall</td>
            </tr>
            <tr>
              <td style="padding:.45rem;font-size:.88rem;">AUC-ROC</td>
              <td style="padding:.45rem;text-align:center;font-weight:800;color:#6B7280;font-size:.95rem;">{bt['roc_auc']:.4f}</td>
              <td style="padding:.45rem;font-size:.82rem;color:#6B7280;">Kemampuan diskriminasi {bt['roc_auc']*100:.1f}%</td>
            </tr>
          </table>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="background:white;padding:1.3rem;border-radius:13px;box-shadow:0 4px 18px rgba(0,0,0,.08);">
          <h4 style="color:#1a1a2e;margin-top:0;font-size:.93rem;">✅ Kriteria Pemilihan Model</h4>
          <ul style="color:#374151;font-size:.86rem;line-height:2;padding-left:1.1rem;">
            <li>F1-Score <strong>tertinggi</strong> pada validation set</li>
            <li>Keseimbangan Precision & Recall terbaik</li>
            <li>AUC-ROC tinggi → diskriminasi kelas unggul</li>
            <li>Robust terhadap class imbalance</li>
            <li>Tidak overfitting (train–test gap kecil)</li>
            <li>Mendukung class_weight dan feature importance</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    # ── Business insights ──
    st.markdown("---")
    st.markdown("### 💼 Insights Bisnis & Rekomendasi Strategis")
    bis_insights = [
        ("#2E86AB","🏥","Rekomendasi untuk Fasilitas Kesehatan",
         "<ul style='line-height:1.9;font-size:.88rem;'>"
         "<li>Implementasikan sistem skrining ML di klinik endokrinologi dan puskesmas</li>"
         "<li>Prioritaskan pemeriksaan kadar glukosa dan BMI sebagai indikator risiko utama</li>"
         "<li>Gunakan model ini untuk triase pasien agar alokasi sumber daya medis lebih efisien</li>"
         "<li>Integrasi dengan Electronic Health Record (EHR) yang sudah berjalan</li>"
         "<li>Tetapkan threshold prediksi ≥ 0.4 untuk meminimalkan false negative berbahaya</li>"
         "</ul>"),

        ("#A23B72","🔬","Faktor Risiko Utama yang Perlu Diperhatikan",
         "<p style='font-size:.88rem;line-height:1.7;'>Berdasarkan analisis feature importance:</p>"
         "<ol style='line-height:1.9;font-size:.88rem;'>"
         "<li><strong>Glukosa</strong> – Prediktor terkuat. Pemantauan rutin kadar gula darah sangat penting</li>"
         "<li><strong>BMI</strong> – Obesitas meningkatkan risiko signifikan. Program diet & olahraga diprioritaskan</li>"
         "<li><strong>Usia</strong> – Risiko meningkat setelah 35 tahun. Skrining berkala untuk kelompok ini</li>"
         "<li><strong>Diabetes Pedigree Function</strong> – Riwayat keluarga berpengaruh besar</li>"
         "<li><strong>Tekanan Darah</strong> – Hipertensi sering menjadi komorbiditas diabetes</li>"
         "</ol>"),

        ("#F18F01","📊","Keterbatasan Model & Area Pengembangan",
         "<ul style='line-height:1.9;font-size:.88rem;'>"
         "<li><strong>Sampel terbatas:</strong> 768 data dari populasi spesifik (wanita Pima Indian ≥21 th)</li>"
         "<li><strong>Class imbalance:</strong> 65% non-DM vs 35% DM → ditangani dengan class_weight='balanced'</li>"
         "<li><strong>Missing values:</strong> ~48% Insulin kosong → imputasi median per kelas</li>"
         "<li><strong>Fitur terbatas:</strong> tidak ada data HbA1c, kolesterol, gaya hidup, pola makan</li>"
         "<li><strong>Saran next step:</strong> SMOTE untuk oversampling, XGBoost/LightGBM, deep learning, SHAP</li>"
         "</ul>"),

        ("#C73E1D","🎯","Status Pencapaian Metrik Kesuksesan",
         f"<table style='width:100%;border-collapse:collapse;font-size:.88rem;'>"
         f"<tr style='background:#f8f9fa;'>"
         f"<th style='padding:.4rem;text-align:left;'>Metrik</th>"
         f"<th style='padding:.4rem;text-align:center;'>Target</th>"
         f"<th style='padding:.4rem;text-align:center;'>Hasil</th>"
         f"<th style='padding:.4rem;text-align:center;'>Status</th></tr>"
         f"<tr style='border-bottom:1px solid #e5e7eb;'>"
         f"<td style='padding:.38rem;'>Accuracy</td>"
         f"<td style='padding:.38rem;text-align:center;'>≥ 75%</td>"
         f"<td style='padding:.38rem;text-align:center;font-weight:700;'>{bt['accuracy']*100:.1f}%</td>"
         f"<td style='padding:.38rem;text-align:center;'>{'✅' if bt['accuracy']>=0.75 else '❌'}</td></tr>"
         f"<tr style='border-bottom:1px solid #e5e7eb;'>"
         f"<td style='padding:.38rem;'>F1-Score</td>"
         f"<td style='padding:.38rem;text-align:center;'>≥ 0.70</td>"
         f"<td style='padding:.38rem;text-align:center;font-weight:700;'>{bt['f1']:.4f}</td>"
         f"<td style='padding:.38rem;text-align:center;'>{'✅' if bt['f1']>=0.70 else '❌'}</td></tr>"
         f"<tr style='border-bottom:1px solid #e5e7eb;'>"
         f"<td style='padding:.38rem;'>AUC-ROC</td>"
         f"<td style='padding:.38rem;text-align:center;'>≥ 0.80</td>"
         f"<td style='padding:.38rem;text-align:center;font-weight:700;'>{bt['roc_auc']:.4f}</td>"
         f"<td style='padding:.38rem;text-align:center;'>{'✅' if bt['roc_auc']>=0.80 else '❌'}</td></tr>"
         f"<tr><td style='padding:.38rem;'>Recall</td>"
         f"<td style='padding:.38rem;text-align:center;'>≥ 0.75</td>"
         f"<td style='padding:.38rem;text-align:center;font-weight:700;'>{bt['recall']:.4f}</td>"
         f"<td style='padding:.38rem;text-align:center;'>{'✅' if bt['recall']>=0.75 else '❌'}</td></tr>"
         f"</table>"),
    ]
    for clr, icon, title, content in bis_insights:
        st.markdown(f"""
        <div style="background:white;padding:1.35rem;border-radius:13px;
                    border-left:5px solid {clr};box-shadow:0 4px 16px rgba(0,0,0,.07);
                    margin-bottom:.9rem;">
          <h4 style="color:{clr};margin-top:0;font-size:.98rem;">{icon} {title}</h4>
          {content}
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 6 – DOKUMENTASI
# ═══════════════════════════════════════════════════════════════
def page_dokumentasi():
    st.markdown("""
    <div class="hero-card" style="padding:1.4rem 2rem;">
      <h1 style="font-size:1.9rem;">📚 Dokumentasi</h1>
      <p style="margin:0;">Dataset · Metodologi · Cara Penggunaan</p>
    </div>
    """, unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["📁 Dataset","⚙️ Metodologi","🚀 Cara Penggunaan"])

    with t1:
        st.markdown("""
        <div style="background:white;padding:1.4rem;border-radius:13px;box-shadow:0 4px 18px rgba(0,0,0,.08);margin-bottom:1rem;">
          <h4 style="color:#1a1a2e;margin-top:0;">Pima Indians Diabetes Database</h4>
          <p style="color:#374151;line-height:1.8;font-size:.92rem;">
            Dataset ini berasal dari <strong>National Institute of Diabetes and Digestive and
            Kidney Diseases (NIDDK)</strong>, USA. Seluruh subjek adalah perempuan berusia ≥21 tahun
            keturunan Pima Indian, yang memiliki prevalensi diabetes tipe 2 sangat tinggi di dunia.
          </p>
          <hr style="border-color:#e5e7eb;">
          <h5>🔗 Sumber Dataset</h5>
          <ul style="color:#374151;line-height:1.9;font-size:.9rem;">
            <li>Kaggle: <em>kaggle.com/uciml/pima-indians-diabetes-database</em></li>
            <li>UCI ML Repository: <em>archive.ics.uci.edu/ml/datasets/diabetes</em></li>
            <li>GitHub (Brownlee): <em>raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv</em></li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

        feat_desc = [
            (1,"Pregnancies","Integer","Jumlah kehamilan","0–17","–"),
            (2,"Glucose","Integer","Konsentrasi plasma glukosa 2 jam (OGTT)","0–199","mg/dL"),
            (3,"BloodPressure","Integer","Tekanan darah diastolik","0–122","mmHg"),
            (4,"SkinThickness","Integer","Ketebalan lipatan kulit tricep","0–99","mm"),
            (5,"Insulin","Integer","Insulin serum 2 jam","0–846","mu U/ml"),
            (6,"BMI","Float","Body Mass Index","0–67.1","kg/m²"),
            (7,"DiabetesPedigreeFunction","Float","Fungsi riwayat diabetes keluarga","0.078–2.42","–"),
            (8,"Age","Integer","Usia pasien","21–81","tahun"),
            (9,"Outcome","Integer","Label target (0=No DM, 1=DM)","0 atau 1","–"),
        ]
        st.markdown("### 📋 Deskripsi Fitur")
        st.dataframe(pd.DataFrame(feat_desc, columns=["No","Fitur","Tipe","Deskripsi","Range","Satuan"]),
                     use_container_width=True, hide_index=True)

    with t2:
        st.markdown("### ⚙️ Metodologi & Alur Kerja")
        steps_doc = [
            ("1","#2E86AB","📥 Data Acquisition",
             "Dataset diunduh dari sumber publik (GitHub Brownlee). Diverifikasi kesesuaian "
             "jumlah baris, kolom, dan statistik deskriptif dengan dokumentasi UCI. Data disimpan "
             "dalam format CSV dengan header yang informatif."),
            ("2","#A23B72","🔍 Exploratory Data Analysis",
             "Analisis menyeluruh meliputi: (a) statistik deskriptif (mean, std, percentile), "
             "(b) identifikasi implicit missing values (nilai 0 tidak valid secara medis), "
             "(c) deteksi outlier dengan IQR, (d) analisis distribusi per kelas, dan "
             "(e) analisis korelasi Pearson antar fitur."),
            ("3","#F18F01","⚙️ Preprocessing",
             "Langkah: (a) ganti nilai 0 tidak valid dengan NaN pada 5 kolom, "
             "(b) imputasi median per kelas (stratified), "
             "(c) feature engineering — BMI_Category, Glucose_Category, Age_Group, Insulin_Glucose_Ratio, "
             "(d) stratified split 70/15/15, "
             "(e) StandardScaler di-fit hanya pada training set untuk mencegah data leakage."),
            ("4","#C73E1D","🤖 Pemodelan",
             "Tiga algoritma dibandingkan: (1) Logistic Regression — baseline linear, "
             "(2) Random Forest — ensemble bagging dengan decision trees, "
             "(3) Gradient Boosting — ensemble boosting sekuensial. "
             "Setiap model di-tuning menggunakan GridSearchCV dengan 5-fold Stratified CV "
             "dengan scoring metrik F1-Score."),
            ("5","#27AE60","📊 Evaluasi",
             "Metrik komprehensif: Accuracy, Precision, Recall, F1-Score, AUC-ROC, dan Confusion Matrix "
             "dihitung pada train/val/test set. Model terbaik dipilih berdasarkan F1-Score pada "
             "validation set — karena F1 menyeimbangkan Precision & Recall, sangat penting untuk "
             "kasus medis di mana false negative (diabetes tidak terdeteksi) sangat berbahaya."),
            ("6","#6B7280","🚀 Deployment",
             "Aplikasi web interaktif dibangun dengan Streamlit, menampilkan 6 halaman: "
             "Beranda, EDA Dashboard, Prediksi Real-time, Evaluasi Model, Interpretasi & Bisnis, "
             "dan Dokumentasi. Semua visualisasi dibuat interaktif menggunakan Plotly."),
        ]
        for num, clr, title, desc in steps_doc:
            st.markdown(f"""
            <div style="background:white;padding:1.2rem;border-radius:12px;
                        border-left:5px solid {clr};box-shadow:0 3px 14px rgba(0,0,0,.07);
                        margin-bottom:.8rem;display:flex;gap:1rem;align-items:flex-start;">
              <div style="background:{clr};color:white;width:36px;height:36px;border-radius:50%;
                          display:flex;align-items:center;justify-content:center;
                          font-weight:800;flex-shrink:0;">{num}</div>
              <div>
                <h4 style="color:{clr};margin:0 0 .4rem;font-size:.96rem;">{title}</h4>
                <p style="color:#374151;line-height:1.72;margin:0;font-size:.88rem;text-align:justify;">{desc}</p>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🛠️ Tech Stack")
        tech = {"Python 3.10+":"Bahasa pemrograman","Scikit-learn":"ML framework",
                "Pandas/NumPy":"Data manipulation","Matplotlib/Seaborn":"Static viz",
                "Plotly":"Interactive viz","Streamlit":"Web app","Joblib":"Model serialization",
                "GridSearchCV":"Hyperparameter tuning"}
        cols = st.columns(4)
        for i, (t, d) in enumerate(tech.items()):
            with cols[i%4]:
                st.markdown(f"""
                <div style="background:white;padding:.7rem;border-radius:9px;
                            box-shadow:0 2px 8px rgba(0,0,0,.07);text-align:center;margin-bottom:.4rem;">
                  <p style="font-weight:700;color:#2E86AB;margin:0;font-size:.83rem;">{t}</p>
                  <p style="color:#6B7280;margin:0;font-size:.74rem;">{d}</p>
                </div>
                """, unsafe_allow_html=True)

    with t3:
        st.markdown("### 🚀 Cara Penggunaan")
        st.code("""
# 1. Clone repository
git clone https://github.com/Nathanielaiueo/diabetes-ml-project.git
cd diabetes-ml-project

# 2. Buat virtual environment (opsional, sangat disarankan)
python -m venv venv
venv\\Scripts\\activate        # Windows
# source venv/bin/activate    # Linux/Mac

# 3. Install semua dependensi
pip install -r requirements.txt

# 4. Jalankan pipeline training (download data + latih 3 model + simpan hasil)
python train.py

# 5. Jalankan aplikasi Streamlit
streamlit run app.py

# 6. Buka browser (otomatis terbuka atau manual)
#    http://localhost:8501
        """, language="bash")

        st.markdown("### 📁 Struktur Repository")
        st.code("""
diabetes-ml-project/
├── 📄 README.md                    # Dokumentasi utama
├── 📄 requirements.txt             # Dependensi Python
├── 📄 train.py                     # Pipeline ML lengkap (soal 1-3)
├── 📄 app.py                       # Aplikasi Streamlit (soal 4)
├── 📄 .gitignore
│
├── 📁 data/
│   ├── diabetes.csv                # Dataset original (Pima Indians)
│   └── diabetes_processed.csv     # Dataset setelah preprocessing
│
├── 📁 models/
│   ├── best_model.pkl              # Model terbaik (siap digunakan)
│   ├── logistic_regression.pkl     # Model LR
│   ├── random_forest.pkl           # Model RF
│   ├── gradient_boosting.pkl       # Model GB
│   ├── scaler.pkl                  # StandardScaler yang sudah di-fit
│   ├── results.json                # Hasil evaluasi semua model
│   ├── feature_info.json           # Informasi nama fitur
│   └── feature_importances.json   # Feature importance per model
│
├── 📁 notebooks/
│   └── 01_EDA_Preprocessing.ipynb # Notebook EDA & Preprocessing lengkap
│
└── 📁 reports/
    ├── Laporan_Teknis.pdf          # Laporan teknis (soal 5)
    └── figures/                   # Grafik & visualisasi
        ├── 01_class_distribution.png
        ├── 02_feature_distributions.png
        ├── 03_correlation_heatmap.png
        ├── 04_boxplots.png
        ├── 05_confusion_matrices.png
        ├── 06_roc_curves.png
        ├── 07_feature_importance.png
        └── 08_model_comparison.png
        """, language="text")

        st.markdown("### 🧭 Panduan Halaman Aplikasi")
        pages_guide = [
            ("🏠 Beranda","Informasi tim, statistik dataset, latar belakang, dan alur pipeline"),
            ("📊 EDA Dashboard","Eksplorasi data interaktif: distribusi, korelasi, kualitas data, 5 key insights"),
            ("🤖 Prediksi Diabetes","Form input data pasien → prediksi real-time + analisis faktor risiko"),
            ("📈 Evaluasi Model","Tabel perbandingan, confusion matrix, ROC curves, radar chart"),
            ("💡 Interpretasi & Bisnis","Feature importance, justifikasi model, rekomendasi strategis"),
            ("📚 Dokumentasi","Deskripsi dataset, metodologi, dan cara penggunaan aplikasi"),
        ]
        for pg, desc in pages_guide:
            st.markdown(f"""
            <div style="display:flex;gap:1rem;align-items:flex-start;padding:.65rem;
                        border-bottom:1px solid #e5e7eb;">
              <div style="min-width:210px;font-weight:700;color:#2E86AB;font-size:.9rem;">{pg}</div>
              <div style="font-size:.88rem;color:#374151;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    page = render_sidebar()

    if   page == "🏠 Beranda":              page_beranda()
    elif page == "📊 EDA Dashboard":         page_eda()
    elif page == "🤖 Prediksi Diabetes":     page_prediksi()
    elif page == "📈 Evaluasi Model":        page_evaluasi()
    elif page == "💡 Interpretasi & Bisnis": page_interpretasi()
    elif page == "📚 Dokumentasi":           page_dokumentasi()

if __name__ == "__main__":
    main()
