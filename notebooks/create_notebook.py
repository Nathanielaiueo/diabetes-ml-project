#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_notebook.py - Generate Jupyter Notebook EDA secara programatik
UAS Pembelajaran Mesin - Genap 2025/2026 - UDINUS Semarang

Kelompok:
  - Fahmi Fatmawati Azzahra  (NIM: A11.2024.15831)
  - Nathaniela Febry Nathasa (NIM: A11.2024.15850)

Jalankan: python notebooks/create_notebook.py
"""

import nbformat as nbf
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "01_EDA_Preprocessing.ipynb"

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    return nbf.v4.new_markdown_cell(src)

def code(src):
    return nbf.v4.new_code_cell(src)

# ===========================================================
# TITLE
# ===========================================================
cells.append(md("""# 🩺 Prediksi Penyakit Diabetes — EDA & Preprocessing
## UAS Pembelajaran Mesin | Semester Genap 2025/2026
### Universitas Dian Nuswantoro (UDINUS) Semarang

| | |
|---|---|
| **Kelompok** | Fahmi Fatmawati Azzahra (A11.2024.15831) & Nathaniela Febry Nathasa (A11.2024.15850) |
| **Dataset** | Pima Indians Diabetes Database (NIDDK) |
| **Task** | Binary Classification — Deteksi Penyakit Diabetes |
| **Notebook** | 01 — Exploratory Data Analysis & Preprocessing |

---
"""))

# ===========================================================
# SECTION 1: IMPORT & CONFIG
# ===========================================================
cells.append(md("""## 1. Import Library & Konfigurasi

Mengimpor semua library yang diperlukan untuk EDA dan preprocessing.
"""))

cells.append(code("""import os
import sys
import warnings
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

# Konfigurasi plot
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
FIGDPI = 100

# Konstanta
COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
ZERO_INVALID_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
RANDOM_STATE = 42

PALETTE = {
    'primary'  : '#2E86AB',
    'secondary': '#A23B72',
    'accent'   : '#F18F01',
    'danger'   : '#C73E1D',
    'success'  : '#27AE60',
}

print("[OK] Library berhasil diimport")
print(f"  NumPy   : {np.__version__}")
print(f"  Pandas  : {pd.__version__}")
"""))

# ===========================================================
# SECTION 2: DATA ACQUISITION
# ===========================================================
cells.append(md("""---
## 2. Data Acquisition (Soal 1)

### 2.1 Problem Statement

**Domain:** Kesehatan / Medis — Prediksi Penyakit Diabetes

Diabetes mellitus merupakan penyakit kronis yang ditandai oleh kadar glukosa darah tinggi akibat gangguan produksi atau kerja insulin. Menurut IDF Diabetes Atlas 2021, terdapat **537 juta** orang dewasa dengan diabetes di seluruh dunia, diproyeksikan meningkat menjadi **783 juta** pada 2045.

**Tujuan Bisnis/Analisis:**
- Mengembangkan model ML yang mampu memprediksi risiko diabetes berdasarkan data kesehatan dasar
- Mendukung skrining dini untuk mengurangi biaya dan risiko komplikasi
- Mengidentifikasi faktor risiko paling berpengaruh terhadap diabetes

**Metrik Kesuksesan:**
- Accuracy ≥ 75%
- F1-Score ≥ 0.70
- AUC-ROC ≥ 0.80
- Recall ≥ 0.75 (minimisasi false negative)

### 2.2 Sumber Dataset

- **Nama:** Pima Indians Diabetes Database
- **Institusi:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
- **Akses:** UCI Machine Learning Repository, Kaggle
- **URL Kaggle:** https://www.kaggle.com/uciml/pima-indians-diabetes-database
- **URL UCI:** https://archive.ics.uci.edu/ml/datasets/diabetes
"""))

cells.append(code("""# Download dataset jika belum ada
DATA_DIR = Path('../data')
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATASET_PATH = DATA_DIR / 'diabetes.csv'

DATASET_URL = ("https://raw.githubusercontent.com/jbrownlee/"
               "Datasets/master/pima-indians-diabetes.data.csv")

if DATASET_PATH.exists():
    print(f"[OK] Dataset sudah ada: {DATASET_PATH}")
else:
    print("-> Mengunduh dataset...")
    urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
    print(f"[OK] Dataset berhasil diunduh -> {DATASET_PATH}")

# Load dataset
df_raw = pd.read_csv(DATASET_PATH, header=None, names=COLUMNS)
df_raw.to_csv(DATASET_PATH, index=False)  # Simpan dengan header

print(f"\\n[STATS] Dataset berhasil dimuat!")
print(f"   Shape: {df_raw.shape[0]} baris × {df_raw.shape[1]} kolom")
"""))

# ===========================================================
# SECTION 3: DESKRIPTIF AWAL
# ===========================================================
cells.append(md("""---
## 3. Statistik Deskriptif Awal (Soal 1 — Keluaran Wajib)
"""))

cells.append(code("""# Preview data
print("=== 5 BARIS PERTAMA ===")
display(df_raw.head())
"""))

cells.append(code("""# Informasi tipe data
print("=== INFO DATASET ===")
df_raw.info()
"""))

cells.append(code("""# Statistik deskriptif lengkap
print("=== STATISTIK DESKRIPTIF ===")
display(df_raw.describe().T.round(3))
"""))

cells.append(code("""# Distribusi kelas target
print("=== DISTRIBUSI KELAS TARGET ===")
vc = df_raw['Outcome'].value_counts()
print(f"Non-Diabetes (0) : {vc[0]:4d} ({vc[0]/len(df_raw)*100:.1f}%)")
print(f"Diabetes     (1) : {vc[1]:4d} ({vc[1]/len(df_raw)*100:.1f}%)")
print(f"\\nTotal           : {len(df_raw)}")
print(f"Rasio imbalance  : {vc[0]/vc[1]:.2f} : 1")
"""))

# ===========================================================
# SECTION 4: EDA
# ===========================================================
cells.append(md("""---
## 4. Exploratory Data Analysis (Soal 2)

### 4.1 Analisis Kualitas Data
"""))

cells.append(code("""# === 4.1.1 MISSING VALUES (NaN eksplisit) ===
print("=== MISSING VALUES (NaN) ===")
nan_counts = df_raw.isnull().sum()
print(nan_counts)
print(f"\\nTotal NaN: {nan_counts.sum()}")
print("-> Tidak ada NaN eksplisit, tapi ada nilai 0 yang tidak valid secara medis.")
"""))

cells.append(code("""# === 4.1.2 NILAI 0 TIDAK VALID (implicit missing values) ===
print("=== NILAI 0 TIDAK VALID ===")
print(f"{'Kolom':<35} {'Jumlah 0':>10} {'Persen %':>10}")
print("-" * 58)
zero_info = {}
for col in ZERO_INVALID_COLS:
    cnt = (df_raw[col] == 0).sum()
    pct = cnt / len(df_raw) * 100
    zero_info[col] = cnt
    print(f"  {col:<33} {cnt:>10} {pct:>9.1f}%")
print(f"\\n-> Kolom Insulin memiliki nilai 0 terbanyak: {zero_info['Insulin']} data ({zero_info['Insulin']/len(df_raw)*100:.1f}%)")
"""))

cells.append(code("""# === 4.1.3 DUPLIKAT ===
n_dup = df_raw.duplicated().sum()
print(f"Jumlah baris duplikat: {n_dup}")
if n_dup == 0:
    print("[OK] Dataset bebas duplikat.")
"""))

cells.append(code("""# === 4.1.4 OUTLIER (IQR Method) ===
print("=== OUTLIER DETECTION (IQR × 1.5) ===")
print(f"{'Kolom':<35} {'Outlier':>8} {'%':>6} {'Batas Bawah':>12} {'Batas Atas':>12}")
print("-" * 78)
for col in COLUMNS[:-1]:
    q1, q3 = df_raw[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    n_out = ((df_raw[col] < lo) | (df_raw[col] > hi)).sum()
    pct = n_out / len(df_raw) * 100
    print(f"  {col:<33} {n_out:>8} {pct:>5.1f}% {lo:>12.2f} {hi:>12.2f}")
"""))

cells.append(md("""### 4.2 Visualisasi EDA — Insight 1: Distribusi Kelas Target
"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Insight 1: Distribusi Kelas Target (Diabetes vs Non-Diabetes)',
             fontsize=13, fontweight='bold')

cnts = df_raw['Outcome'].value_counts()
clrs = [PALETTE['primary'], PALETTE['danger']]
lbls = ['Non-Diabetes (0)', 'Diabetes (1)']

# Bar
axes[0].bar(lbls, [cnts[0], cnts[1]], color=clrs, edgecolor='white', linewidth=2)
for i, v in enumerate([cnts[0], cnts[1]]):
    axes[0].text(i, v+4, f'{v}\\n({v/len(df_raw)*100:.1f}%)', ha='center', fontweight='bold')
axes[0].set_ylabel('Jumlah Sampel')
axes[0].set_title('Bar Chart Distribusi Kelas')

# Pie
axes[1].pie([cnts[0], cnts[1]], labels=lbls, colors=clrs,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize':11})
axes[1].set_title('Pie Chart Proporsi Kelas')

plt.tight_layout()
plt.savefig('../reports/figures/01_class_distribution.png', dpi=FIGDPI, bbox_inches='tight')
plt.show()
print("[WARNING]  Dataset TIDAK SEIMBANG: rasio 65:35. Perlu class_weight='balanced' pada model.")
"""))

cells.append(md("""### 4.3 Visualisasi — Insight 2: Distribusi Fitur per Kelas
"""))

cells.append(code("""fig, axes = plt.subplots(3, 3, figsize=(16, 13))
fig.suptitle('Insight 2: Distribusi Fitur Berdasarkan Kelas Target',
             fontsize=13, fontweight='bold')
axes_flat = axes.flatten()

for i, col in enumerate(COLUMNS[:-1]):
    ax = axes_flat[i]
    for ov, lbl, clr in [(0, 'Non-DM', PALETTE['primary']),
                          (1, 'Diabetes', PALETTE['danger'])]:
        data = df_raw[df_raw['Outcome'] == ov][col]
        ax.hist(data, bins=22, alpha=0.65, label=lbl, color=clr, edgecolor='none')
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frekuensi')
    ax.legend(fontsize=8)

for j in range(i+1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.tight_layout()
plt.savefig('../reports/figures/02_feature_distributions.png', dpi=FIGDPI, bbox_inches='tight')
plt.show()
"""))

cells.append(md("""### 4.4 Visualisasi — Insight 3: Korelasi Antar Fitur
"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Insight 3: Analisis Korelasi', fontsize=13, fontweight='bold')

# Heatmap
corr = df_raw.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, ax=axes[0], linewidths=0.5, square=True, cbar_kws={'shrink':.8})
axes[0].set_title('Correlation Heatmap (Lower Triangle)', fontweight='bold')

# Korelasi vs Outcome
tc = corr['Outcome'].drop('Outcome').sort_values(ascending=False)
clrs_bar = [PALETTE['danger'] if v > 0 else PALETTE['primary'] for v in tc.values]
axes[1].barh(tc.index, tc.values, color=clrs_bar, edgecolor='white', linewidth=0.5)
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_xlabel('Korelasi Pearson (r)')
axes[1].set_title('Korelasi Fitur vs Outcome', fontweight='bold')

for i, (name, val) in enumerate(tc.items()):
    axes[1].text(val + (0.008 if val >= 0 else -0.008), i,
                f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right',
                fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('../reports/figures/03_correlation_heatmap.png', dpi=FIGDPI, bbox_inches='tight')
plt.show()

print("\\nKorelasi Fitur dengan Outcome (urutan):")
for feat, val in tc.items():
    print(f"  {feat:<30}: {val:+.4f}")
"""))

cells.append(md("""### 4.5 Visualisasi — Insight 4: Deteksi Outlier (Boxplots)
"""))

cells.append(code("""fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('Insight 4: Boxplot per Fitur dan Kelas (Deteksi Outlier)',
             fontsize=13, fontweight='bold')
axes_flat = axes.flatten()

for i, col in enumerate(COLUMNS[:-1]):
    ax = axes_flat[i]
    data0 = df_raw[df_raw['Outcome'] == 0][col].dropna().values
    data1 = df_raw[df_raw['Outcome'] == 1][col].dropna().values
    bp = ax.boxplot([data0, data1], labels=['Non-DM','Diabetes'],
                    patch_artist=True, medianprops=dict(color='black', lw=2.5),
                    flierprops=dict(marker='o', markersize=4, alpha=0.4))
    bp['boxes'][0].set_facecolor(PALETTE['primary'] + '99')
    bp['boxes'][1].set_facecolor(PALETTE['danger']  + '99')
    ax.set_title(col, fontweight='bold')
    ax.set_ylabel('Nilai')

plt.tight_layout()
plt.savefig('../reports/figures/04_boxplots.png', dpi=FIGDPI, bbox_inches='tight')
plt.show()
"""))

cells.append(md("""### 4.6 Visualisasi — Insight 5: Nilai 0 Tidak Valid
"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Insight 5: Nilai 0 Tidak Valid (Implicit Missing Values)',
             fontsize=13, fontweight='bold')

# Bar chart jumlah nilai 0
zero_counts = {col: (df_raw[col] == 0).sum() for col in ZERO_INVALID_COLS}
pct_vals = [v/len(df_raw)*100 for v in zero_counts.values()]
bar_clrs = [PALETTE['primary'], PALETTE['secondary'], PALETTE['accent'],
            PALETTE['danger'], PALETTE['success']]

bars = axes[0].bar(zero_counts.keys(), zero_counts.values(), color=bar_clrs, edgecolor='white')
for bar, val, pct in zip(bars, zero_counts.values(), pct_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}\\n({pct:.1f}%)', ha='center', fontsize=9, fontweight='bold')
axes[0].set_ylabel('Jumlah Nilai 0')
axes[0].set_title('Jumlah dan Persentase Nilai 0 Tidak Valid')

# Distribusi insulin sebelum dan sesudah filter 0
axes[1].hist(df_raw['Insulin'], bins=30, alpha=0.6, color=PALETTE['primary'],
             label=f'Semua data (inkl. 0)', edgecolor='none')
axes[1].hist(df_raw[df_raw['Insulin'] > 0]['Insulin'], bins=30, alpha=0.6,
             color=PALETTE['danger'], label='Insulin > 0 saja', edgecolor='none')
axes[1].axvline(0, color='black', linewidth=2, linestyle='--', label='Nilai 0')
axes[1].set_xlabel('Insulin (mu U/ml)')
axes[1].set_ylabel('Frekuensi')
axes[1].set_title('Distribusi Insulin (Nilai 0 = Data Hilang)')
axes[1].legend()

plt.tight_layout()
plt.show()
"""))

cells.append(md("""---
## 5. Preprocessing (Soal 2)

### 5.1 Ganti Nilai 0 Tidak Valid -> NaN
"""))

cells.append(code("""df = df_raw.copy()

# Ganti nilai 0 yang tidak valid secara medis dengan NaN
for col in ZERO_INVALID_COLS:
    sebelum = (df[col] == 0).sum()
    df[col] = df[col].replace(0, np.nan)
    print(f"  {col:<30}: {sebelum} nilai 0 -> NaN")

print(f"\\nTotal NaN setelah penggantian: {df.isnull().sum().sum()}")
"""))

cells.append(md("""### 5.2 Imputasi Nilai Hilang (Stratified Median Imputation)

Imputasi menggunakan **median per kelas** (bukan median global) untuk mempertahankan
distribusi statistik dalam setiap kelompok.
"""))

cells.append(code("""print("=== STRATIFIED MEDIAN IMPUTATION ===")
print(f"{'Kolom':<30} {'Median(0)':>10} {'Median(1)':>10} {'Jumlah Diimputasi':>18}")
print("-" * 70)

for col in ZERO_INVALID_COLS:
    n_nan = df[col].isna().sum()
    for outcome_val in [0, 1]:
        mask_nan   = (df['Outcome'] == outcome_val) & df[col].isna()
        mask_valid = (df['Outcome'] == outcome_val) & df[col].notna()
        med = df.loc[mask_valid, col].median()
        df.loc[mask_nan, col] = med
    
    med0 = df[df['Outcome']==0][col].median()
    med1 = df[df['Outcome']==1][col].median()
    print(f"  {col:<28} {med0:>10.2f} {med1:>10.2f} {n_nan:>18}")

print(f"\\nNaN setelah imputasi: {df.isnull().sum().sum()}")
print("[OK] Semua nilai berhasil diimputasi!")
"""))

cells.append(md("""### 5.3 Feature Engineering

Menambahkan fitur turunan yang relevan secara domain medis.
"""))

cells.append(code("""# BMI Category: 0=Underweight(<18.5), 1=Normal(18.5-24.9), 2=Overweight(25-29.9), 3=Obese(≥30)
df['BMI_Category'] = pd.cut(df['BMI'],
                             bins=[0, 18.5, 25.0, 30.0, float('inf')],
                             labels=[0, 1, 2, 3]).astype(float)

# Glucose Category: 0=Normal(<100), 1=Pre-Diabetes(100-125), 2=Diabetes(≥126)
df['Glucose_Category'] = pd.cut(df['Glucose'],
                                  bins=[0, 99.9, 125.9, float('inf')],
                                  labels=[0, 1, 2]).astype(float)

# Age Group: 0=Young(<35), 1=Middle(35-50), 2=Senior(>50)
df['Age_Group'] = pd.cut(df['Age'],
                          bins=[0, 34.9, 49.9, float('inf')],
                          labels=[0, 1, 2]).astype(float)

# Insulin-Glucose Ratio
df['Insulin_Glucose_Ratio'] = df['Insulin'] / (df['Glucose'] + 1.0)

feat_eng = ['BMI_Category', 'Glucose_Category', 'Age_Group', 'Insulin_Glucose_Ratio']
print("Fitur baru yang ditambahkan:")
for f in feat_eng:
    print(f"  + {f}")

print(f"\\nTotal fitur sekarang: {df.shape[1]-1} (dari {len(COLUMNS)-1} + {len(feat_eng)} baru)")
display(df[feat_eng + ['Outcome']].head(10))
"""))

cells.append(md("""### 5.4 Train / Validation / Test Split (70% / 15% / 15%)
"""))

cells.append(code("""from sklearn.model_selection import train_test_split

feature_cols = [c for c in df.columns if c != 'Outcome']
X = df[feature_cols].values
y = df['Outcome'].values

# Split pertama: TV (85%) vs Test (15%)
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)

# Split kedua: Train (70% total) vs Val (15% total)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.1765, random_state=RANDOM_STATE, stratify=y_tv
)

print("=== PEMBAGIAN DATASET ===")
print(f"  Training set  : {len(X_train):4d} sampel ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Validation set: {len(X_val):4d} sampel ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test set      : {len(X_test):4d} sampel ({len(X_test)/len(X)*100:.1f}%)")
print(f"  Total         : {len(X):4d} sampel")
print()
print("Distribusi kelas di setiap split:")
for name, ys in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    print(f"  {name}: Non-DM={sum(ys==0)} | Diabetes={sum(ys==1)}")
"""))

cells.append(md("""### 5.5 Feature Scaling (StandardScaler)

StandardScaler menstandardisasi fitur dengan mengurangi mean dan membagi std deviation.
**Penting:** Scaler di-fit HANYA pada training data untuk mencegah data leakage.
"""))

cells.append(code("""from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # Fit + Transform pada training
X_val_sc   = scaler.transform(X_val)          # Hanya Transform (no fit)
X_test_sc  = scaler.transform(X_test)         # Hanya Transform (no fit)

print("StandardScaler berhasil diterapkan!")
print()
print("Statistik sebelum scaling (train):")
df_train_pre = pd.DataFrame(X_train, columns=feature_cols)
print(df_train_pre.describe().round(3).T[['mean','std','min','max']])
print()
print("Statistik setelah scaling (train):")
df_train_sc = pd.DataFrame(X_train_sc, columns=feature_cols)
print(df_train_sc.describe().round(3).T[['mean','std','min','max']])
"""))

cells.append(md("""---
## 6. Ringkasan Preprocessing

| Langkah | Teknik | Justifikasi |
|---------|--------|-------------|
| Deteksi Missing Values | Nilai 0 pada kolom medis | 0 tidak mungkin secara medis untuk Glucose, BMI, dll. |
| Imputasi | Median per kelas (stratified) | Mempertahankan distribusi per kelas, robust terhadap outlier |
| Feature Engineering | BMI/Glucose/Age categories + Ratio | Informasi domain medis tambahan |
| Train/Val/Test Split | 70/15/15 stratified | Distribusi kelas terjaga di semua split |
| Feature Scaling | StandardScaler (fit on train only) | Mencegah data leakage |

---

## 7. Ringkasan 5 Key Insights

1. **🩸 Glukosa = Prediktor Terkuat** — Korelasi tertinggi dengan Outcome (r ≈ 0.47)
2. **⚖️ BMI Tinggi -> Risiko Tinggi** — Penderita diabetes rata-rata BMI jauh lebih tinggi
3. **🎂 Usia Berpengaruh** — Penderita diabetes rata-rata lebih tua
4. **💉 Data Insulin Banyak Hilang** — ~48% data Insulin bernilai 0 (tidak valid)
5. **⚖️ Class Imbalance** — 65% Non-DM vs 35% DM -> perlu penanganan khusus

---

*Notebook ini merupakan bagian dari Soal 2 (EDA & Preprocessing) UAS Pembelajaran Mesin.*  
*Kelompok: Fahmi Fatmawati Azzahra (A11.2024.15831) & Nathaniela Febry Nathasa (A11.2024.15850)*
"""))

# ── Assemble & write ──────────────────────────────────────────
nb['cells'] = cells
nb.metadata.update({
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbformat": 4,
        "nbformat_minor": 5,
        "pygments_lexer": "ipython3",
        "version": "3.10.0"
    }
})

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"[OK] Notebook berhasil dibuat: {OUTPUT_PATH}")
print(f"  Jumlah sel: {len(nb['cells'])}")
print()
print("Untuk membuka notebook:")
print("  jupyter notebook notebooks/01_EDA_Preprocessing.ipynb")
