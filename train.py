#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
# Fix Windows console encoding (cp1252 → utf-8)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
"""
=============================================================================
TRAIN.PY - Machine Learning Pipeline untuk Prediksi Diabetes
=============================================================================
Mata Kuliah  : Pembelajaran Mesin
Semester     : Genap 2025/2026
Universitas  : Universitas Dian Nuswantoro (UDINUS) Semarang

Kelompok:
  - Fahmi Fatmawati Azzahra  (NIM: A11.2024.15831)
  - Nathaniela Febry Nathasa (NIM: A11.2024.15850)

Deskripsi:
  Script ini menjalankan pipeline Machine Learning end-to-end untuk prediksi
  penyakit diabetes, mencakup: akuisisi data, EDA, preprocessing, pelatihan
  3 model (Logistic Regression, Random Forest, Gradient Boosting), evaluasi
  komprehensif, hyperparameter tuning, dan penyimpanan model terbaik.

Dataset:
  Pima Indians Diabetes Database - National Institute of Diabetes and
  Digestive and Kidney Diseases (NIDDK)
  Sumber: https://raw.githubusercontent.com/jbrownlee/Datasets/master/

Cara Menjalankan:
  python train.py
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
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, average_precision_score
)
import joblib

warnings.filterwarnings('ignore')

# =============================================================================
# KONFIGURASI & PATH
# =============================================================================
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / 'data'
MODEL_DIR   = BASE_DIR / 'models'
REPORT_DIR  = BASE_DIR / 'reports'
FIGURES_DIR = REPORT_DIR / 'figures'

for d in [DATA_DIR, MODEL_DIR, REPORT_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATASET_URL  = ("https://raw.githubusercontent.com/jbrownlee/"
                "Datasets/master/pima-indians-diabetes.data.csv")
DATASET_PATH = DATA_DIR / 'diabetes.csv'

COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

# Kolom dengan nilai 0 yang tidak valid secara medis
ZERO_INVALID_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

RANDOM_STATE = 42

# Palet warna konsisten
PALETTE = {
    'primary'   : '#2E86AB',
    'secondary' : '#A23B72',
    'accent'    : '#F18F01',
    'danger'    : '#C73E1D',
    'success'   : '#27AE60',
    'neutral'   : '#6B7280',
}

# =============================================================================
# HEADER
# =============================================================================
SEP = "=" * 70
print(SEP)
print("  UAS PEMBELAJARAN MESIN - SEMESTER GENAP 2025/2026")
print("  Universitas Dian Nuswantoro (UDINUS) Semarang")
print()
print("  Kelompok:")
print("    - Fahmi Fatmawati Azzahra  (NIM: A11.2024.15831)")
print("    - Nathaniela Febry Nathasa (NIM: A11.2024.15850)")
print(SEP)
print()
print("  Topic  : Prediksi Penyakit Diabetes")
print("  Dataset: Pima Indians Diabetes Database (NIDDK)")
print("  Models : Logistic Regression | Random Forest | Gradient Boosting")
print(SEP)
print()

# =============================================================================
# STEP 1 - DATA ACQUISITION
# =============================================================================
print("[STEP 1/7] DATA ACQUISITION")
print("-" * 50)

def download_dataset() -> pd.DataFrame:
    """Unduh dataset jika belum ada, lalu muat ke DataFrame."""
    if DATASET_PATH.exists():
        print(f"  [OK] Dataset ditemukan di: {DATASET_PATH}")
    else:
        print(f"  -> Mengunduh dataset dari URL...")
        print(f"    {DATASET_URL}")
        try:
            urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
            print(f"  [OK] Dataset berhasil diunduh -> {DATASET_PATH}")
        except Exception as e:
            print(f"  [FAIL] Gagal mengunduh: {e}")
            sys.exit(1)

    # Cek apakah file sudah punya header atau belum
    with open(DATASET_PATH, 'r') as f:
        first_line = f.readline().strip()
    if first_line.startswith('Pregnancies'):
        # File sudah punya header
        df = pd.read_csv(DATASET_PATH)
        # Pastikan semua kolom adalah numerik
        for col in COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        # File tanpa header (format asli)
        df = pd.read_csv(DATASET_PATH, header=None, names=COLUMNS)
    return df

df_raw = download_dataset()

# Simpan ulang dengan header (overwrite dengan data bersih)
df_raw.to_csv(DATA_DIR / 'diabetes.csv', index=False)

print()
print(f"  Dimensi dataset  : {df_raw.shape[0]} baris × {df_raw.shape[1]} kolom")
print(f"  Fitur input      : {list(df_raw.columns[:-1])}")
print(f"  Target           : {df_raw.columns[-1]}  (0 = Tidak Diabetes, 1 = Diabetes)")
print()
print(f"  Distribusi kelas :")
n0 = (df_raw['Outcome'] == 0).sum()
n1 = (df_raw['Outcome'] == 1).sum()
print(f"    Tidak Diabetes (0) : {n0:4d}  ({n0/len(df_raw)*100:.1f}%)")
print(f"    Diabetes       (1) : {n1:4d}  ({n1/len(df_raw)*100:.1f}%)")

# =============================================================================
# STEP 2 - EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print()
print("[STEP 2/7] EXPLORATORY DATA ANALYSIS")
print("-" * 50)

print("\n  [2.1] Statistik Deskriptif:")
desc = df_raw.describe().round(3)
print(desc.to_string())

print("\n  [2.2] Nilai Hilang (NaN eksplisit):")
nan_counts = df_raw.isnull().sum()
print(f"    Total NaN = {nan_counts.sum()}")

print("\n  [2.3] Nilai 0 Tidak Valid (implicit missing values):")
zero_dict = {}
for col in ZERO_INVALID_COLS:
    cnt = (df_raw[col] == 0).sum()
    pct = cnt / len(df_raw) * 100
    zero_dict[col] = cnt
    print(f"    {col:<30}: {cnt:3d} nilai ({pct:.1f}%)")

print("\n  [2.4] Duplikat:")
n_dup = df_raw.duplicated().sum()
print(f"    Jumlah baris duplikat = {n_dup}")

print("\n  [2.5] Outlier (Metode IQR, batas 1.5×IQR):")
for col in COLUMNS[:-1]:
    q1, q3 = df_raw[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    n_out = ((df_raw[col] < lo) | (df_raw[col] > hi)).sum()
    if n_out > 0:
        print(f"    {col:<30}: {n_out:3d} outlier  (batas [{lo:.1f}, {hi:.1f}])")

print("\n  [2.6] Korelasi dengan Target (Outcome):")
corr_with_target = df_raw.corr()['Outcome'].drop('Outcome').sort_values(ascending=False)
for feat, val in corr_with_target.items():
    bar = "#" * int(abs(val) * 20)
    sign = "+" if val >= 0 else "-"
    print(f"    {feat:<30}: {sign}{abs(val):.4f}  {bar}")

# =============================================================================
# STEP 3 - PREPROCESSING
# =============================================================================
print()
print("[STEP 3/7] PREPROCESSING")
print("-" * 50)

df = df_raw.copy()

# 3.1 Ganti nilai 0 tidak valid → NaN
print("\n  [3.1] Mengganti nilai 0 tidak valid dengan NaN...")
for col in ZERO_INVALID_COLS:
    df[col] = df[col].replace(0, np.nan)

# 3.2 Imputasi dengan median per kelas (stratified median imputation)
print("  [3.2] Imputasi NaN dengan median per kelas...")
for col in ZERO_INVALID_COLS:
    for outcome_val in [0, 1]:
        mask_nan   = (df['Outcome'] == outcome_val) & df[col].isna()
        mask_valid = (df['Outcome'] == outcome_val) & df[col].notna()
        median_val = df.loc[mask_valid, col].median()
        df.loc[mask_nan, col] = median_val
    cnt_imputed = zero_dict[col]
    print(f"    {col:<30}: {cnt_imputed} nilai diimputasi (median per kelas)")

print(f"\n    NaN setelah imputasi: {df.isnull().sum().sum()}")

# 3.3 Feature Engineering
print("\n  [3.3] Feature Engineering...")

# BMI Category: 0=Underweight, 1=Normal, 2=Overweight, 3=Obese
df['BMI_Category'] = pd.cut(
    df['BMI'],
    bins=[0, 18.5, 25.0, 30.0, float('inf')],
    labels=[0, 1, 2, 3]
).astype(float)

# Glucose Category: 0=Normal(<100), 1=Pre-Diabetes(100-125), 2=Diabetes(>=126)
df['Glucose_Category'] = pd.cut(
    df['Glucose'],
    bins=[0, 99.9, 125.9, float('inf')],
    labels=[0, 1, 2]
).astype(float)

# Age Group: 0=Young(<35), 1=Middle(35-50), 2=Senior(>50)
df['Age_Group'] = pd.cut(
    df['Age'],
    bins=[0, 34.9, 49.9, float('inf')],
    labels=[0, 1, 2]
).astype(float)

# Insulin-Glucose Ratio
df['Insulin_Glucose_Ratio'] = df['Insulin'] / (df['Glucose'] + 1.0)

engineered = ['BMI_Category', 'Glucose_Category', 'Age_Group', 'Insulin_Glucose_Ratio']
for feat in engineered:
    print(f"    + {feat}")

# 3.4 Definisi fitur & target
feature_cols = [c for c in df.columns if c != 'Outcome']
X = df[feature_cols].values
y = df['Outcome'].values
print(f"\n  Total fitur setelah engineering: {len(feature_cols)}")

# 3.5 Split: 70% Train / 15% Validation / 15% Test (stratified)
print("\n  [3.4] Pembagian dataset 70/15/15 (stratified)...")
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)
# 0.15 / 0.85 ≈ 0.1765 untuk mendapat ~15% total dari tv set
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.1765, random_state=RANDOM_STATE, stratify=y_tv
)

print(f"    Training set   : {len(X_train):4d} sampel ({len(X_train)/len(X)*100:.1f}%)")
print(f"    Validation set : {len(X_val):4d} sampel ({len(X_val)/len(X)*100:.1f}%)")
print(f"    Test set       : {len(X_test):4d} sampel ({len(X_test)/len(X)*100:.1f}%)")

# 3.6 Feature Scaling (StandardScaler)
print("\n  [3.5] Feature Scaling (StandardScaler) ...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit HANYA pada training set
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')
print("    [OK] scaler.pkl disimpan")

# Simpan info fitur
feat_info = {
    'feature_names'      : feature_cols,
    'original_features'  : COLUMNS[:-1],
    'engineered_features': engineered,
    'n_features'         : len(feature_cols),
}
with open(MODEL_DIR / 'feature_info.json', 'w') as f:
    json.dump(feat_info, f, indent=2)

# Simpan data yang sudah diproses
df.to_csv(DATA_DIR / 'diabetes_processed.csv', index=False)
print("    [OK] diabetes_processed.csv disimpan")

# =============================================================================
# STEP 4 - MODEL TRAINING & HYPERPARAMETER TUNING
# =============================================================================
print()
print("[STEP 4/7] MODEL TRAINING & HYPERPARAMETER TUNING")
print("-" * 50)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ---- Model 1: Logistic Regression ----
print("\n  [Model 1] Logistic Regression")
print("  Mencari hyperparameter terbaik dengan GridSearchCV (5-fold CV)...")
lr_param_grid = {
    'C'       : [0.01, 0.1, 1, 10, 100],
    'solver'  : ['lbfgs', 'liblinear'],
    'max_iter': [2000],
}
lr_grid = GridSearchCV(
    LogisticRegression(random_state=RANDOM_STATE),
    lr_param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=0,
    refit=True,
)
lr_grid.fit(X_train_sc, y_train)
lr_model = lr_grid.best_estimator_
print(f"  Best params : {lr_grid.best_params_}")
print(f"  Best CV F1  : {lr_grid.best_score_:.4f}")
joblib.dump(lr_model, MODEL_DIR / 'logistic_regression.pkl')
print("  [OK] logistic_regression.pkl disimpan")

# ---- Model 2: Random Forest ----
print("\n  [Model 2] Random Forest Classifier")
print("  Mencari hyperparameter terbaik dengan GridSearchCV (5-fold CV)...")
rf_param_grid = {
    'n_estimators'    : [100, 200, 300],
    'max_depth'       : [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf' : [1, 2],
    'class_weight'    : ['balanced', None],
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    rf_param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=0,
    refit=True,
)
rf_grid.fit(X_train_sc, y_train)
rf_model = rf_grid.best_estimator_
print(f"  Best params : {rf_grid.best_params_}")
print(f"  Best CV F1  : {rf_grid.best_score_:.4f}")
joblib.dump(rf_model, MODEL_DIR / 'random_forest.pkl')
print("  [OK] random_forest.pkl disimpan")

# ---- Model 3: Gradient Boosting ----
print("\n  [Model 3] Gradient Boosting Classifier")
print("  Mencari hyperparameter terbaik dengan GridSearchCV (5-fold CV)...")
gb_param_grid = {
    'n_estimators' : [100, 200, 300],
    'max_depth'    : [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample'    : [0.8, 1.0],
}
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=RANDOM_STATE),
    gb_param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=0,
    refit=True,
)
gb_grid.fit(X_train_sc, y_train)
gb_model = gb_grid.best_estimator_
print(f"  Best params : {gb_grid.best_params_}")
print(f"  Best CV F1  : {gb_grid.best_score_:.4f}")
joblib.dump(gb_model, MODEL_DIR / 'gradient_boosting.pkl')
print("  [OK] gradient_boosting.pkl disimpan")

# =============================================================================
# STEP 5 - EVALUASI MODEL
# =============================================================================
print()
print("[STEP 5/7] EVALUASI MODEL")
print("-" * 50)

def evaluate_model(model, X_tr, y_tr, X_v, y_v, X_te, y_te, name):
    """Evaluasi model di semua split; kembalikan dict hasil."""
    res = {'name': name}
    for split_name, Xs, ys in [('train', X_tr, y_tr),
                                ('val',   X_v,  y_v),
                                ('test',  X_te, y_te)]:
        y_pred = model.predict(Xs)
        y_prob = model.predict_proba(Xs)[:, 1]
        res[split_name] = {
            'accuracy' : float(accuracy_score(ys, y_pred)),
            'precision': float(precision_score(ys, y_pred, zero_division=0)),
            'recall'   : float(recall_score(ys, y_pred, zero_division=0)),
            'f1'       : float(f1_score(ys, y_pred, zero_division=0)),
            'roc_auc'  : float(roc_auc_score(ys, y_prob)),
            'confusion_matrix': confusion_matrix(ys, y_pred).tolist(),
        }
        # Simpan prediksi untuk plot di app
        res[f'{split_name}_y_true'] = ys.tolist()
        res[f'{split_name}_y_pred'] = y_pred.tolist()
        res[f'{split_name}_y_prob'] = y_prob.tolist()

    # Cetak tabel ringkasan
    print(f"\n  ── {name} ──")
    print(f"  {'Metric':<12} {'Train':>8} {'Val':>8} {'Test':>8}")
    print(f"  {'-'*38}")
    for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        print(f"  {m:<12} {res['train'][m]:>8.4f} {res['val'][m]:>8.4f} {res['test'][m]:>8.4f}")
    return res

models_dict = {
    'Logistic Regression': lr_model,
    'Random Forest'      : rf_model,
    'Gradient Boosting'  : gb_model,
}

all_results = {}
for mname, mmodel in models_dict.items():
    all_results[mname] = evaluate_model(
        mmodel,
        X_train_sc, y_train,
        X_val_sc,   y_val,
        X_test_sc,  y_test,
        mname,
    )

# Tentukan model terbaik berdasarkan F1 pada validation set
best_model_name = max(all_results, key=lambda k: all_results[k]['val']['f1'])
best_model      = models_dict[best_model_name]

print()
print(f"  [BEST] MODEL TERBAIK: {best_model_name}")
print(f"     F1 (val) : {all_results[best_model_name]['val']['f1']:.4f}")
print(f"     F1 (test): {all_results[best_model_name]['test']['f1']:.4f}")
print(f"     Alasan   : F1-Score tertinggi pada validation set.")
print(f"               Seimbang antara Precision & Recall — penting dalam")
print(f"               konteks medis untuk meminimalkan false negative.")

joblib.dump(best_model, MODEL_DIR / 'best_model.pkl')
print(f"\n  [OK] best_model.pkl disimpan")

# Simpan semua hasil evaluasi
summary = {
    'best_model'   : best_model_name,
    'feature_names': feature_cols,
    'dataset_info' : {
        'total'  : len(df),
        'train'  : len(X_train),
        'val'    : len(X_val),
        'test'   : len(X_test),
        'n_features': len(feature_cols),
        'class_0': int(n0),
        'class_1': int(n1),
    },
    'results': all_results,
}
with open(MODEL_DIR / 'results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("  [OK] results.json disimpan")

# Feature importances
fi_dict = {
    'logistic_regression': {
        'feature_names'  : feature_cols,
        'coefficients'   : lr_model.coef_[0].tolist(),
        'abs_coefficients': np.abs(lr_model.coef_[0]).tolist(),
    },
    'random_forest': {
        'feature_names': feature_cols,
        'importances'  : rf_model.feature_importances_.tolist(),
    },
    'gradient_boosting': {
        'feature_names': feature_cols,
        'importances'  : gb_model.feature_importances_.tolist(),
    },
}
with open(MODEL_DIR / 'feature_importances.json', 'w') as f:
    json.dump(fi_dict, f, indent=2)
print("  [OK] feature_importances.json disimpan")

# =============================================================================
# STEP 6 - GENERATE VISUALIZATIONS
# =============================================================================
print()
print("[STEP 6/7] GENERATING VISUALIZATIONS")
print("-" * 50)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
FIGDPI = 150

# ── Gambar 1: Distribusi Kelas ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Distribusi Kelas Target (Diabetes vs Non-Diabetes)',
             fontsize=14, fontweight='bold', y=1.02)

cnts  = [n0, n1]
clrs  = [PALETTE['primary'], PALETTE['danger']]
lbls  = ['Non-Diabetes (0)', 'Diabetes (1)']

axes[0].bar(lbls, cnts, color=clrs, edgecolor='white', linewidth=2)
for i, v in enumerate(cnts):
    axes[0].text(i, v + 5, f'{v} ({v/sum(cnts)*100:.1f}%)',
                 ha='center', fontweight='bold', fontsize=11)
axes[0].set_ylabel('Jumlah Sampel')
axes[0].set_title('Bar Chart')

axes[1].pie(cnts, labels=lbls, colors=clrs, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
axes[1].set_title('Pie Chart')

plt.tight_layout()
fig.savefig(FIGURES_DIR / '01_class_distribution.png', dpi=FIGDPI, bbox_inches='tight')
plt.close(fig)
print("  [OK] 01_class_distribution.png")

# ── Gambar 2: Distribusi Fitur per Kelas ───────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 13))
fig.suptitle('Distribusi Fitur Berdasarkan Kelas Target',
             fontsize=14, fontweight='bold')
axes_flat = axes.flatten()
orig_cols = COLUMNS[:-1]
for i, col in enumerate(orig_cols):
    ax = axes_flat[i]
    for ov, lbl, clr in [(0, 'Non-DM', PALETTE['primary']),
                          (1, 'Diabetes', PALETTE['danger'])]:
        ax.hist(df_raw[df_raw['Outcome'] == ov][col],
                bins=22, alpha=0.65, label=lbl, color=clr, edgecolor='none')
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frekuensi')
    ax.legend(fontsize=8)
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)
plt.tight_layout()
fig.savefig(FIGURES_DIR / '02_feature_distributions.png', dpi=FIGDPI, bbox_inches='tight')
plt.close(fig)
print("  [OK] 02_feature_distributions.png")

# ── Gambar 3: Correlation Heatmap ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
corr = df_raw.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, ax=ax, linewidths=0.5, square=True,
            cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Heatmap – Pima Indians Diabetes Dataset',
             fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
fig.savefig(FIGURES_DIR / '03_correlation_heatmap.png', dpi=FIGDPI, bbox_inches='tight')
plt.close(fig)
print("  [OK] 03_correlation_heatmap.png")

# ── Gambar 4: Boxplots per Kelas ───────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('Boxplot Fitur per Kelas (Analisis Outlier)',
             fontsize=13, fontweight='bold')
axes_flat = axes.flatten()
for i, col in enumerate(orig_cols):
    ax = axes_flat[i]
    data_plot = [df_raw[df_raw['Outcome'] == 0][col].values,
                 df_raw[df_raw['Outcome'] == 1][col].values]
    bp = ax.boxplot(data_plot, tick_labels=['Non-DM', 'Diabetes'],
                    patch_artist=True, medianprops=dict(color='black', lw=2))
    bp['boxes'][0].set_facecolor(PALETTE['primary'] + '99')
    bp['boxes'][1].set_facecolor(PALETTE['danger']  + '99')
    ax.set_title(col, fontweight='bold')
    ax.set_ylabel('Nilai')
plt.tight_layout()
fig.savefig(FIGURES_DIR / '04_boxplots.png', dpi=FIGDPI, bbox_inches='tight')
plt.close(fig)
print("  [OK] 04_boxplots.png")

# ── Gambar 5: Confusion Matrices ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Confusion Matrix – Semua Model (Test Set)',
             fontsize=13, fontweight='bold')
for ax, (mname, mres) in zip(axes, all_results.items()):
    cm = np.array(mres['test']['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, linewidths=1,
                xticklabels=['Non-DM', 'Diabetes'],
                yticklabels=['Non-DM', 'Diabetes'],
                cbar=False)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    f1v = mres['test']['f1']
    ax.set_title(f'{mname}\nF1 = {f1v:.4f}', fontweight='bold')
plt.tight_layout()
fig.savefig(FIGURES_DIR / '05_confusion_matrices.png', dpi=FIGDPI, bbox_inches='tight')
plt.close(fig)
print("  [OK] 05_confusion_matrices.png")

# ── Gambar 6: ROC Curves ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
roc_colors = [PALETTE['primary'], PALETTE['secondary'], PALETTE['accent']]
for (mname, mres), clr in zip(all_results.items(), roc_colors):
    yt = np.array(mres['test_y_true'])
    yp = np.array(mres['test_y_prob'])
    fpr, tpr, _ = roc_curve(yt, yp)
    auc_val = mres['test']['roc_auc']
    ax.plot(fpr, tpr, lw=2.5, color=clr,
            label=f'{mname}  (AUC = {auc_val:.4f})')
ax.plot([0,1],[0,1], 'k--', lw=1.2, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves – Perbandingan Semua Model', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
plt.tight_layout()
fig.savefig(FIGURES_DIR / '06_roc_curves.png', dpi=FIGDPI, bbox_inches='tight')
plt.close(fig)
print("  [OK] 06_roc_curves.png")

# ── Gambar 7: Feature Importance ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Feature Importance – Semua Model', fontsize=13, fontweight='bold')

# LR Coefficients
lr_coef    = lr_model.coef_[0]
lr_abs     = np.abs(lr_coef)
sorted_idx = np.argsort(lr_abs)
colors_lr  = [PALETTE['danger'] if lr_coef[i] > 0 else PALETTE['primary']
              for i in sorted_idx]
axes[0].barh(np.array(feature_cols)[sorted_idx], lr_abs[sorted_idx], color=colors_lr)
axes[0].set_title('Logistic Regression\n(|Koefisien|)', fontweight='bold')
axes[0].set_xlabel('Absolute Coefficient')

# RF Feature Importances
rf_imp     = rf_model.feature_importances_
sorted_idx = np.argsort(rf_imp)
axes[1].barh(np.array(feature_cols)[sorted_idx], rf_imp[sorted_idx],
             color=PALETTE['secondary'])
axes[1].set_title('Random Forest\n(Feature Importances)', fontweight='bold')
axes[1].set_xlabel('Importance Score')

# GB Feature Importances
gb_imp     = gb_model.feature_importances_
sorted_idx = np.argsort(gb_imp)
axes[2].barh(np.array(feature_cols)[sorted_idx], gb_imp[sorted_idx],
             color=PALETTE['accent'])
axes[2].set_title('Gradient Boosting\n(Feature Importances)', fontweight='bold')
axes[2].set_xlabel('Importance Score')

plt.tight_layout()
fig.savefig(FIGURES_DIR / '07_feature_importance.png', dpi=FIGDPI, bbox_inches='tight')
plt.close(fig)
print("  [OK] 07_feature_importance.png")

# ── Gambar 8: Perbandingan Metrik ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
metrics_list   = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
x      = np.arange(len(metrics_list))
width  = 0.25

for i, (mname, clr) in enumerate(zip(all_results, roc_colors)):
    vals = [all_results[mname]['test'][m] for m in metrics_list]
    bars = ax.bar(x + i*width, vals, width, label=mname, color=clr, alpha=0.82,
                  edgecolor='white', linewidth=1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax.set_xticks(x + width)
ax.set_xticklabels(metrics_labels, fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Perbandingan Performa Model pada Test Set', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, 1.12)
ax.yaxis.grid(True, alpha=0.4)
plt.tight_layout()
fig.savefig(FIGURES_DIR / '08_model_comparison.png', dpi=FIGDPI, bbox_inches='tight')
plt.close(fig)
print("  [OK] 08_model_comparison.png")

# =============================================================================
# STEP 7 - RINGKASAN AKHIR
# =============================================================================
print()
print("[STEP 7/7] RINGKASAN & PENUTUP")
print("-" * 50)

bres = all_results[best_model_name]
print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║            HASIL TRAINING SELESAI                   ║
  ╠══════════════════════════════════════════════════════╣
  ║  [BEST] Model Terbaik : {best_model_name:<34}║
  ╠══════════════════════════════════════════════════════╣
  ║  Metrik          │ Validation │   Test                ║
  ║  ─────────────── │ ─────────  │ ─────────             ║
  ║  Accuracy        │  {bres['val']['accuracy']:7.4f}   │  {bres['test']['accuracy']:7.4f}         ║
  ║  Precision       │  {bres['val']['precision']:7.4f}   │  {bres['test']['precision']:7.4f}         ║
  ║  Recall          │  {bres['val']['recall']:7.4f}   │  {bres['test']['recall']:7.4f}         ║
  ║  F1-Score        │  {bres['val']['f1']:7.4f}   │  {bres['test']['f1']:7.4f}         ║
  ║  AUC-ROC         │  {bres['val']['roc_auc']:7.4f}   │  {bres['test']['roc_auc']:7.4f}         ║
  ╚══════════════════════════════════════════════════════╝
""")

print("  [FILES] File yang tersimpan:")
files_saved = [
    ("models/", "logistic_regression.pkl"),
    ("models/", "random_forest.pkl"),
    ("models/", "gradient_boosting.pkl"),
    ("models/", "best_model.pkl"),
    ("models/", "scaler.pkl"),
    ("models/", "results.json"),
    ("models/", "feature_importances.json"),
    ("models/", "feature_info.json"),
    ("data/",   "diabetes.csv"),
    ("data/",   "diabetes_processed.csv"),
    ("reports/figures/", "01–08 *.png (8 grafik)"),
]
for folder, fname in files_saved:
    print(f"    [OK] {folder}{fname}")

print()
print("  [RUN] Langkah selanjutnya:")
print("     streamlit run app.py")
print()
print(SEP)
print("  Pipeline selesai!")
print(SEP)
