#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/utils.py - Fungsi utilitas untuk pipeline ML
UAS Pembelajaran Mesin - UDINUS Semarang - 2026

Kelompok:
  - Fahmi Fatmawati Azzahra  (NIM: A11.2024.15831)
  - Nathaniela Febry Nathasa (NIM: A11.2024.15850)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    classification_report
)

# Warna konsisten di seluruh project
PALETTE = {
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


def load_model(model_path: str):
    """Memuat model dari file .pkl."""
    return joblib.load(model_path)


def load_scaler(scaler_path: str):
    """Memuat scaler dari file .pkl."""
    return joblib.load(scaler_path)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict:
    """Menghitung semua metrik evaluasi untuk satu split."""
    return {
        'accuracy' : float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall'   : float(recall_score(y_true, y_pred, zero_division=0)),
        'f1'       : float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc'  : float(roc_auc_score(y_true, y_prob)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred,
                                                        target_names=['Non-DM','Diabetes'],
                                                        zero_division=0),
    }


def print_metrics_table(metrics: dict, model_name: str, split: str = "Test"):
    """Mencetak tabel metrik yang rapi."""
    print(f"\n  ── {model_name} ({split} Set) ──")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:<15}: {v:.4f}")


def plot_roc_curves(models_results: dict, save_path: str = None):
    """Plot ROC Curves untuk semua model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for mname, mres in models_results.items():
        yt = np.array(mres['test_y_true'])
        yp = np.array(mres['test_y_prob'])
        fpr, tpr, _ = roc_curve(yt, yp)
        auc_val = auc(fpr, tpr)
        clr = MODEL_COLORS.get(mname, '#333')
        ax.plot(fpr, tpr, lw=2.5, color=clr,
                label=f'{mname}  (AUC = {auc_val:.4f})')

    ax.plot([0,1],[0,1], 'k--', lw=1.2, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves – Perbandingan Model', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1.05])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_confusion_matrix(cm: np.ndarray, title: str = "", save_path: str = None):
    """Plot confusion matrix sebagai heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, linewidths=1,
                xticklabels=['Non-DM', 'Diabetes'],
                yticklabels=['Non-DM', 'Diabetes'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def feature_importance_table(feature_names: list, importances: np.ndarray,
                              top_n: int = 10) -> pd.DataFrame:
    """Membuat tabel feature importance yang terurut."""
    fi_df = pd.DataFrame({
        'Feature'   : feature_names,
        'Importance': importances,
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    fi_df['Rank'] = fi_df.index + 1
    return fi_df.head(top_n)[['Rank','Feature','Importance']]


def load_results_summary(results_path: str) -> dict:
    """Memuat summary hasil evaluasi dari JSON."""
    p = Path(results_path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def print_banner():
    """Print banner informasi tim."""
    sep = "=" * 65
    print(sep)
    print("  UAS Pembelajaran Mesin — Semester Genap 2025/2026")
    print("  Universitas Dian Nuswantoro (UDINUS) Semarang")
    print()
    print("  Kelompok:")
    print("    • Fahmi Fatmawati Azzahra  — NIM: A11.2024.15831")
    print("    • Nathaniela Febry Nathasa — NIM: A11.2024.15850")
    print(sep)
    print()
