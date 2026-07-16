#!/usr/bin/env python3
"""
predict.py - CLI inference tool untuk prediksi diabetes
UAS Pembelajaran Mesin - UDINUS - 2026

Kelompok:
  - Fahmi Fatmawati Azzahra  (NIM: A11.2024.15831)
  - Nathaniela Febry Nathasa (NIM: A11.2024.15850)

Penggunaan:
  python predict.py --pregnancies 3 --glucose 120 --bp 70 --skin 23 \
                    --insulin 79 --bmi 32.0 --dpf 0.471 --age 33
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR / 'models'

def build_input(preg, gluc, bp, skin, insulin, bmi, dpf, age, feat_names, scaler):
    d = {
        'Pregnancies'            : preg,
        'Glucose'                : gluc,
        'BloodPressure'          : bp,
        'SkinThickness'          : skin,
        'Insulin'                : insulin,
        'BMI'                    : bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age'                    : age,
        'BMI_Category'           : (0 if bmi<18.5 else 1 if bmi<25 else 2 if bmi<30 else 3),
        'Glucose_Category'       : (0 if gluc<100 else 1 if gluc<126 else 2),
        'Age_Group'              : (0 if age<35 else 1 if age<50 else 2),
        'Insulin_Glucose_Ratio'  : insulin / (gluc + 1.0),
    }
    df = pd.DataFrame([d])
    if feat_names:
        df = df[feat_names]
    return scaler.transform(df)

def main():
    parser = argparse.ArgumentParser(
        description="Prediksi risiko diabetes dari data kesehatan pasien",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--pregnancies', type=int,   default=3,     help="Jumlah kehamilan")
    parser.add_argument('--glucose',     type=int,   default=120,   help="Kadar glukosa (mg/dL)")
    parser.add_argument('--bp',          type=int,   default=70,    help="Tekanan darah diastolik (mmHg)")
    parser.add_argument('--skin',        type=int,   default=23,    help="Ketebalan kulit tricep (mm)")
    parser.add_argument('--insulin',     type=int,   default=79,    help="Insulin serum 2 jam (mu U/ml)")
    parser.add_argument('--bmi',         type=float, default=32.0,  help="BMI (kg/m²)")
    parser.add_argument('--dpf',         type=float, default=0.471, help="Diabetes Pedigree Function")
    parser.add_argument('--age',         type=int,   default=33,    help="Usia (tahun)")
    parser.add_argument('--model',       type=str,   default='best',
                        choices=['best','lr','rf','gb'],
                        help="Model yang digunakan (best/lr/rf/gb)")
    args = parser.parse_args()

    # Load model
    model_map = {
        'best': 'best_model.pkl',
        'lr'  : 'logistic_regression.pkl',
        'rf'  : 'random_forest.pkl',
        'gb'  : 'gradient_boosting.pkl',
    }
    model_path = MODEL_DIR / model_map[args.model]
    scaler_path = MODEL_DIR / 'scaler.pkl'
    fi_path = MODEL_DIR / 'feature_info.json'

    if not model_path.exists():
        print("ERROR: Model tidak ditemukan. Jalankan: python train.py")
        sys.exit(1)

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feat_names = None
    if fi_path.exists():
        feat_names = json.loads(fi_path.read_text())['feature_names']

    # Build input
    X = build_input(args.pregnancies, args.glucose, args.bp, args.skin,
                    args.insulin, args.bmi, args.dpf, args.age,
                    feat_names, scaler)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    dm_prob  = float(prob[1])
    ndm_prob = float(prob[0])

    # Output
    print()
    print("=" * 50)
    print("  HASIL PREDIKSI DIABETES")
    print("=" * 50)
    print(f"  Pregnancies : {args.pregnancies}")
    print(f"  Glucose     : {args.glucose} mg/dL")
    print(f"  BloodPressure: {args.bp} mmHg")
    print(f"  Insulin     : {args.insulin} mu U/ml")
    print(f"  BMI         : {args.bmi} kg/m²")
    print(f"  Age         : {args.age} tahun")
    print("-" * 50)
    print(f"  Model       : {args.model.upper()}")
    if pred == 1:
        risk = "TINGGI" if dm_prob >= 0.7 else "SEDANG"
        print(f"  ⚠️  PREDIKSI  : DIABETES TERDETEKSI")
        print(f"  Tingkat Risiko : {risk}")
    else:
        print(f"  ✅ PREDIKSI  : TIDAK DIABETES")
        print(f"  Tingkat Risiko : RENDAH")
    print(f"  Prob. Diabetes  : {dm_prob*100:.2f}%")
    print(f"  Prob. Non-DM    : {ndm_prob*100:.2f}%")
    print("=" * 50)
    print()
    print("⚠️  DISCLAIMER: Prediksi ini bersifat indikatif dan TIDAK menggantikan")
    print("   diagnosis medis. Selalu konsultasikan dengan dokter.")
    print()

if __name__ == "__main__":
    main()
