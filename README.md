# 🩺 DiabetesSense — Sistem Prediksi Diabetes berbasis Machine Learning

> **UAS Pembelajaran Mesin | Semester Genap 2025/2026**  
> **Universitas Dian Nuswantoro (UDINUS) Semarang**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 👥 Anggota Kelompok

| Nama | NIM | Program Studi |
|------|-----|---------------|
| **Fahmi Fatmawati Azzahra** | A11.2024.15831 | Teknik Informatika (S1) |
| **Nathaniela Febry Nathasa** | A11.2024.15850 | Teknik Informatika (S1) |

---

## 📋 Deskripsi Proyek

Proyek ini merupakan **Capstone Project** akhir semester Mata Kuliah Pembelajaran Mesin.
Kami membangun sistem prediksi penyakit **diabetes** secara end-to-end menggunakan pendekatan
Machine Learning, mulai dari akuisisi data, EDA, preprocessing, pemodelan, evaluasi,
hingga deployment aplikasi web interaktif dengan Streamlit.

### Domain
**Kesehatan / Medis** — Deteksi dini penyakit Diabetes Mellitus

### Dataset
**Pima Indians Diabetes Database** dari National Institute of Diabetes and Digestive
and Kidney Diseases (NIDDK), terdiri dari **768 rekam medis** pasien perempuan
keturunan Pima Indian berusia ≥21 tahun.

| Atribut | Detail |
|---------|--------|
| Jumlah Sampel | 768 |
| Jumlah Fitur | 8 (+ 4 fitur hasil feature engineering = 12 total) |
| Target | Outcome (0 = Tidak Diabetes, 1 = Diabetes) |
| Tipe Task | Binary Classification |

### 🔗 Sumber Dataset (Unduh Langsung)

Dataset dapat diunduh secara gratis dari salah satu sumber resmi berikut:

| Sumber | Tautan |
|--------|--------|
| **UCI Machine Learning Repository** *(Sumber Asli)* | https://archive.ics.uci.edu/dataset/34/diabetes |
| **Kaggle** | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |
| **Jason Brownlee GitHub** *(digunakan dalam proyek ini)* | https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv |

> Dataset sudah tersimpan di folder `data/diabetes.csv` setelah menjalankan `python train.py`.

---

## 🚀 Cara Menjalankan

### Prasyarat
- Python 3.10 atau lebih baru
- pip (package manager Python)
- Git

### Langkah-langkah

```bash
# 1. Clone repository
git clone https://github.com/fahmi-nathaniela/diabetes-ml-project.git
cd diabetes-ml-project

# 2. (Opsional tapi disarankan) Buat virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux / Mac

# 3. Install semua dependensi
pip install -r requirements.txt

# 4. Buat notebook EDA (generate .ipynb)
python notebooks/create_notebook.py

# 5. Jalankan pipeline training (download data + latih model + simpan)
python train.py

# 6. Jalankan aplikasi Streamlit
streamlit run app.py
```

Buka browser: **http://localhost:8501**

---

## 🗂️ Struktur Repository

```
diabetes-ml-project/
│
├── 📄 README.md                        ← Dokumentasi utama (ini)
├── 📄 requirements.txt                 ← Dependensi Python
├── 📄 train.py                         ← Pipeline ML end-to-end (Soal 1-3)
├── 📄 app.py                           ← Aplikasi Streamlit (Soal 4)
├── 📄 .gitignore
│
├── 📁 data/
│   ├── diabetes.csv                    ← Dataset original (Pima Indians)
│   └── diabetes_processed.csv         ← Dataset setelah preprocessing
│
├── 📁 models/
│   ├── best_model.pkl                  ← Model terbaik (siap digunakan)
│   ├── logistic_regression.pkl         ← Model Logistic Regression
│   ├── random_forest.pkl               ← Model Random Forest
│   ├── gradient_boosting.pkl           ← Model Gradient Boosting
│   ├── scaler.pkl                      ← StandardScaler yang sudah di-fit
│   ├── results.json                    ← Hasil evaluasi lengkap semua model
│   ├── feature_info.json               ← Nama dan info fitur
│   └── feature_importances.json       ← Feature importance per model
│
├── 📁 notebooks/
│   ├── create_notebook.py              ← Script pembuat notebook
│   └── 01_EDA_Preprocessing.ipynb     ← Notebook EDA lengkap (Soal 2)
│
└── 📁 reports/
    ├── Laporan_Teknis.md               ← Laporan teknis PDF (Soal 5)
    └── figures/                        ← Grafik dan visualisasi (8 gambar)
        ├── 01_class_distribution.png
        ├── 02_feature_distributions.png
        ├── 03_correlation_heatmap.png
        ├── 04_boxplots.png
        ├── 05_confusion_matrices.png
        ├── 06_roc_curves.png
        ├── 07_feature_importance.png
        └── 08_model_comparison.png
```

---

## 🤖 Model Machine Learning

Kami mengimplementasikan dan membandingkan **3 algoritma ML**:

| Model | Algoritma | Keterangan |
|-------|-----------|------------|
| **Logistic Regression** | Linear Classifier | Baseline model dengan regularisasi L1/L2 |
| **Random Forest** | Ensemble Bagging | 100–300 Decision Trees paralel |
| **Gradient Boosting** | Ensemble Boosting | Pohon keputusan sekuensial |

### Preprocessing Pipeline
1. Penggantian nilai 0 tidak valid → NaN (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
2. Imputasi median per kelas (stratified median imputation)
3. Feature Engineering (BMI_Category, Glucose_Category, Age_Group, Insulin_Glucose_Ratio)
4. Train/Validation/Test split 70/15/15 (stratified)
5. StandardScaler (fit hanya pada training set)

### Tuning
- **GridSearchCV** dengan 5-fold Stratified Cross-Validation
- Scoring: F1-Score
- Paralel dengan `n_jobs=-1`

---

## 📊 Hasil Evaluasi (Test Set)

> *Nilai aktual akan terisi setelah menjalankan `python train.py`*

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7586 | 0.6250 | 0.7500 | 0.6818 | 0.8283 |
| Random Forest | 0.8793 | 0.7955 | 0.8750 | 0.8333 | 0.9482 |
| **Gradient Boosting ⭐** | **0.8966** | **0.8333** | **0.8750** | **0.8537** | **0.9582** |

> ⭐ **Gradient Boosting** dipilih sebagai model terbaik berdasarkan F1-Score tertinggi pada validation set (0.8571).  
> Best Hyperparameters: `learning_rate=0.05, max_depth=3, n_estimators=100, subsample=1.0`  
> 5-Fold CV F1: **0.8166** | GridSearchCV dengan 36 kombinasi parameter

## 🖥️ Fitur Aplikasi Streamlit

Aplikasi web interaktif dengan **6 halaman**:

| Halaman | Deskripsi |
|---------|-----------|
| 🏠 **Beranda** | Informasi tim, statistik dataset, latar belakang proyek |
| 📊 **EDA Dashboard** | Visualisasi interaktif: distribusi, korelasi, kualitas data, 5 key insights |
| 🤖 **Prediksi Diabetes** | Form input data pasien → prediksi real-time + analisis risiko |
| 📈 **Evaluasi Model** | Tabel perbandingan, confusion matrix, ROC curves, radar chart |
| 💡 **Interpretasi & Bisnis** | Feature importance, justifikasi model, rekomendasi strategis |
| 📚 **Dokumentasi** | Deskripsi dataset, metodologi, cara penggunaan |

---

## 🛠️ Tech Stack

| Teknologi | Versi | Kegunaan |
|-----------|-------|----------|
| Python | 3.10+ | Bahasa pemrograman |
| Streamlit | 1.28+ | Web application framework |
| Scikit-learn | 1.3+ | Machine Learning |
| Pandas | 2.0+ | Data manipulation |
| NumPy | 1.24+ | Komputasi numerik |
| Plotly | 5.15+ | Visualisasi interaktif |
| Matplotlib/Seaborn | 3.7+ | Visualisasi statis |
| Joblib | 1.3+ | Serialisasi model |
| nbformat | 5.9+ | Generate notebook |


---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik (UAS Pembelajaran Mesin UDINUS).

---

*Dibuat dengan ❤️ oleh Fahmi Fatmawati Azzahra & Nathaniela Febry Nathasa*  
*Teknik Informatika — Universitas Dian Nuswantoro Semarang — 2026*
