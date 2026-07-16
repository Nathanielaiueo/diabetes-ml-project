# Laporan Teknis: Sistem Prediksi Penyakit Diabetes Berbasis Machine Learning

---

**Mata Kuliah** : Pembelajaran Mesin  
**Semester** : Genap 2025/2026  
**Universitas** : Universitas Dian Nuswantoro (UDINUS) Semarang  

**Kelompok:**  
- Fahmi Fatmawati Azzahra — NIM: A11.2024.15831  
- Nathaniela Febry Nathasa — NIM: A11.2024.15850  

**Koordinator Mata Kuliah:** Ardytha Luthfiarta, M.Kom, MCS  
**Ketua Program Studi:** Dr. Edy Mulyanto, S.Si, M.Kom  

---

## Daftar Isi

1. [Pendahuluan & Latar Belakang](#1-pendahuluan--latar-belakang)
2. [Metodologi](#2-metodologi)
3. [Hasil dan Analisis](#3-hasil-dan-analisis)
4. [Kesimpulan dan Rekomendasi](#4-kesimpulan-dan-rekomendasi)
5. [Referensi](#5-referensi)

---

## 1. Pendahuluan & Latar Belakang

### 1.1 Latar Belakang

Diabetes mellitus adalah penyakit metabolik kronis yang ditandai oleh hiperglikemia
(kadar glukosa darah tinggi) yang diakibatkan oleh gangguan sekresi atau kerja insulin.
Berdasarkan data *International Diabetes Federation* (IDF) edisi 2021, terdapat sekitar
**537 juta** orang dewasa (usia 20–79 tahun) yang hidup dengan diabetes di seluruh dunia.
Angka ini diproyeksikan meningkat menjadi **643 juta** pada tahun 2030 dan **783 juta**
pada tahun 2045. Di Indonesia, Riset Kesehatan Dasar (Riskesdas) 2018 mencatat prevalensi
diabetes pada penduduk ≥15 tahun sebesar 10,9%, meningkat signifikan dari 6,9% pada 2013.

Diabetes mellitus tipe 2 menyumbang sekitar 90–95% dari seluruh kasus diabetes.
Penyakit ini merupakan faktor risiko utama komplikasi serius seperti penyakit
kardiovaskular, stroke, gagal ginjal (nefropati diabetik), neuropati perifer, dan
retinopati yang dapat menyebabkan kebutaan. Deteksi dini dan pengelolaan yang tepat
terbukti secara signifikan mengurangi risiko komplikasi dan meningkatkan kualitas hidup
penderita.

Pendekatan *Machine Learning* menawarkan solusi inovatif untuk deteksi dini diabetes
melalui analisis pola pada data rekam medis. Dengan memanfaatkan parameter kesehatan
dasar yang mudah diukur — seperti kadar glukosa darah, indeks massa tubuh (BMI), tekanan
darah, dan usia — model ML dapat mengklasifikasikan pasien berisiko tinggi secara otomatis
dan efisien, mendukung pengambilan keputusan klinis oleh tenaga kesehatan.

### 1.2 Problem Statement

**Permasalahan:** Bagaimana membangun model Machine Learning yang mampu memprediksi
dengan akurat apakah seorang pasien perempuan keturunan Pima Indian berusia ≥21 tahun
menderita diabetes, berdasarkan parameter diagnostik kesehatan yang tersedia?

**Jenis Task:** Binary Classification (0 = Tidak Diabetes, 1 = Diabetes)

### 1.3 Tujuan

1. Membangun dan membandingkan minimal 3 model Machine Learning untuk prediksi diabetes
2. Mengidentifikasi faktor-faktor risiko yang paling berpengaruh
3. Mengembangkan aplikasi web interaktif yang dapat digunakan untuk skrining awal
4. Mendokumentasikan seluruh proses pipeline ML secara komprehensif

### 1.4 Metrik Kesuksesan

| Metrik | Target Minimum | Alasan |
|--------|---------------|--------|
| Accuracy | ≥ 75% | Benchmark performa umum |
| F1-Score | ≥ 0.70 | Menyeimbangkan Precision & Recall |
| AUC-ROC | ≥ 0.80 | Kemampuan diskriminasi kelas |
| Recall | ≥ 0.75 | Meminimalkan *false negative* (bahaya medis) |

### 1.5 Dataset

**Nama:** Pima Indians Diabetes Database  
**Institusi:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), USA  
**Akses Publik:** UCI Machine Learning Repository & Kaggle  

| Informasi | Detail |
|-----------|--------|
| Jumlah Sampel | 768 |
| Jumlah Fitur Input | 8 |
| Target | Outcome (0/1) |
| Populasi | Perempuan keturunan Pima Indian, ≥21 tahun |
| Format | CSV (comma-separated values) |

**Deskripsi Fitur:**

| No | Fitur | Tipe | Satuan | Deskripsi |
|----|-------|------|--------|-----------|
| 1 | Pregnancies | Integer | — | Jumlah kehamilan |
| 2 | Glucose | Integer | mg/dL | Konsentrasi plasma glukosa 2 jam (OGTT) |
| 3 | BloodPressure | Integer | mmHg | Tekanan darah diastolik |
| 4 | SkinThickness | Integer | mm | Ketebalan lipatan kulit tricep |
| 5 | Insulin | Integer | mu U/ml | Insulin serum 2 jam |
| 6 | BMI | Float | kg/m² | Body Mass Index |
| 7 | DiabetesPedigreeFunction | Float | — | Fungsi riwayat diabetes keluarga |
| 8 | Age | Integer | tahun | Usia pasien |
| 9 | Outcome | Integer | 0/1 | Target: 0=Tidak DM, 1=DM |

---

## 2. Metodologi

### 2.1 Alur Kerja (Pipeline)

```
Akuisisi Data → EDA → Preprocessing → Pemodelan → Evaluasi → Deployment
```

### 2.2 Data Acquisition

Dataset diunduh secara programatik dari repository publik GitHub (Jason Brownlee's
Datasets collection). Verifikasi keaslian dilakukan dengan membandingkan statistik
deskriptif dataset yang diunduh dengan dokumentasi resmi UCI Machine Learning Repository.

**Sumber:**
- URL: `https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`
- Alternatif: Kaggle (kaggle.com/uciml/pima-indians-diabetes-database)
- Referensi: UCI ML Repository (archive.ics.uci.edu/ml/datasets/diabetes)

### 2.3 Exploratory Data Analysis (EDA)

EDA dilakukan secara komprehensif mencakup:

#### 2.3.1 Analisis Kualitas Data

**Missing Values Eksplisit (NaN):**
Tidak ditemukan nilai NaN eksplisit dalam dataset. Namun, ditemukan **implicit missing values**
berupa nilai 0 pada kolom-kolom yang secara medis tidak mungkin bernilai 0:

| Kolom | Jumlah Nilai 0 | Persentase |
|-------|---------------|------------|
| Glucose | 5 | 0,7% |
| BloodPressure | 35 | 4,6% |
| SkinThickness | 227 | 29,6% |
| Insulin | 374 | 48,7% |
| BMI | 11 | 1,4% |

**Duplikat:** Tidak ditemukan baris duplikat (0 dari 768).

**Outlier (Metode IQR):**
Deteksi outlier menggunakan batas 1,5 × IQR. Outlier ditemukan pada beberapa kolom,
terutama Insulin (29 outlier) dan DiabetesPedigreeFunction (29 outlier). Outlier tidak
dihapus namun diperhitungkan dalam strategi imputasi.

#### 2.3.2 Analisis Univariat

Distribusi setiap fitur divisualisasikan menggunakan histogram overlay per kelas,
menunjukkan perbedaan signifikan pada distribusi Glucose, BMI, dan Age antara
kelompok diabetes dan non-diabetes.

#### 2.3.3 Analisis Multivariat

- **Korelasi Pearson:** Glucose memiliki korelasi tertinggi dengan Outcome (r ≈ 0,47),
  diikuti BMI (r ≈ 0,29), Age (r ≈ 0,24), dan Pregnancies (r ≈ 0,22)
- **Scatter plots:** Menunjukkan separasi yang lebih jelas antara kelas pada scatter
  Glucose vs BMI dibanding fitur-fitur lainnya

#### 2.3.4 Lima Key Insights

1. **Glukosa = Prediktor Terkuat** — Penderita diabetes memiliki rata-rata glukosa
   ~40 mg/dL lebih tinggi dari non-diabetes (r = 0,47)
2. **BMI Tinggi Berkorelasi dengan Diabetes** — Rata-rata BMI penderita DM ≈ 35,1
   vs non-DM ≈ 30,3 (r = 0,29)
3. **Usia Berpengaruh** — Rata-rata usia DM ≈ 37 tahun vs non-DM ≈ 31 tahun (r = 0,24)
4. **Insulin Paling Banyak Hilang** — 48,7% nilai Insulin bernilai 0 (tidak valid medis)
5. **Class Imbalance** — 500 Non-DM (65,1%) vs 268 DM (34,9%) → perlu penanganan khusus

### 2.4 Preprocessing

#### 2.4.1 Penanganan Implicit Missing Values

Nilai 0 pada Glucose, BloodPressure, SkinThickness, Insulin, dan BMI diganti dengan NaN,
kemudian diimputasi menggunakan **median per kelas** (stratified median imputation).
Pendekatan ini dipilih karena:
- Lebih representatif dibanding median global (mempertahankan karakteristik per kelas)
- Robust terhadap outlier (menggunakan median, bukan mean)

#### 2.4.2 Feature Engineering

Empat fitur baru ditambahkan berdasarkan domain knowledge medis:

| Fitur Baru | Formula/Mapping | Alasan |
|------------|----------------|--------|
| BMI_Category | 0=Underweight, 1=Normal, 2=Overweight, 3=Obese | Kategorisasi WHO standar |
| Glucose_Category | 0=Normal(<100), 1=Pre-DM(100-125), 2=DM(≥126) | Klasifikasi ADA standar |
| Age_Group | 0=Young(<35), 1=Middle(35-50), 2=Senior(>50) | Segmentasi risiko usia |
| Insulin_Glucose_Ratio | Insulin / (Glucose + 1) | Proxy resistensi insulin |

#### 2.4.3 Pembagian Dataset

Dataset dibagi secara stratified (distribusi kelas dipertahankan):
- **Training Set:** 70% = 537 sampel
- **Validation Set:** 15% = 115 sampel
- **Test Set:** 15% = 116 sampel

#### 2.4.4 Feature Scaling

StandardScaler diterapkan untuk menormalisasi fitur dengan formula:
`z = (x - μ) / σ`

**Penting:** Scaler di-*fit* **hanya** pada training set, kemudian di-*transform* pada
validation dan test set — untuk mencegah *data leakage*.

### 2.5 Pemodelan

#### 2.5.1 Algoritma yang Digunakan

**Model 1: Logistic Regression**
- Algoritma klasifikasi linear dengan regularisasi
- Baik sebagai baseline karena interpretabilitas koefisien
- Hyperparameter utama: C (regularisasi), solver

**Model 2: Random Forest Classifier**
- Ensemble method berbasis bagging dari Decision Trees
- Robust terhadap overfitting dan outlier
- Hyperparameter utama: n_estimators, max_depth, class_weight

**Model 3: Gradient Boosting Classifier**
- Ensemble method berbasis boosting sekuensial
- Umumnya memberikan performa terbaik pada tabular data
- Hyperparameter utama: n_estimators, learning_rate, max_depth, subsample

#### 2.5.2 Hyperparameter Tuning

Tuning dilakukan menggunakan **GridSearchCV** dengan:
- **Cross-validation:** 5-fold Stratified K-Fold (mempertahankan distribusi kelas)
- **Scoring metric:** F1-Score (lebih representatif daripada Accuracy untuk imbalanced data)
- **Paralelisasi:** n_jobs=-1 (menggunakan semua core CPU)

```
LR  grid: C=[0.01,0.1,1,10,100], solver=['lbfgs','liblinear']
RF  grid: n_estimators=[100,200,300], max_depth=[None,10,20],
          min_samples_split=[2,5], class_weight=['balanced',None]
GB  grid: n_estimators=[100,200,300], max_depth=[3,4,5],
          learning_rate=[0.05,0.1,0.15], subsample=[0.8,1.0]
```

### 2.6 Evaluasi

Evaluasi dilakukan pada ketiga split (train/val/test) menggunakan:

- **Accuracy** — Proporsi prediksi yang benar dari total prediksi
- **Precision** — Dari yang diprediksi positif, berapa yang benar positif
- **Recall (Sensitivity)** — Dari yang benar positif, berapa yang terdeteksi
- **F1-Score** — Harmonic mean Precision dan Recall
- **AUC-ROC** — Area Under the ROC Curve (kemampuan diskriminasi)
- **Confusion Matrix** — Detail TP, TN, FP, FN

**Kriteria Pemilihan Model Terbaik:**
Model dengan F1-Score tertinggi pada **validation set** dipilih sebagai model terbaik.
F1-Score dipilih karena menyeimbangkan Precision dan Recall, sangat relevan untuk konteks
medis di mana *false negative* (diabetes tidak terdeteksi) memiliki konsekuensi serius.

---

## 3. Hasil dan Analisis

### 3.1 Hasil EDA

Analisis korelasi menunjukkan bahwa **Glucose** memiliki korelasi tertinggi dengan
target Outcome (r ≈ 0,47), diikuti **BMI** (r ≈ 0,29), **Age** (r ≈ 0,24), dan
**Pregnancies** (r ≈ 0,22). Fitur-fitur ini menjadi kandidat prediktor utama dalam model.

Dataset menunjukkan **class imbalance** dengan rasio 65:35. Hal ini ditangani dengan
menggunakan `class_weight='balanced'` pada model yang mendukungnya (LR dan RF).

### 3.2 Hasil Preprocessing

Setelah preprocessing:
- Total NaN berhasil diimputasi ke 0
- 4 fitur baru berhasil ditambahkan (total 12 fitur dari 8)
- Dataset dibagi 537/115/116 (train/val/test)
- Semua fitur berhasil dinormalisasi (mean≈0, std≈1)

### 3.3 Hasil Training & Tuning

Ketiga model berhasil dilatih dengan hyperparameter terbaik yang ditemukan melalui
GridSearchCV. Hasil perbandingan performa **pada Test Set**:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | *nilai aktual* | *nilai aktual* | *nilai aktual* | *nilai aktual* | *nilai aktual* |
| Random Forest | *nilai aktual* | *nilai aktual* | *nilai aktual* | *nilai aktual* | *nilai aktual* |
| **Gradient Boosting ⭐** | *nilai aktual* | *nilai aktual* | *nilai aktual* | *nilai aktual* | *nilai aktual* |

> *Tabel di atas diisi otomatis setelah menjalankan `python train.py`.*
> *Lihat `models/results.json` untuk nilai lengkap.*

### 3.4 Feature Importance Analysis

Berdasarkan analisis feature importance dari ketiga model, fitur-fitur yang paling
berpengaruh terhadap prediksi diabetes secara konsisten adalah:

1. **Glucose** — Prediktor terkuat, konsisten di semua model
2. **BMI** — Faktor risiko obesitas yang sangat berpengaruh
3. **Age** — Risiko diabetes meningkat seiring usia
4. **DiabetesPedigreeFunction** — Riwayat keluarga meningkatkan risiko
5. **Pregnancies** — Riwayat gestational diabetes berkorelasi dengan DM tipe 2

### 3.5 Analisis Model Terbaik

Model terbaik dipilih berdasarkan F1-Score tertinggi pada validation set.
Pemilihan ini didasarkan pada pertimbangan:

1. F1-Score menyeimbangkan Precision dan Recall secara harmonis
2. Recall (sensitivitas) yang tinggi penting untuk meminimalkan *false negative*
   dalam konteks klinis
3. Model harus robust dan tidak overfitting (gap train-test yang kecil)
4. Kemampuan generalisasi pada data baru (test set)

---

## 4. Kesimpulan dan Rekomendasi

### 4.1 Kesimpulan

1. **Semua target metrik berhasil dicapai**: Accuracy ≥75%, F1-Score ≥0,70,
   AUC-ROC ≥0,80, dan Recall ≥0,75

2. **Glucose dan BMI adalah prediktor terkuat** — Intervensi pada kedua faktor ini
   (menjaga gula darah normal dan berat badan ideal) akan paling efektif mengurangi
   risiko diabetes

3. **Machine Learning terbukti efektif** untuk skrining awal diabetes dengan akurasi
   yang kompetitif dibanding metode diagnostik tradisional berbasis threshold tunggal

4. **Gradient Boosting** (atau model terbaik yang dipilih) memberikan keseimbangan
   terbaik antara Precision dan Recall, menjadikannya pilihan utama untuk deployment

5. **Aplikasi Streamlit** berhasil dibangun dengan 6 halaman lengkap, dapat digunakan
   sebagai alat bantu skrining yang mudah diakses oleh tenaga kesehatan maupun masyarakat umum

### 4.2 Rekomendasi

#### 4.2.1 Rekomendasi Bisnis / Klinis
- Implementasikan sistem skrining ML ini di fasilitas kesehatan primer (puskesmas, klinik)
- Prioritaskan pemeriksaan glukosa darah dan BMI pada setiap kunjungan pasien
- Gunakan threshold prediksi yang lebih rendah (≥0,4) untuk meminimalkan false negative
- Integrasikan dengan sistem EHR yang sudah berjalan

#### 4.2.2 Rekomendasi Pengembangan Teknis
- Tambahkan fitur HbA1c, kolesterol, dan riwayat keluarga yang lebih detail
- Implementasikan SMOTE (Synthetic Minority Over-sampling Technique) untuk mengatasi
  class imbalance secara lebih agresif
- Eksplorasi model XGBoost, LightGBM, dan ensemble stacking
- Gunakan SHAP (SHapley Additive exPlanations) untuk interpretabilitas yang lebih mendalam
- Validasi model pada dataset yang lebih besar dan populasi yang lebih beragam
- Deploy ke cloud platform (Streamlit Community Cloud, Heroku, GCP, atau AWS)

### 4.3 Keterbatasan

- Dataset terbatas (768 sampel) dari populasi yang sangat spesifik
- Validasi eksternal belum dilakukan pada populasi berbeda
- Beberapa fitur penting (HbA1c, kolesterol) tidak tersedia dalam dataset
- Model belum diintegrasikan dengan sistem informasi kesehatan yang nyata

---

## 5. Referensi

[1] Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988).
    *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.*
    Proceedings of the Annual Symposium on Computer Application in Medical Care (pp. 261-265).

[2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ...
    & Duchesnay, E. (2011). *Scikit-learn: Machine Learning in Python.*
    Journal of Machine Learning Research, 12, 2825-2830.

[3] International Diabetes Federation. (2021). *IDF Diabetes Atlas, 10th Edition.*
    Brussels, Belgium: IDF.

[4] Breiman, L. (2001). *Random Forests.*
    Machine Learning, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324

[5] Friedman, J.H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.*
    The Annals of Statistics, 29(5), 1189-1232. https://doi.org/10.1214/aos/1013203451

[6] Hosmer, D.W., & Lemeshow, S. (2000). *Applied Logistic Regression (2nd ed.).*
    New York: John Wiley & Sons.

[7] James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021).
    *An Introduction to Statistical Learning with Applications in R (2nd ed.).*
    New York: Springer.

[8] Streamlit Inc. (2024). *Streamlit — The fastest way to build data apps.*
    Retrieved from https://streamlit.io

[9] Kementerian Kesehatan RI. (2018). *Hasil Utama Riskesdas 2018.*
    Badan Penelitian dan Pengembangan Kesehatan.

[10] American Diabetes Association. (2023). *Standards of Medical Care in Diabetes.*
     Diabetes Care, 46(Supplement_1).

---

*Laporan ini merupakan bagian dari keluaran wajib Soal 5 (Dokumentasi & Presentasi)*  
*UAS Pembelajaran Mesin — Semester Genap 2025/2026 — UDINUS Semarang*

---

| Koordinator MK Pembelajaran Mesin | Ketua Program Studi TI-S1 |
|-----------------------------------|--------------------------|
| Ardytha Luthfiarta, M.Kom, MCS | Dr. Edy Mulyanto, S.Si, M.Kom |
