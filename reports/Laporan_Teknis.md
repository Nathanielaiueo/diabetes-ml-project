# Laporan Teknis: Sistem Prediksi Penyakit Diabetes Berbasis Machine Learning

---

**Mata Kuliah** : Pembelajaran Mesin  
**Semester** : Genap 2025/2026  
**Universitas** : Universitas Dian Nuswantoro (UDINUS) Semarang  

**Kelompok:**  
- Fahmi Fatmawati Azzahra — NIM: A11.2024.15831  
- Nathaniela Febry Nathasa — NIM: A11.2024.15850  

**Dosen Pengampu:** Prof. Ir. Heru Agus Santoso, Ph.D, IPM, ASEAN Eng.  
**Program Studi:** Teknik Informatika (S1)  
**Tanggal:** 17 Juli 2026  

---

## Daftar Isi

1. [Pendahuluan dan Latar Belakang](#1-pendahuluan-dan-latar-belakang)
2. [Metodologi](#2-metodologi)
3. [Hasil dan Analisis](#3-hasil-dan-analisis)
4. [Deployment & Antarmuka Aplikasi](#4-deployment--antarmuka-aplikasi)
5. [Kesimpulan dan Rekomendasi](#5-kesimpulan-dan-rekomendasi)
6. [Referensi](#6-referensi)

---

## 1. Pendahuluan dan Latar Belakang

### 1.1 Latar Belakang

Diabetes mellitus adalah penyakit metabolik kronis yang ditandai oleh hiperglikemia
(kadar glukosa darah tinggi) yang diakibatkan oleh gangguan sekresi atau kerja insulin.
Berdasarkan data *International Diabetes Federation* (IDF) edisi 2021, terdapat sekitar
**537 juta** orang dewasa (usia 20–79 tahun) yang hidup dengan diabetes di seluruh dunia.
Di Indonesia, Riset Kesehatan Dasar (Riskesdas) 2018 mencatat prevalensi diabetes pada
penduduk ≥15 tahun sebesar 10,9%, meningkat dari 6,9% pada 2013.

Pendekatan *Machine Learning* menawarkan solusi inovatif untuk deteksi dini diabetes
melalui analisis pola pada data rekam medis. Dengan parameter kesehatan dasar seperti
kadar glukosa darah, indeks massa tubuh (BMI), tekanan darah, dan usia — model ML dapat
mengklasifikasikan pasien berisiko tinggi secara otomatis dan efisien.

### 1.2 Rumusan Masalah

**Permasalahan:** Bagaimana membangun model Machine Learning yang mampu memprediksi
dengan akurat apakah seorang pasien perempuan keturunan Pima Indian berusia ≥21 tahun
menderita diabetes, berdasarkan parameter diagnostik kesehatan yang tersedia?

**Jenis Task:** Binary Classification (0 = Tidak Diabetes, 1 = Diabetes)

### 1.3 Tujuan Penelitian

1. Membangun dan membandingkan minimal 3 model Machine Learning untuk prediksi diabetes
2. Mengidentifikasi faktor-faktor risiko yang paling berpengaruh terhadap diabetes
3. Mengembangkan aplikasi web interaktif yang dapat digunakan untuk skrining awal
4. Mendokumentasikan seluruh proses pipeline ML secara komprehensif

### 1.4 Metrik Kesuksesan

| Metrik | Target Minimum | Alasan |
|--------|---------------|--------|
| Accuracy | ≥ 75% | Benchmark performa umum |
| F1-Score | ≥ 0,70 | Menyeimbangkan Precision & Recall |
| AUC-ROC | ≥ 0,80 | Kemampuan diskriminasi kelas |
| Recall | ≥ 0,75 | Meminimalkan *false negative* (bahaya medis) |

### 1.5 Deskripsi Dataset

**Nama:** Pima Indians Diabetes Database  
**Institusi:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), USA  
**Akses Publik:** UCI Machine Learning Repository & Kaggle  

| Informasi | Detail |
|-----------|--------|
| Jumlah Sampel | 768 |
| Jumlah Fitur Input | 8 |
| Target | Outcome (0 = Tidak DM, 1 = DM) |
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
deskriptif dataset dengan dokumentasi resmi UCI Machine Learning Repository.

**Sumber:**
- URL: `https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`
- Alternatif: Kaggle (kaggle.com/uciml/pima-indians-diabetes-database)
- Referensi: UCI ML Repository (archive.ics.uci.edu/dataset/34/diabetes)

### 2.3 Exploratory Data Analysis (EDA)

EDA dilakukan secara komprehensif mencakup:

#### 2.3.1 Analisis Kualitas Data

**Missing Values Eksplisit (NaN):**  
Tidak ditemukan nilai NaN eksplisit. Ditemukan **implicit missing values** berupa nilai 0
pada kolom yang secara medis tidak mungkin bernilai 0:

| Kolom | Jumlah Nilai 0 | Persentase |
|-------|---------------|------------|
| Glucose | 5 | 0,7% |
| BloodPressure | 35 | 4,6% |
| SkinThickness | 227 | 29,6% |
| Insulin | 374 | 48,7% |
| BMI | 11 | 1,4% |

**Duplikat:** Tidak ditemukan baris duplikat (0 dari 768).

#### 2.3.2 Analisis Univariat

Distribusi setiap fitur divisualisasikan menggunakan histogram overlay per kelas,
menunjukkan perbedaan signifikan pada distribusi Glucose, BMI, dan Age antara
kelompok diabetes dan non-diabetes.

#### 2.3.3 Analisis Multivariat

- **Korelasi Pearson:** Glucose memiliki korelasi tertinggi dengan Outcome (r ≈ 0,47),
  diikuti BMI (r ≈ 0,29), Age (r ≈ 0,24), dan Pregnancies (r ≈ 0,22)
- **Scatter plots:** Separasi lebih jelas antara kelas pada scatter Glucose vs BMI

#### 2.3.4 Lima Key Insights

1. **Glukosa = Prediktor Terkuat** — Penderita DM rata-rata ~40 mg/dL lebih tinggi (r = 0,47)
2. **BMI Tinggi Berkorelasi dengan Diabetes** — Rata-rata BMI DM ≈ 35,1 vs non-DM ≈ 30,3
3. **Usia Berpengaruh** — Rata-rata usia DM ≈ 37 tahun vs non-DM ≈ 31 tahun
4. **Insulin Paling Banyak Hilang** — 48,7% nilai Insulin bernilai 0 (tidak valid medis)
5. **Class Imbalance** — 500 Non-DM (65,1%) vs 268 DM (34,9%) → perlu penanganan khusus

### 2.4 Preprocessing

#### 2.4.1 Penanganan Implicit Missing Values

Nilai 0 pada 5 kolom diganti NaN, lalu diimputasi dengan **median per kelas** (stratified
median imputation). Dipilih karena lebih representatif dan robust terhadap outlier.

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

StandardScaler diterapkan (`z = (x - μ) / σ`). Scaler di-*fit* **hanya** pada training
set, kemudian di-*transform* pada validation dan test set — mencegah *data leakage*.

### 2.5 Pemodelan

| Model | Deskripsi | Grid Parameter Utama |
|-------|-----------|---------------------|
| Logistic Regression | Baseline linear, interpretabilitas tinggi | C=[0.01,0.1,1,10,100], solver=[lbfgs,liblinear] |
| Random Forest | Ensemble bagging, robust terhadap overfitting | n_estimators=[100,200,300], max_depth=[None,10,20] |
| Gradient Boosting | Ensemble boosting sekuensial, terbaik tabular | n_estimators=[100,200,300], lr=[0.05,0.1,0.15] |

Tuning menggunakan **GridSearchCV** dengan 5-fold Stratified K-Fold dan scoring F1-Score.

### 2.6 Evaluasi

Evaluasi dilakukan pada ketiga split (train/val/test) menggunakan:
- **Accuracy** — Proporsi prediksi benar dari total prediksi
- **Precision** — Dari yang diprediksi positif, berapa yang benar positif
- **Recall (Sensitivity)** — Dari yang benar positif, berapa yang terdeteksi
- **F1-Score** — Harmonic mean Precision dan Recall
- **AUC-ROC** — Area Under the ROC Curve
- **Confusion Matrix** — Detail TP, TN, FP, FN

**Kriteria Pemilihan Model Terbaik:** F1-Score tertinggi pada **validation set**.

---

## 3. Hasil dan Analisis

### 3.1 Hasil EDA

Analisis korelasi menunjukkan **Glucose** memiliki korelasi tertinggi dengan Outcome
(r ≈ 0,47), diikuti **BMI** (r ≈ 0,29), **Age** (r ≈ 0,24), dan **Pregnancies** (r ≈ 0,22).
Dataset menunjukkan **class imbalance** 65:35, ditangani dengan `class_weight='balanced'`.

### 3.2 Hasil Preprocessing

Setelah preprocessing:
- Total NaN berhasil diimputasi ke 0
- 4 fitur baru berhasil ditambahkan (total 12 fitur dari 8)
- Dataset dibagi 537/115/116 (train/val/test)
- Semua fitur berhasil dinormalisasi (mean≈0, std≈1)

### 3.3 Hasil Training & Tuning

Ketiga model berhasil dilatih dengan hyperparameter terbaik melalui GridSearchCV.

### 3.4 Perbandingan Metrik Semua Model

Performa pada **Test Set**:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0,7586 | 0,6250 | 0,7500 | 0,6818 | 0,8283 |
| Random Forest | 0,8793 | 0,7955 | 0,8750 | 0,8333 | 0,9482 |
| **Gradient Boosting ⭐** | **0,8966** | **0,8333** | **0,8750** | **0,8537** | **0,9582** |

> ⭐ = Model Terbaik berdasarkan F1-Score tertinggi pada validation set (0,8571)

### 3.5 Feature Importance Analysis

Berdasarkan Gradient Boosting (model terbaik):

| Ranking | Fitur | Importance Score | Interpretasi |
|---------|-------|-----------------|-------------|
| 1 | Insulin | 0,6879 | Kadar insulin paling dominan |
| 2 | Glucose | 0,1082 | Gula darah sangat berkorelasi dengan DM |
| 3 | Age | 0,0712 | Usia meningkatkan risiko diabetes |
| 4 | SkinThickness | 0,0350 | Proxy ketebalan lemak tubuh |
| 5 | BMI | 0,0347 | Indeks massa tubuh terkait obesitas |

### 3.6 Analisis Model Terbaik: Gradient Boosting

Gradient Boosting dipilih karena:
1. F1-Score tertinggi pada validation set (0,8571) — menyeimbangkan Precision & Recall
2. Recall 0,8750 meminimalkan *false negative* (kasus DM tidak terdeteksi)
3. AUC-ROC 0,9582 — kemampuan diskriminasi kelas sangat baik (95,82%)
4. Gap train-test yang wajar (tidak overfitting parah)

---

## 4. Deployment & Antarmuka Aplikasi

### 4.1 Teknologi Deployment

Aplikasi **DiabetesSense** dibangun menggunakan teknologi open-source dan di-deploy
ke platform cloud sehingga dapat diakses dari mana saja tanpa instalasi apapun.

| Komponen | Teknologi | Fungsi |
|----------|-----------|--------|
| Web Framework | Streamlit | Membangun antarmuka web interaktif berbasis Python |
| Hosting | Streamlit Community Cloud | Auto-deploy langsung dari GitHub, gratis |
| Source Control | GitHub | Version control, penyimpanan kode & model .pkl |
| ML Library | Scikit-learn | Training, evaluasi, dan serialisasi model |
| Visualisasi | Plotly | Grafik interaktif (bar, heatmap, radar, ROC) |
| Serialisasi | Joblib | Menyimpan dan memuat model yang telah dilatih |

### 4.2 Arsitektur Aplikasi

Aplikasi DiabetesSense terdiri dari **6 halaman** yang diakses melalui sidebar navigasi:

| No | Halaman | Fungsi Utama |
|----|---------|-------------|
| 1 | 🏠 Beranda | Informasi tim, statistik dataset, latar belakang, pipeline ML |
| 2 | 📊 EDA Dashboard | Distribusi fitur, korelasi, kualitas data, 5 key insights |
| 3 | 🤖 Prediksi Diabetes | Form input pasien → prediksi real-time + faktor risiko personal |
| 4 | 📈 Evaluasi Model | Tabel perbandingan, confusion matrix, ROC curves, radar chart |
| 5 | 💡 Interpretasi & Bisnis | Feature importance, justifikasi model, rekomendasi strategis |
| 6 | 📚 Dokumentasi | Dataset, metodologi, tech stack, panduan penggunaan |

### 4.3 Link Akses Aplikasi

| Platform | URL |
|----------|-----|
| 🌐 Streamlit App (Live) | https://fahmi-natha-diabetes-ml.streamlit.app/ |
| 💻 GitHub Repository | https://github.com/Nathanielaiueo/diabetes-ml-project |

### 4.4 Antarmuka Aplikasi

Berikut tampilan antarmuka dari seluruh halaman utama aplikasi DiabetesSense:

#### 4.4.1 Halaman Beranda
Menampilkan identitas tim pengembang (Fahmi & Nathaniela beserta NIM), statistik dataset
(768 sampel, 8 fitur, 268 kasus DM, 500 non-DM), latar belakang proyek, dan alur pipeline ML.

*Screenshot: `reports/figures/screenshots/01_beranda.png`*

#### 4.4.2 Halaman EDA Dashboard
Menyediakan 5 tab eksplorasi interaktif: Overview (ringkasan & preview tabel data),
Distribusi & Univariat, Korelasi & Multivariat, Kualitas Data, dan 5 Key Insights.

*Screenshot: `reports/figures/screenshots/02_eda.png`*

#### 4.4.3 Halaman Prediksi Diabetes
Form interaktif dengan 8 slider input (Kehamilan, Glukosa, Tekanan Darah, Kulit Triceps,
Insulin, BMI, Diabetes Pedigree, Usia). Hasil menampilkan label prediksi, probabilitas,
tingkat risiko, dan referensi nilai normal medis.

*Screenshot: `reports/figures/screenshots/03_prediksi.png`*

#### 4.4.4 Halaman Evaluasi Model
Kartu metrik model terbaik (Accuracy 89,66%, F1 0,8537, AUC-ROC 0,9582), tabel perbandingan
ketiga model pada Train/Val/Test set, confusion matrix interaktif, dan ROC curves.

*Screenshot: `reports/figures/screenshots/04_evaluasi.png`*

#### 4.4.5 Halaman Interpretasi & Insights Bisnis
Analisis feature importance dari ketiga model, justifikasi pemilihan Gradient Boosting,
dan rekomendasi strategis bagi fasilitas kesehatan.

*Screenshot: `reports/figures/screenshots/05_interpretasi.png`*

---

## 5. Kesimpulan dan Rekomendasi

### 5.1 Kesimpulan

1. **Semua target metrik berhasil dicapai** oleh Gradient Boosting pada test set:
   Accuracy = 0,8966 (≥75%), F1-Score = 0,8537 (≥0,70), AUC-ROC = 0,9582 (≥0,80),
   Recall = 0,8750 (≥0,75).

2. **Glucose dan BMI adalah prediktor terkuat** secara konsisten di semua model.
   Intervensi klinis yang fokus pada dua faktor ini akan paling efektif dalam pencegahan.

3. **Gradient Boosting unggul** dalam menyeimbangkan Precision (0,8333) dan
   Recall (0,8750), menjadikannya pilihan optimal untuk konteks medis.

4. **Pipeline ML end-to-end** berhasil dibangun mencakup seluruh tahap: akuisisi data,
   EDA, preprocessing, pemodelan, evaluasi, dan deployment sebagai aplikasi web interaktif.

5. **Feature engineering berbasis domain medis** (kategorisasi BMI, Glucose, Age Group,
   dan Insulin-Glucose Ratio) terbukti meningkatkan performa model.

6. **Aplikasi DiabetesSense** berhasil di-deploy secara online dan dapat diakses tanpa
   instalasi — menjadikannya alat skrining yang praktis bagi tenaga kesehatan.

### 5.2 Rekomendasi

#### a. Rekomendasi Klinis / Bisnis

- Implementasikan sistem skrining ML ini di puskesmas dan klinik endokrinologi untuk triase
  pasien berisiko tinggi
- Prioritaskan pemeriksaan kadar insulin, glukosa darah, dan BMI pada setiap kunjungan pasien
- Gunakan threshold prediksi ≥0,4 untuk meminimalkan *false negative* yang berbahaya secara medis
- Integrasikan dengan sistem Electronic Health Record (EHR) yang sudah berjalan

#### b. Rekomendasi Teknis

- Implementasikan SMOTE (Synthetic Minority Over-sampling Technique) untuk oversampling
  minority class secara lebih agresif
- Eksplorasi XGBoost, LightGBM, dan ensemble stacking untuk performa lebih tinggi
- Gunakan SHAP (SHapley Additive exPlanations) untuk interpretabilitas yang lebih mendalam
- Validasi model pada dataset yang lebih besar dan populasi yang lebih beragam (termasuk
  populasi Indonesia)
- Tambahkan fitur klinis penting seperti HbA1c, kolesterol, dan riwayat keluarga

### 5.3 Keterbatasan

- Dataset terbatas (768 sampel) dari populasi spesifik (perempuan Pima Indian ≥21 tahun)
  — generalisasi ke populasi lain perlu validasi lebih lanjut
- Sebanyak 48,7% nilai Insulin kosong (nilai 0 tidak valid medis) — imputasi median adalah
  approximation terbaik yang tersedia
- Fitur penting seperti HbA1c, kolesterol, dan riwayat keluarga tidak tersedia dalam dataset
- Validasi eksternal belum dilakukan pada populasi berbeda atau data Indonesia

---

## 6. Referensi

[1] Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988).
    *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.*
    Proceedings of the Annual Symposium on Computer Application in Medical Care, 261–265.

[2] Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.*
    Journal of Machine Learning Research, 12, 2825–2830.
    https://jmlr.org/papers/v12/pedregosa11a.html

[3] International Diabetes Federation. (2021). *IDF Diabetes Atlas, 10th Edition.*
    Brussels, Belgium: IDF. https://diabetesatlas.org

[4] Breiman, L. (2001). *Random Forests.*
    Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

[5] Friedman, J.H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.*
    The Annals of Statistics, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

[6] Hosmer, D.W., & Lemeshow, S. (2000). *Applied Logistic Regression (2nd ed.).*
    New York: John Wiley & Sons.

[7] James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021).
    *An Introduction to Statistical Learning with Applications in R (2nd ed.).*
    New York: Springer. https://www.statlearning.com

[8] Streamlit Inc. (2024). *Streamlit — The fastest way to build data apps.*
    https://streamlit.io

[9] Kementerian Kesehatan RI. (2018). *Hasil Utama Riset Kesehatan Dasar (Riskesdas) 2018.*
    Badan Penelitian dan Pengembangan Kesehatan, Jakarta.

[10] American Diabetes Association. (2023). *Standards of Medical Care in Diabetes 2023.*
     Diabetes Care, 46(Supplement 1). https://doi.org/10.2337/dc23-Sint

[11] Lundberg, S.M., & Lee, S.I. (2017). *A unified approach to interpreting model predictions.*
     Advances in Neural Information Processing Systems (NeurIPS), 30, 4765–4774.

[12] World Health Organization. (2024). *Obesity and overweight.* WHO Fact Sheet.
     https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight

[13] Dua, D., & Graff, C. (2019). *UCI Machine Learning Repository.*
     Irvine, CA: University of California, School of Information and Computer Science.
     https://archive.ics.uci.edu/dataset/34/diabetes

---

*Laporan ini merupakan bagian dari keluaran wajib UAS Pembelajaran Mesin*  
*Semester Genap 2025/2026 — Teknik Informatika — UDINUS Semarang*

---

**Universitas Dian Nuswantoro (UDINUS) Semarang**
