"""
Generate Laporan Teknis UAS PDF — DiabetesSense
Menggunakan reportlab untuk membuat PDF dengan screenshot aplikasi.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image, PageBreak, HRFlowable,
                                 KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import Flowable
from pathlib import Path
import os

# ── Paths ──────────────────────────────────────────────────────
BASE   = Path(__file__).parent
SS_DIR = BASE / 'reports' / 'figures' / 'screenshots'
OUT    = BASE / 'reports' / 'Laporan_Teknis_UAS.pdf'

# ── Colour palette ─────────────────────────────────────────────
C_NAVY   = HexColor('#1a2744')
C_BLUE   = HexColor('#2E86AB')
C_PURPLE = HexColor('#A23B72')
C_ORANGE = HexColor('#F18F01')
C_RED    = HexColor('#C73E1D')
C_GREEN  = HexColor('#27AE60')
C_GRAY   = HexColor('#6B7280')
C_LGRAY  = HexColor('#F3F4F6')
C_BORDER = HexColor('#E5E7EB')
C_WHITE  = white

# ── Styles ─────────────────────────────────────────────────────
base_styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

TITLE_STYLE = S('title_main', fontName='Helvetica-Bold', fontSize=22,
                textColor=C_NAVY, alignment=TA_CENTER, spaceAfter=6)
SUB_STYLE   = S('sub_main',   fontName='Helvetica',      fontSize=11,
                textColor=C_GRAY, alignment=TA_CENTER, spaceAfter=4)
H1 = S('h1', fontName='Helvetica-Bold', fontSize=15, textColor=C_NAVY,
        spaceBefore=18, spaceAfter=8, borderPad=4)
H2 = S('h2', fontName='Helvetica-Bold', fontSize=12, textColor=C_BLUE,
        spaceBefore=12, spaceAfter=6)
H3 = S('h3', fontName='Helvetica-Bold', fontSize=10.5, textColor=C_NAVY,
        spaceBefore=8, spaceAfter=4)
BODY = S('body', fontName='Helvetica', fontSize=9.5, leading=15,
         textColor=HexColor('#1F2937'), alignment=TA_JUSTIFY, spaceAfter=6)
BODY_C = S('body_c', fontName='Helvetica', fontSize=9.5, leading=15,
           textColor=HexColor('#1F2937'), alignment=TA_CENTER, spaceAfter=4)
CAPTION = S('cap', fontName='Helvetica-Oblique', fontSize=8.5,
            textColor=C_GRAY, alignment=TA_CENTER, spaceAfter=12)
BULLET = S('bullet', fontName='Helvetica', fontSize=9.5, leading=14,
           leftIndent=14, firstLineIndent=-10, textColor=HexColor('#1F2937'),
           spaceAfter=3)
SMALL = S('small', fontName='Helvetica', fontSize=8.5, textColor=C_GRAY,
          alignment=TA_CENTER)
LINK  = S('link',  fontName='Helvetica', fontSize=9.5, textColor=C_BLUE,
          spaceAfter=4)
BOLD  = S('bold', fontName='Helvetica-Bold', fontSize=9.5,
          textColor=HexColor('#1F2937'), spaceAfter=4)
LABEL_BLUE = S('label_b', fontName='Helvetica-Bold', fontSize=9.5,
               textColor=C_BLUE, spaceAfter=3)

def hr(color=C_BORDER, thickness=0.8):
    return HRFlowable(width='100%', thickness=thickness, color=color,
                      spaceAfter=8, spaceBefore=4)

def sp(h=8):
    return Spacer(1, h)

def img(filename, width=15*cm, caption=None):
    path = SS_DIR / filename
    if not path.exists():
        return [Paragraph(f'[Gambar tidak ditemukan: {filename}]', CAPTION)]
    items = [Image(str(path), width=width,
                   height=width * 0.625)]  # 16:10 ratio
    if caption:
        items.append(Paragraph(caption, CAPTION))
    return items

def table_style_base():
    return [
        ('BACKGROUND',  (0,0), (-1,0), C_NAVY),
        ('TEXTCOLOR',   (0,0), (-1,0), C_WHITE),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,-1), 8.5),
        ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_WHITE, C_LGRAY]),
        ('GRID',        (0,0), (-1,-1), 0.4, C_BORDER),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0),(-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING',(0,0), (-1,-1), 6),
    ]

# ── Cover page helper ──────────────────────────────────────────
class ColorRect(Flowable):
    def __init__(self, w, h, color):
        Flowable.__init__(self)
        self.w, self.h, self.color = w, h, color
    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.w, self.h, fill=1, stroke=0)

# ── Build story ────────────────────────────────────────────────
def build():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        title='Laporan Teknis UAS – DiabetesSense',
        author='Fahmi Fatmawati Azzahra & Nathaniela Febry Nathasa',
    )
    W = doc.width
    story = []

    # ════════════════════════════════════════════════════════════
    # COVER
    # ════════════════════════════════════════════════════════════
    story.append(sp(40))
    story.append(Paragraph('LAPORAN TEKNIS', S('ct', fontName='Helvetica-Bold',
                 fontSize=13, textColor=C_GRAY, alignment=TA_CENTER)))
    story.append(sp(6))
    story.append(Paragraph('Sistem Prediksi Penyakit Diabetes',
                 S('ct2', fontName='Helvetica-Bold', fontSize=24,
                   textColor=C_NAVY, alignment=TA_CENTER)))
    story.append(Paragraph('Berbasis Machine Learning',
                 S('ct3', fontName='Helvetica-Bold', fontSize=20,
                   textColor=C_BLUE, alignment=TA_CENTER, spaceAfter=4)))
    story.append(sp(4))
    story.append(Paragraph('DiabetesSense — ML Classifier',
                 S('ct4', fontName='Helvetica-Oblique', fontSize=12,
                   textColor=C_GRAY, alignment=TA_CENTER)))
    story.append(sp(30))
    story.append(hr(C_NAVY, 1.5))
    story.append(sp(16))

    # info table
    info = [
        ['Mata Kuliah', ':', 'Pembelajaran Mesin'],
        ['Semester',    ':', 'Genap 2025/2026'],
        ['Universitas', ':', 'Universitas Dian Nuswantoro (UDINUS) Semarang'],
        ['', '', ''],
        ['Anggota 1',   ':', 'Fahmi Fatmawati Azzahra — NIM: A11.2024.15831'],
        ['Anggota 2',   ':', 'Nathaniela Febry Nathasa — NIM: A11.2024.15850'],
        ['', '', ''],
        ['Dosen Pengampu', ':', 'Prof. Ir. Heru Agus Santoso, Ph.D, IPM, ASEAN Eng.'],
    ]
    ts = TableStyle([
        ('FONTNAME',  (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',  (2,0), (2,-1), 'Helvetica'),
        ('FONTSIZE',  (0,0), (-1,-1), 10),
        ('TEXTCOLOR', (0,0), (-1,-1), HexColor('#1F2937')),
        ('TOPPADDING',(0,0),(-1,-1), 4),
        ('BOTTOMPADDING',(0,0),(-1,-1), 4),
        ('LEFTPADDING',(0,0),(-1,-1), 0),
        ('ALIGN',     (0,0),(-1,-1),'LEFT'),
    ])
    t = Table(info, colWidths=[3.8*cm, 0.4*cm, W-4.2*cm], style=ts)
    story.append(t)
    story.append(sp(20))
    story.append(hr(C_NAVY, 1.5))
    story.append(sp(40))
    story.append(Paragraph('Teknik Informatika · UDINUS Semarang · 2025/2026',
                 S('ft', fontName='Helvetica-Oblique', fontSize=10,
                   textColor=C_GRAY, alignment=TA_CENTER)))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # DAFTAR ISI
    # ════════════════════════════════════════════════════════════
    story.append(Paragraph('DAFTAR ISI', H1))
    story.append(hr())
    toc = [
        ('1.', 'Pendahuluan & Latar Belakang'),
        ('2.', 'Metodologi'),
        ('3.', 'Hasil dan Analisis'),
        ('4.', 'Deployment & Antarmuka Aplikasi'),
        ('5.', 'Kesimpulan dan Rekomendasi'),
        ('6.', 'Referensi'),
    ]
    for num, title in toc:
        story.append(Paragraph(f'{num}&nbsp;&nbsp;&nbsp;{title}',
                     S('toc', fontName='Helvetica', fontSize=10.5,
                       textColor=C_NAVY, spaceAfter=8, leading=16)))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # BAB 1 — PENDAHULUAN
    # ════════════════════════════════════════════════════════════
    story.append(Paragraph('1. Pendahuluan & Latar Belakang', H1))
    story.append(hr())

    story.append(Paragraph('1.1 Latar Belakang', H2))
    story.append(Paragraph(
        'Diabetes mellitus adalah penyakit metabolik kronis yang ditandai oleh '
        'hiperglikemia (kadar glukosa darah tinggi) akibat gangguan sekresi atau kerja insulin. '
        'Berdasarkan data <i>International Diabetes Federation</i> (IDF) edisi 2021, '
        'terdapat sekitar <b>537 juta</b> orang dewasa yang hidup dengan diabetes di seluruh dunia. '
        'Di Indonesia, Riskesdas 2018 mencatat prevalensi diabetes pada penduduk ≥15 tahun '
        'sebesar 10,9%, meningkat dari 6,9% pada 2013.', BODY))
    story.append(Paragraph(
        'Pendekatan <i>Machine Learning</i> menawarkan solusi inovatif untuk deteksi dini '
        'diabetes melalui analisis pola pada data rekam medis. Dengan parameter kesehatan '
        'dasar seperti kadar glukosa, BMI, tekanan darah, dan usia, model ML dapat '
        'mengklasifikasikan pasien berisiko tinggi secara otomatis dan efisien.', BODY))

    story.append(Paragraph('1.2 Problem Statement', H2))
    story.append(Paragraph(
        '<b>Permasalahan:</b> Bagaimana membangun model Machine Learning yang mampu '
        'memprediksi apakah seorang pasien perempuan keturunan Pima Indian berusia '
        '≥21 tahun menderita diabetes, berdasarkan parameter diagnostik kesehatan yang tersedia?', BODY))
    story.append(Paragraph('<b>Jenis Task:</b> Binary Classification (0 = Tidak Diabetes, 1 = Diabetes)', BODY))

    story.append(Paragraph('1.3 Tujuan', H2))
    goals = [
        'Membangun dan membandingkan minimal 3 model Machine Learning untuk prediksi diabetes',
        'Mengidentifikasi faktor-faktor risiko yang paling berpengaruh terhadap diabetes',
        'Mengembangkan aplikasi web interaktif untuk skrining awal diabetes',
        'Mendokumentasikan seluruh proses pipeline ML secara komprehensif',
    ]
    for g in goals:
        story.append(Paragraph(f'• {g}', BULLET))
    story.append(sp(4))

    story.append(Paragraph('1.4 Metrik Kesuksesan', H2))
    met_data = [
        ['Metrik', 'Target Minimum', 'Alasan'],
        ['Accuracy',  '≥ 75%', 'Benchmark performa umum'],
        ['F1-Score',  '≥ 0,70', 'Menyeimbangkan Precision & Recall'],
        ['AUC-ROC',   '≥ 0,80', 'Kemampuan diskriminasi kelas'],
        ['Recall',    '≥ 0,75', 'Meminimalkan false negative (bahaya medis)'],
    ]
    t = Table(met_data, colWidths=[3*cm, 3.5*cm, W-6.5*cm], style=table_style_base())
    story.append(t)
    story.append(sp(6))

    story.append(Paragraph('1.5 Dataset', H2))
    story.append(Paragraph(
        '<b>Nama:</b> Pima Indians Diabetes Database<br/>'
        '<b>Institusi:</b> National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), USA<br/>'
        '<b>Akses Publik:</b> UCI Machine Learning Repository & Kaggle', BODY))

    ds_data = [
        ['Informasi', 'Detail'],
        ['Jumlah Sampel', '768'],
        ['Jumlah Fitur Input', '8'],
        ['Target', 'Outcome (0 = Tidak DM, 1 = DM)'],
        ['Populasi', 'Perempuan keturunan Pima Indian, ≥21 tahun'],
        ['Format', 'CSV (comma-separated values)'],
    ]
    t = Table(ds_data, colWidths=[5*cm, W-5*cm], style=table_style_base())
    story.append(t)
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # BAB 2 — METODOLOGI
    # ════════════════════════════════════════════════════════════
    story.append(Paragraph('2. Metodologi', H1))
    story.append(hr())

    story.append(Paragraph('2.1 Alur Kerja (Pipeline)', H2))
    pipeline = [['Akuisisi Data', '→', 'EDA', '→', 'Preprocessing', '→', 'Pemodelan', '→', 'Evaluasi', '→', 'Deployment']]
    pt = Table(pipeline, colWidths=[2.8*cm,0.5*cm,1.5*cm,0.5*cm,2.8*cm,0.5*cm,2.5*cm,0.5*cm,2.2*cm,0.5*cm,2.5*cm])
    pt.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('TEXTCOLOR', (0,0), (-1,-1), C_NAVY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BACKGROUND', (0,0),(0,0), HexColor('#dbeafe')),
        ('BACKGROUND', (2,0),(2,0), HexColor('#fce7f3')),
        ('BACKGROUND', (4,0),(4,0), HexColor('#fef3c7')),
        ('BACKGROUND', (6,0),(6,0), HexColor('#d1fae5')),
        ('BACKGROUND', (8,0),(8,0), HexColor('#ede9fe')),
        ('BACKGROUND', (10,0),(10,0), HexColor('#fee2e2')),
        ('TOPPADDING', (0,0),(-1,-1), 6),
        ('BOTTOMPADDING', (0,0),(-1,-1), 6),
        ('BOX', (0,0),(0,0), 1, C_BLUE),
        ('ROUNDEDCORNERS', [4], ),
    ]))
    story.append(pt)
    story.append(sp(8))

    story.append(Paragraph('2.2 Data Acquisition', H2))
    story.append(Paragraph(
        'Dataset diunduh dari repository publik GitHub (Jason Brownlee\'s Datasets). '
        'Verifikasi keaslian dilakukan dengan membandingkan statistik deskriptif dengan '
        'dokumentasi resmi UCI Machine Learning Repository.', BODY))

    story.append(Paragraph('2.3 Exploratory Data Analysis (EDA)', H2))
    story.append(Paragraph(
        'EDA mencakup: (a) analisis kualitas data — missing values eksplisit, implicit '
        'missing (nilai 0 tidak valid medis), dan duplikat; (b) analisis univariat — distribusi '
        'histogram per kelas; (c) analisis multivariat — korelasi Pearson dan scatter plots.', BODY))

    eda_kualitas = [
        ['Kolom', 'Jumlah Nilai 0', 'Persentase'],
        ['Glucose',        '5',   '0,7%'],
        ['BloodPressure',  '35',  '4,6%'],
        ['SkinThickness',  '227', '29,6%'],
        ['Insulin',        '374', '48,7%'],
        ['BMI',            '11',  '1,4%'],
    ]
    t = Table(eda_kualitas, colWidths=[5*cm, 4*cm, W-9*cm], style=table_style_base())
    story.append(Paragraph('Tabel 2.1 — Implicit Missing Values (Nilai 0 Tidak Valid)', CAPTION))
    story.append(t)
    story.append(sp(6))

    story.append(Paragraph('2.4 Preprocessing', H2))
    pre_steps = [
        ('Penanganan Missing Values', 'Nilai 0 pada 5 kolom diganti NaN, lalu diimputasi dengan median per kelas (stratified median imputation).'),
        ('Feature Engineering',      '4 fitur baru: BMI_Category, Glucose_Category, Age_Group, Insulin_Glucose_Ratio.'),
        ('Pembagian Dataset',         'Stratified split: Train 70% (537), Validation 15% (115), Test 15% (116).'),
        ('Feature Scaling',           'StandardScaler di-fit hanya pada training set untuk mencegah data leakage.'),
    ]
    for step, desc in pre_steps:
        story.append(Paragraph(f'<b>• {step}:</b> {desc}', BULLET))
    story.append(sp(6))

    story.append(Paragraph('2.5 Pemodelan & Hyperparameter Tuning', H2))
    model_data = [
        ['Model', 'Deskripsi', 'Grid Parameter Utama'],
        ['Logistic Regression', 'Baseline linear, interpretabilitas koefisien tinggi',
         'C=[0.01,0.1,1,10,100], solver=[lbfgs, liblinear]'],
        ['Random Forest', 'Ensemble bagging, robust terhadap overfitting',
         'n_estimators=[100,200,300], max_depth=[None,10,20]'],
        ['Gradient Boosting', 'Ensemble boosting sekuensial, umumnya terbaik pada tabular',
         'n_estimators=[100,200,300], lr=[0.05,0.1,0.15]'],
    ]
    t = Table(model_data, colWidths=[3.8*cm, 5.2*cm, W-9*cm], style=table_style_base())
    story.append(t)
    story.append(Paragraph(
        'Tuning menggunakan GridSearchCV dengan 5-fold Stratified K-Fold dan scoring F1-Score.', BODY))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # BAB 3 — HASIL DAN ANALISIS
    # ════════════════════════════════════════════════════════════
    story.append(Paragraph('3. Hasil dan Analisis', H1))
    story.append(hr())

    story.append(Paragraph('3.1 Hasil EDA', H2))
    story.append(Paragraph(
        'Analisis korelasi Pearson menunjukkan <b>Glucose</b> memiliki korelasi tertinggi '
        'dengan target Outcome (r ≈ 0,47), diikuti <b>BMI</b> (r ≈ 0,29), <b>Age</b> '
        '(r ≈ 0,24), dan <b>Pregnancies</b> (r ≈ 0,22). Dataset menunjukkan <b>class '
        'imbalance</b> 65:35 yang ditangani dengan class_weight="balanced".', BODY))

    story.append(Paragraph('3.2 Hasil Training & Evaluasi', H2))
    story.append(Paragraph(
        'Berikut performa ketiga model pada <b>Test Set</b> (data yang belum pernah '
        'dilihat selama training):', BODY))

    res_data = [
        ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
        ['Logistic Regression', '0,7586', '0,6250', '0,7500', '0,6818', '0,8283'],
        ['Random Forest',       '0,8793', '0,7955', '0,8750', '0,8333', '0,9482'],
        ['Gradient Boosting ⭐', '0,8966', '0,8333', '0,8750', '0,8537', '0,9582'],
    ]
    ts2 = table_style_base() + [
        ('FONTNAME',    (0,3), (-1,3), 'Helvetica-Bold'),
        ('TEXTCOLOR',   (0,3), (-1,3), C_ORANGE),
        ('BACKGROUND',  (0,3), (-1,3), HexColor('#FFF8E1')),
    ]
    t = Table(res_data, colWidths=[4.5*cm,2.2*cm,2.2*cm,2.2*cm,2.2*cm,2.2*cm],
              style=TableStyle(ts2))
    story.append(t)
    story.append(Paragraph('Tabel 3.1 — Perbandingan Performa Model (Test Set). ⭐ = Model Terbaik', CAPTION))
    story.append(sp(6))

    story.append(Paragraph('3.3 Model Terbaik: Gradient Boosting', H2))
    story.append(Paragraph(
        'Gradient Boosting dipilih sebagai model terbaik berdasarkan F1-Score tertinggi '
        'pada validation set (0,8571). Model ini memberikan keseimbangan terbaik antara '
        'Precision (0,8333) dan Recall (0,8750), dengan AUC-ROC 0,9582 — artinya model '
        'mampu membedakan kelas diabetes dan non-diabetes dengan sangat baik (95,82%).', BODY))

    story.append(Paragraph('3.4 Feature Importance', H2))
    fi_data = [
        ['Ranking', 'Fitur', 'Nilai Importance (GB)', 'Interpretasi'],
        ['1', 'Insulin',    '0,6879', 'Kadar insulin paling dominan dalam prediksi'],
        ['2', 'Glucose',    '0,1082', 'Gula darah sangat berkorelasi dengan DM'],
        ['3', 'Age',        '0,0712', 'Usia meningkatkan risiko diabetes'],
        ['4', 'SkinThickness', '0,0350', 'Proxy ketebalan lemak tubuh'],
        ['5', 'BMI',        '0,0347', 'Indeks massa tubuh terkait obesitas'],
    ]
    t = Table(fi_data, colWidths=[1.8*cm, 4.2*cm, 4.5*cm, W-10.5*cm], style=table_style_base())
    story.append(t)
    story.append(Paragraph('Tabel 3.2 — Top 5 Feature Importance (Gradient Boosting)', CAPTION))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # BAB 4 — DEPLOYMENT & ANTARMUKA APLIKASI
    # ════════════════════════════════════════════════════════════
    story.append(Paragraph('4. Deployment & Antarmuka Aplikasi', H1))
    story.append(hr())

    story.append(Paragraph('4.1 Teknologi Deployment', H2))
    story.append(Paragraph(
        'Aplikasi <b>DiabetesSense</b> dibangun menggunakan teknologi open-source dan '
        'di-deploy ke platform cloud gratis sehingga dapat diakses dari mana saja tanpa '
        'instalasi apapun.', BODY))

    tech_data = [
        ['Komponen', 'Teknologi', 'Fungsi'],
        ['Web Framework',  'Streamlit 1.x',          'Membangun antarmuka web interaktif berbasis Python'],
        ['Hosting',        'Streamlit Community Cloud', 'Platform hosting gratis, auto-deploy dari GitHub'],
        ['Source Control', 'GitHub',                  'Version control, penyimpanan kode & model .pkl'],
        ['ML Library',     'Scikit-learn',             'Training, evaluasi, dan serialisasi model ML'],
        ['Visualisasi',    'Plotly',                   'Grafik interaktif (bar, pie, heatmap, radar, ROC)'],
        ['Serialisasi',    'Joblib',                   'Menyimpan dan memuat model yang telah dilatih'],
    ]
    t = Table(tech_data, colWidths=[3.5*cm, 4.5*cm, W-8*cm], style=table_style_base())
    story.append(t)
    story.append(Paragraph('Tabel 4.1 — Tech Stack Deployment Aplikasi', CAPTION))
    story.append(sp(4))

    story.append(Paragraph('4.2 Arsitektur Aplikasi', H2))
    story.append(Paragraph(
        'Aplikasi DiabetesSense terdiri dari <b>6 halaman</b> yang dapat diakses '
        'melalui sidebar navigasi di sisi kiri layar. Setiap halaman dirancang untuk '
        'tujuan yang spesifik dan saling melengkapi.', BODY))

    arch_data = [
        ['No.', 'Halaman', 'Fungsi Utama'],
        ['1', '🏠 Beranda',            'Informasi tim pengembang, statistik dataset, latar belakang proyek, dan alur pipeline ML'],
        ['2', '📊 EDA Dashboard',      'Eksplorasi data interaktif: distribusi fitur, korelasi antar variabel, kualitas data, dan 5 key insights'],
        ['3', '🤖 Prediksi Diabetes',  'Form input data pasien → prediksi real-time dengan probabilitas + analisis faktor risiko personal'],
        ['4', '📈 Evaluasi Model',     'Tabel perbandingan 3 model, confusion matrix interaktif, ROC curves, dan radar chart performa'],
        ['5', '💡 Interpretasi & Bisnis', 'Feature importance per model, justifikasi pemilihan model terbaik, rekomendasi strategis'],
        ['6', '📚 Dokumentasi',        'Deskripsi dataset, metodologi pipeline ML, tech stack, dan panduan penggunaan aplikasi'],
    ]
    t = Table(arch_data, colWidths=[1*cm, 4*cm, W-5*cm], style=table_style_base())
    story.append(t)
    story.append(Paragraph('Tabel 4.2 — Arsitektur Halaman Aplikasi DiabetesSense', CAPTION))
    story.append(sp(6))

    story.append(Paragraph('4.3 Link Akses Aplikasi', H2))
    link_data = [
        ['Platform', 'URL / Link'],
        ['🌐 Streamlit App (Live)', 'https://fahmi-natha-diabetes-ml.streamlit.app/'],
        ['💻 GitHub Repository',    'https://github.com/Nathanielaiueo/diabetes-ml-project'],
    ]
    lts = table_style_base() + [
        ('TEXTCOLOR', (1,1), (1,-1), C_BLUE),
        ('FONTNAME',  (1,1), (1,-1), 'Helvetica-Oblique'),
        ('ALIGN',     (1,1), (1,-1), 'LEFT'),
    ]
    t = Table(link_data, colWidths=[5*cm, W-5*cm], style=TableStyle(lts))
    story.append(t)
    story.append(Paragraph('Tabel 4.3 — Link Akses Aplikasi', CAPTION))
    story.append(PageBreak())

    story.append(Paragraph('4.4 Antarmuka Aplikasi', H2))
    story.append(Paragraph(
        'Berikut adalah tampilan antarmuka dari seluruh halaman utama aplikasi '
        'DiabetesSense yang telah berhasil di-deploy dan dapat diakses secara online.', BODY))
    story.append(sp(6))

    # Screenshot 1 — Beranda
    story.append(Paragraph('4.4.1 Halaman Beranda', H3))
    story.append(Paragraph(
        'Halaman pembuka menampilkan identitas tim pengembang beserta NIM, '
        'statistik ringkas dataset (768 sampel, 8 fitur, 268 kasus DM), '
        'latar belakang proyek, dan alur pipeline Machine Learning.', BODY))
    story += img('01_beranda.png', width=W,
                 caption='Gambar 4.1 — Halaman Beranda: Tim pengembang, statistik dataset, dan informasi proyek')
    story.append(PageBreak())

    # Screenshot 2 — EDA
    story.append(Paragraph('4.4.2 Halaman EDA Dashboard', H3))
    story.append(Paragraph(
        'EDA Dashboard menyediakan 5 tab eksplorasi data interaktif: Overview '
        '(ringkasan dataset & preview tabel), Distribusi & Univariat (histogram per kelas), '
        'Korelasi & Multivariat (heatmap & scatter), Kualitas Data (missing values & outlier), '
        'dan 5 Key Insights.', BODY))
    story += img('02_eda.png', width=W,
                 caption='Gambar 4.2 — EDA Dashboard: Tab Overview menampilkan ringkasan dataset dan preview data')
    story.append(PageBreak())

    # Screenshot 3 — Prediksi
    story.append(Paragraph('4.4.3 Halaman Prediksi Diabetes', H3))
    story.append(Paragraph(
        'Halaman prediksi menyediakan form interaktif dengan 8 slider input (Kehamilan, Glukosa, '
        'Tekanan Darah, Kulit Triceps, Insulin, BMI, Diabetes Pedigree, Usia). Pengguna dapat '
        'memilih model (LR, RF, atau GB) dan mendapatkan hasil prediksi real-time beserta '
        'probabilitas, tingkat risiko, dan referensi nilai normal medis.', BODY))
    story += img('03_prediksi.png', width=W,
                 caption='Gambar 4.3 — Halaman Prediksi: Form input pasien dan hasil prediksi "Tidak Terdeteksi Diabetes"')
    story.append(PageBreak())

    # Screenshot 4 — Evaluasi
    story.append(Paragraph('4.4.4 Halaman Evaluasi Model', H3))
    story.append(Paragraph(
        'Halaman Evaluasi Model menampilkan: (a) kartu metrik model terbaik (Gradient Boosting: '
        'Accuracy 89,66%, F1 0,8537, AUC-ROC 0,9582), (b) tabel perbandingan lengkap ketiga '
        'model pada Train/Val/Test set, (c) confusion matrix interaktif per model, '
        'dan (d) ROC curves serta radar chart perbandingan visual.', BODY))
    story += img('04_evaluasi.png', width=W,
                 caption='Gambar 4.4 — Evaluasi Model: Kartu performa Gradient Boosting dan tabel perbandingan 3 model')
    story.append(PageBreak())

    # Screenshot 5 — Interpretasi
    story.append(Paragraph('4.4.5 Halaman Interpretasi & Insights Bisnis', H3))
    story.append(Paragraph(
        'Halaman ini menyajikan analisis feature importance dari ketiga model, justifikasi '
        'pemilihan Gradient Boosting sebagai model terbaik, serta rekomendasi strategis '
        'bagi fasilitas kesehatan. Feature terpenting dari Gradient Boosting adalah '
        '<b>Insulin</b> (0,6879), diikuti Glucose (0,1082) dan Age (0,0712).', BODY))
    story += img('05_interpretasi.png', width=W,
                 caption='Gambar 4.5 — Interpretasi: Feature Importance Gradient Boosting dan justifikasi model terbaik')
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # BAB 5 — KESIMPULAN
    # ════════════════════════════════════════════════════════════
    story.append(Paragraph('5. Kesimpulan dan Rekomendasi', H1))
    story.append(hr())

    story.append(Paragraph('5.1 Kesimpulan', H2))
    conclusions = [
        'Semua target metrik berhasil dicapai: Accuracy 89,66% (≥75%), F1-Score 0,8537 (≥0,70), AUC-ROC 0,9582 (≥0,80), dan Recall 0,8750 (≥0,75).',
        'Insulin dan Glucose adalah prediktor terkuat pada model Gradient Boosting terbaik.',
        'Machine Learning terbukti efektif untuk skrining awal diabetes dengan akurasi kompetitif.',
        'Gradient Boosting memberikan keseimbangan terbaik antara Precision dan Recall.',
        'Aplikasi Streamlit DiabetesSense berhasil dibangun dan di-deploy secara online, dapat diakses tanpa instalasi.',
    ]
    for i, c in enumerate(conclusions, 1):
        story.append(Paragraph(f'<b>{i}.</b> {c}', BULLET))
    story.append(sp(6))

    story.append(Paragraph('5.2 Rekomendasi', H2))
    story.append(Paragraph('Rekomendasi Klinis / Bisnis:', H3))
    rec_klinis = [
        'Implementasikan sistem skrining ML ini di fasilitas kesehatan primer (puskesmas, klinik)',
        'Prioritaskan pemeriksaan kadar insulin, glukosa, dan BMI pada setiap kunjungan',
        'Gunakan threshold prediksi ≥0,4 untuk meminimalkan false negative yang berbahaya',
    ]
    for r in rec_klinis:
        story.append(Paragraph(f'• {r}', BULLET))

    story.append(Paragraph('Rekomendasi Teknis:', H3))
    rec_tech = [
        'Implementasikan SMOTE untuk oversampling minority class secara lebih agresif',
        'Eksplorasi XGBoost, LightGBM, dan ensemble stacking untuk performa lebih tinggi',
        'Gunakan SHAP (SHapley Additive exPlanations) untuk interpretabilitas yang lebih mendalam',
        'Validasi model pada dataset yang lebih besar dan populasi yang lebih beragam',
    ]
    for r in rec_tech:
        story.append(Paragraph(f'• {r}', BULLET))

    story.append(Paragraph('5.3 Keterbatasan', H2))
    lims = [
        'Dataset terbatas (768 sampel) dari populasi spesifik (perempuan Pima Indian ≥21 tahun)',
        'Validasi eksternal belum dilakukan pada populasi berbeda atau data Indonesia',
        'Fitur penting seperti HbA1c dan kolesterol tidak tersedia dalam dataset',
    ]
    for l in lims:
        story.append(Paragraph(f'• {l}', BULLET))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # BAB 6 — REFERENSI
    # ════════════════════════════════════════════════════════════
    story.append(Paragraph('6. Referensi', H1))
    story.append(hr())
    refs = [
        'Smith, J.W. et al. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. Proceedings of SCAMC, 261–265.',
        'Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.',
        'International Diabetes Federation. (2021). IDF Diabetes Atlas, 10th Edition. Brussels: IDF.',
        'Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.',
        'Friedman, J.H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Ann. Stat., 29(5), 1189–1232.',
        'Hosmer, D.W. & Lemeshow, S. (2000). Applied Logistic Regression (2nd ed.). Wiley.',
        'James, G. et al. (2021). An Introduction to Statistical Learning (2nd ed.). Springer.',
        'Streamlit Inc. (2024). Streamlit — The fastest way to build data apps. https://streamlit.io',
        'Kementerian Kesehatan RI. (2018). Hasil Utama Riskesdas 2018. Balitbangkes.',
        'American Diabetes Association. (2023). Standards of Medical Care in Diabetes. Diabetes Care, 46(S1).',
    ]
    for i, r in enumerate(refs, 1):
        story.append(Paragraph(f'[{i}] {r}', S('ref', fontName='Helvetica', fontSize=9,
                     textColor=HexColor('#374151'), spaceAfter=7, leading=14,
                     leftIndent=20, firstLineIndent=-20)))
    story.append(sp(20))
    story.append(hr(C_NAVY, 1.5))
    story.append(sp(8))
    story.append(Paragraph(
        '<i>Laporan ini merupakan bagian dari keluaran wajib UAS Pembelajaran Mesin<br/>'
        'Semester Genap 2025/2026 — Teknik Informatika — UDINUS Semarang</i>',
        S('footer', fontName='Helvetica-Oblique', fontSize=9, textColor=C_GRAY,
          alignment=TA_CENTER)))

    # ── Build ──
    doc.build(story)
    print(f'PDF berhasil dibuat: {OUT}')
    print(f'   Ukuran: {OUT.stat().st_size / 1024:.1f} KB')

if __name__ == '__main__':
    build()
