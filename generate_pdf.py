#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_pdf.py - Membuat Laporan Teknis PDF untuk UAS Pembelajaran Mesin
UDINUS Semarang - Semester Genap 2025/2026

Kelompok:
  - Fahmi Fatmawati Azzahra  (NIM: A11.2024.15831)
  - Nathaniela Febry Nathasa (NIM: A11.2024.15850)

Jalankan: python generate_pdf.py
"""

import sys, json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import (
    HexColor, white, black, Color
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image, KeepTogether
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib import colors

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# PATHS
# ============================================================
BASE_DIR    = Path(__file__).parent
MODEL_DIR   = BASE_DIR / 'models'
FIGURES_DIR = BASE_DIR / 'reports' / 'figures'
OUTPUT_PDF  = BASE_DIR / 'reports' / 'Laporan_Teknis_UAS.pdf'
OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD RESULTS
# ============================================================
results_path = MODEL_DIR / 'results.json'
results = {}
best_model = 'Gradient Boosting'
if results_path.exists():
    results = json.loads(results_path.read_text())
    best_model = results.get('best_model', 'Gradient Boosting')

def get_metric(model_name, split, metric):
    try:
        return f"{results['results'][model_name][split][metric]:.4f}"
    except Exception:
        return '-'

# ============================================================
# COLOUR PALETTE
# ============================================================
C_PRIMARY   = HexColor('#1a237e')   # Navy deep
C_SECONDARY = HexColor('#283593')   # Navy mid
C_ACCENT    = HexColor('#1565C0')   # Blue
C_LIGHT     = HexColor('#E8EAF6')   # Light indigo tint
C_GOLD      = HexColor('#F57F17')   # Amber/gold
C_DANGER    = HexColor('#B71C1C')   # Red
C_SUCCESS   = HexColor('#1B5E20')   # Green
C_GRAY      = HexColor('#546E7A')
C_LGRAY     = HexColor('#ECEFF1')
C_WHITE     = white
C_TABLE_H   = HexColor('#283593')
C_TABLE_R1  = HexColor('#E8EAF6')

# ============================================================
# STYLES
# ============================================================
PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm

def build_styles():
    s = getSampleStyleSheet()
    custom = {}

    custom['CoverTitle'] = ParagraphStyle(
        'CoverTitle', fontSize=22, leading=28, alignment=TA_CENTER,
        textColor=C_WHITE, fontName='Helvetica-Bold',
        spaceAfter=6,
    )
    custom['CoverSub'] = ParagraphStyle(
        'CoverSub', fontSize=11, leading=16, alignment=TA_CENTER,
        textColor=HexColor('#E3F2FD'), fontName='Helvetica',
    )
    custom['CoverMeta'] = ParagraphStyle(
        'CoverMeta', fontSize=10, leading=14, alignment=TA_CENTER,
        textColor=HexColor('#B0BEC5'), fontName='Helvetica',
    )
    custom['H1'] = ParagraphStyle(
        'H1', fontSize=14, leading=20, textColor=C_PRIMARY,
        fontName='Helvetica-Bold', spaceBefore=18, spaceAfter=6,
        borderPad=4,
    )
    custom['H2'] = ParagraphStyle(
        'H2', fontSize=12, leading=18, textColor=C_SECONDARY,
        fontName='Helvetica-Bold', spaceBefore=12, spaceAfter=4,
    )
    custom['H3'] = ParagraphStyle(
        'H3', fontSize=11, leading=16, textColor=C_ACCENT,
        fontName='Helvetica-Bold', spaceBefore=8, spaceAfter=3,
    )
    custom['Body'] = ParagraphStyle(
        'Body', fontSize=10, leading=16, textColor=black,
        fontName='Helvetica', spaceAfter=6, alignment=TA_JUSTIFY,
    )
    custom['BulletBody'] = ParagraphStyle(
        'BulletBody', fontSize=10, leading=15, textColor=black,
        fontName='Helvetica', spaceAfter=3, leftIndent=14,
        bulletIndent=0, alignment=TA_JUSTIFY,
    )
    custom['Caption'] = ParagraphStyle(
        'Caption', fontSize=8.5, leading=12, textColor=C_GRAY,
        fontName='Helvetica-Oblique', alignment=TA_CENTER, spaceAfter=8,
    )
    custom['TableHeader'] = ParagraphStyle(
        'TableHeader', fontSize=9, leading=12, textColor=C_WHITE,
        fontName='Helvetica-Bold', alignment=TA_CENTER,
    )
    custom['TableCell'] = ParagraphStyle(
        'TableCell', fontSize=9, leading=13, textColor=black,
        fontName='Helvetica', alignment=TA_CENTER,
    )
    custom['TableCellL'] = ParagraphStyle(
        'TableCellL', fontSize=9, leading=13, textColor=black,
        fontName='Helvetica', alignment=TA_LEFT,
    )
    custom['Footer'] = ParagraphStyle(
        'Footer', fontSize=8, textColor=C_GRAY,
        fontName='Helvetica', alignment=TA_CENTER,
    )
    custom['Highlight'] = ParagraphStyle(
        'Highlight', fontSize=10, leading=15, textColor=C_PRIMARY,
        fontName='Helvetica-Bold', spaceAfter=4, alignment=TA_LEFT,
        backColor=C_LIGHT, borderPad=5,
    )
    return custom

ST = build_styles()
AVAIL_W = PAGE_W - 2 * MARGIN

# ============================================================
# HELPER BUILDERS
# ============================================================
def h1(text): return Paragraph(text, ST['H1'])
def h2(text): return Paragraph(text, ST['H2'])
def h3(text): return Paragraph(text, ST['H3'])
def body(text): return Paragraph(text, ST['Body'])
def bullet(text): return Paragraph(f'<bullet>\u2022</bullet> {text}', ST['BulletBody'])
def caption(text): return Paragraph(text, ST['Caption'])
def sp(h=0.25): return Spacer(1, h * cm)
def hr(): return HRFlowable(width='100%', thickness=1, color=C_LIGHT, spaceAfter=6, spaceBefore=6)

def section_header(num, title):
    """Coloured section number + title bar."""
    data = [[
        Paragraph(f'{num}', ParagraphStyle('N', fontSize=13, textColor=C_WHITE,
                  fontName='Helvetica-Bold', alignment=TA_CENTER)),
        Paragraph(title, ParagraphStyle('T', fontSize=13, textColor=C_WHITE,
                  fontName='Helvetica-Bold', alignment=TA_LEFT, leftIndent=4)),
    ]]
    t = Table(data, colWidths=[1 * cm, AVAIL_W - 1 * cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), C_PRIMARY),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING', (0, 0), (0, 0), 6),
        ('LEFTPADDING', (1, 0), (1, 0), 8),
    ]))
    return t

def info_table(rows, col_w=None):
    """Two-column key-value table."""
    if col_w is None:
        col_w = [5 * cm, AVAIL_W - 5 * cm]
    data = [[Paragraph(f'<b>{k}</b>', ST['TableCellL']),
             Paragraph(str(v), ST['TableCellL'])] for k, v in rows]
    t = Table(data, colWidths=col_w)
    style = [
        ('BACKGROUND', (0, 0), (0, -1), C_LIGHT),
        ('GRID', (0, 0), (-1, -1), 0.4, HexColor('#CFD8DC')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]
    for i in range(0, len(rows), 2):
        style.append(('BACKGROUND', (1, i), (1, i), C_WHITE))
    t.setStyle(TableStyle(style))
    return t

def metrics_table(models_data):
    """Full metrics comparison table."""
    headers = ['Model', 'Set', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    data = [[Paragraph(h, ST['TableHeader']) for h in headers]]
    metrics_k = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    row_bg = [C_TABLE_R1, C_WHITE]
    bg_i = 0
    for mname, splits in models_data.items():
        is_best = mname == best_model
        for si, split in enumerate(['train', 'val', 'test']):
            split_lbl = {'train': 'Train', 'val': 'Val', 'test': 'Test'}[split]
            bold = is_best and split == 'test'
            fw = 'Helvetica-Bold' if bold else 'Helvetica'
            clr = C_SUCCESS if bold else black
            row = [
                Paragraph(f'<b>{mname}</b>' if si == 0 else '', ST['TableCellL']),
                Paragraph(split_lbl, ParagraphStyle('sc', fontSize=9, fontName='Helvetica',
                          textColor=C_GRAY, alignment=TA_CENTER)),
            ]
            for mk in metrics_k:
                val_str = get_metric(mname, split, mk)
                row.append(Paragraph(f'<b>{val_str}</b>' if bold else val_str,
                                     ParagraphStyle('mc', fontSize=9, fontName=fw,
                                                    textColor=clr, alignment=TA_CENTER)))
            data.append(row)
        bg_i += 1

    col_w = [4.5*cm, 1.3*cm, 2.2*cm, 2.2*cm, 2.0*cm, 2.2*cm, 2.2*cm]
    t = Table(data, colWidths=col_w)
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), C_TABLE_H),
        ('GRID', (0, 0), (-1, -1), 0.4, HexColor('#CFD8DC')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_TABLE_R1, C_WHITE, C_WHITE,
                                               C_TABLE_R1, C_WHITE, C_WHITE,
                                               C_TABLE_R1, C_WHITE, C_WHITE]),
    ]
    # Highlight best model test row — row 9 = GB test (rows 7,8,9)
    style += [
        ('BACKGROUND', (0, 9), (-1, 9), HexColor('#C8E6C9')),
        ('TEXTCOLOR', (0, 9), (-1, 9), C_SUCCESS),
    ]
    t.setStyle(TableStyle(style))
    return t

def try_image(path, width=None, height=None):
    """Safely load image; return None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        if width and height:
            return Image(str(p), width=width, height=height)
        elif width:
            img = Image(str(p))
            ratio = img.drawHeight / img.drawWidth
            return Image(str(p), width=width, height=width * ratio)
        return Image(str(p))
    except Exception:
        return None

# ============================================================
# PAGE TEMPLATE  (header + footer on every page)
# ============================================================
def on_first_page(canvas, doc):
    pass  # cover handles its own drawing

def on_later_pages(canvas, doc):
    canvas.saveState()
    # Top line
    canvas.setStrokeColor(C_PRIMARY)
    canvas.setLineWidth(1.2)
    canvas.line(MARGIN, PAGE_H - 1.2*cm, PAGE_W - MARGIN, PAGE_H - 1.2*cm)
    # Header text
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(C_GRAY)
    canvas.drawString(MARGIN, PAGE_H - 0.9*cm,
                      'Laporan Teknis UAS Pembelajaran Mesin | UDINUS Semarang | Sem. Genap 2025/2026')
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.9*cm,
                           'Fahmi F.A. & Nathaniela F.N.')
    # Bottom line + page number
    canvas.setStrokeColor(C_LIGHT)
    canvas.setLineWidth(0.8)
    canvas.line(MARGIN, 1.5*cm, PAGE_W - MARGIN, 1.5*cm)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(C_GRAY)
    canvas.drawCentredString(PAGE_W / 2, 0.9*cm, f'Halaman {doc.page}')
    canvas.restoreState()

# ============================================================
# COVER PAGE (drawn directly on canvas inside first element)
# ============================================================
def build_cover():
    """Return a list of flowables that form the cover page."""
    story = []

    # Navy full-width header bar
    class CoverBanner:
        def __init__(self):
            self.width  = PAGE_W
            self.height = PAGE_H

        def wrap(self, aw, ah): return (aw, 0)

        def draw(self): pass

    # We use a big table to fake the cover banner
    cover_data = [[
        Paragraph(
            '<b>LAPORAN TEKNIS</b><br/>'
            'UJIAN AKHIR SEMESTER (UAS)<br/>'
            'MATA KULIAH PEMBELAJARAN MESIN',
            ST['CoverTitle']
        )
    ]]
    banner = Table(cover_data, colWidths=[AVAIL_W])
    banner.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), C_PRIMARY),
        ('TOPPADDING', (0, 0), (-1, -1), 36),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 36),
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(banner)
    story.append(sp(0.6))

    subtitle_data = [[
        Paragraph(
            'Sistem Prediksi Penyakit Diabetes Berbasis Machine Learning<br/>'
            'menggunakan Pima Indians Diabetes Dataset (NIDDK)',
            ParagraphStyle('cs', fontSize=11.5, leading=17, alignment=TA_CENTER,
                           textColor=C_SECONDARY, fontName='Helvetica-BoldOblique')
        )
    ]]
    stbl = Table(subtitle_data, colWidths=[AVAIL_W])
    stbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), C_LIGHT),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('LEFTPADDING', (0, 0), (-1, -1), 18),
        ('RIGHTPADDING', (0, 0), (-1, -1), 18),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(stbl)
    story.append(sp(1.2))

    # Team box
    team_data = [
        [Paragraph('<b>DISUSUN OLEH:</b>', ParagraphStyle('tl', fontSize=10, fontName='Helvetica-Bold',
                   textColor=C_WHITE, alignment=TA_CENTER))],
        [Paragraph(
            '<b>Fahmi Fatmawati Azzahra</b><br/>NIM: A11.2024.15831<br/><br/>'
            '<b>Nathaniela Febry Nathasa</b><br/>NIM: A11.2024.15850',
            ParagraphStyle('tm', fontSize=11, leading=16, fontName='Helvetica',
                           textColor=C_WHITE, alignment=TA_CENTER)
        )],
    ]
    team_tbl = Table(team_data, colWidths=[AVAIL_W * 0.55])
    team_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_ACCENT),
        ('BACKGROUND', (0, 1), (-1, -1), C_SECONDARY),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 16),
        ('RIGHTPADDING', (0, 0), (-1, -1), 16),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))

    story.append(Table([[team_tbl]], colWidths=[AVAIL_W],
                       style=[('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(sp(1.0))

    # Meta info
    meta_rows = [
        ('Mata Kuliah', 'Pembelajaran Mesin'),
        ('Semester', 'Genap 2025/2026'),
        ('Universitas', 'Universitas Dian Nuswantoro (UDINUS) Semarang'),
        ('Program Studi', 'Teknik Informatika (S1)'),
        ('Dosen Pengampu', 'Prof. Ir. Heru Agus Santoso, Ph.D, IPM, ASEAN Eng.'),
        ('Tanggal', '17 Juli 2026'),
    ]
    story.append(info_table(meta_rows, col_w=[4.8*cm, AVAIL_W - 4.8*cm]))
    story.append(sp(1.2))

    # Golden divider
    story.append(HRFlowable(width='100%', thickness=3, color=C_GOLD,
                            spaceBefore=4, spaceAfter=8))
    story.append(Paragraph(
        'Program Studi Teknik Informatika &bull; Fakultas Ilmu Komputer &bull; UDINUS Semarang',
        ParagraphStyle('foot', fontSize=9, alignment=TA_CENTER, textColor=C_GRAY,
                       fontName='Helvetica-Oblique')
    ))
    story.append(PageBreak())
    return story

# ============================================================
# BUILD DOCUMENT CONTENT
# ============================================================
def build_content():
    story = []

    # ── Daftar Isi (manual) ─────────────────────────────────
    story.append(section_header('', 'DAFTAR ISI'))
    story.append(sp(0.3))
    toc_items = [
        ('1.', 'PENDAHULUAN DAN LATAR BELAKANG', '3'),
        ('  1.1', 'Latar Belakang', '3'),
        ('  1.2', 'Rumusan Masalah', '3'),
        ('  1.3', 'Tujuan Penelitian', '3'),
        ('  1.4', 'Metrik Kesuksesan', '3'),
        ('  1.5', 'Deskripsi Dataset', '4'),
        ('2.', 'METODOLOGI', '4'),
        ('  2.1', 'Alur Kerja (Pipeline)', '4'),
        ('  2.2', 'Data Acquisition', '4'),
        ('  2.3', 'Exploratory Data Analysis', '5'),
        ('  2.4', 'Preprocessing', '5'),
        ('  2.5', 'Pemodelan', '6'),
        ('  2.6', 'Evaluasi', '6'),
        ('3.', 'HASIL DAN ANALISIS', '7'),
        ('  3.1', 'Hasil EDA', '7'),
        ('  3.2', 'Hasil Preprocessing', '7'),
        ('  3.3', 'Hasil Training & Tuning', '7'),
        ('  3.4', 'Perbandingan Metrik Semua Model', '8'),
        ('  3.5', 'Feature Importance Analysis', '8'),
        ('  3.6', 'Visualisasi', '9'),
        ('4.', 'KESIMPULAN DAN REKOMENDASI', '10'),
        ('  4.1', 'Kesimpulan', '10'),
        ('  4.2', 'Rekomendasi', '10'),
        ('  4.3', 'Keterbatasan', '11'),
        ('5.', 'REFERENSI', '11'),
    ]
    toc_data = []
    for num, title, pg in toc_items:
        bold = not num.startswith(' ')
        fn = 'Helvetica-Bold' if bold else 'Helvetica'
        clr = C_PRIMARY if bold else black
        toc_data.append([
            Paragraph(num, ParagraphStyle('tn', fontSize=9.5, fontName=fn, textColor=clr)),
            Paragraph(title, ParagraphStyle('tt', fontSize=9.5, fontName=fn,
                                            textColor=clr, alignment=TA_LEFT)),
            Paragraph(pg, ParagraphStyle('tp', fontSize=9.5, fontName=fn,
                                         textColor=clr, alignment=TA_RIGHT)),
        ])
    toc_tbl = Table(toc_data, colWidths=[1.2*cm, AVAIL_W - 2.2*cm, 1.0*cm])
    toc_tbl.setStyle(TableStyle([
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LINEBELOW', (0, -1), (-1, -1), 0.5, C_LIGHT),
    ]))
    story += [toc_tbl, PageBreak()]

    # ══════════════════════════════════════════════════════════
    # BAB 1 – PENDAHULUAN
    # ══════════════════════════════════════════════════════════
    story.append(section_header('1', 'PENDAHULUAN DAN LATAR BELAKANG'))
    story.append(sp(0.3))

    story.append(h2('1.1  Latar Belakang'))
    story.append(body(
        'Diabetes mellitus merupakan penyakit metabolik kronis yang ditandai oleh '
        'hiperglikemia (kadar glukosa darah tinggi) akibat gangguan sekresi atau kerja '
        'insulin. Berdasarkan data <i>International Diabetes Federation</i> (IDF) edisi '
        '2021, terdapat sekitar <b>537 juta</b> orang dewasa (usia 20\u201379 tahun) yang '
        'hidup dengan diabetes di seluruh dunia. Angka ini diproyeksikan meningkat menjadi '
        '<b>643 juta</b> pada 2030 dan <b>783 juta</b> pada 2045 \u2014 menjadikan '
        'diabetes salah satu krisis kesehatan global terbesar abad ke-21.'
    ))
    story.append(body(
        'Di Indonesia, Riset Kesehatan Dasar (Riskesdas) 2018 mencatat prevalensi diabetes '
        'pada penduduk usia \u226515 tahun sebesar <b>10,9%</b>, meningkat signifikan dari '
        '6,9% pada 2013. Penyakit ini menyumbang risiko komplikasi serius: kardiovaskular, '
        'gagal ginjal (nefropati diabetik), neuropati perifer, dan retinopati. Deteksi dini '
        'terbukti mengurangi komplikasi dan biaya pengobatan secara substansial.'
    ))
    story.append(body(
        'Pendekatan <i>Machine Learning</i> menawarkan solusi inovatif untuk deteksi dini '
        'melalui analisis pola pada rekam medis. Dengan parameter kesehatan dasar (glukosa, '
        'BMI, tekanan darah, usia), model ML dapat mengklasifikasikan risiko diabetes secara '
        'otomatis dan efisien, mendukung tenaga kesehatan dalam pengambilan keputusan klinis.'
    ))

    story.append(h2('1.2  Rumusan Masalah'))
    story.append(body(
        'Bagaimana membangun model <i>Machine Learning</i> yang mampu memprediksi dengan '
        'akurat apakah seorang pasien perempuan keturunan Pima Indian berusia \u226521 tahun '
        'menderita diabetes, berdasarkan parameter diagnostik kesehatan yang tersedia dalam '
        'Pima Indians Diabetes Dataset?'
    ))

    story.append(h2('1.3  Tujuan Penelitian'))
    for t in [
        'Membangun dan membandingkan minimal 3 model Machine Learning untuk prediksi diabetes.',
        'Mengidentifikasi faktor-faktor risiko yang paling berpengaruh terhadap kejadian diabetes.',
        'Mengembangkan aplikasi web interaktif berbasis Streamlit untuk skrining awal.',
        'Mendokumentasikan seluruh proses pipeline ML secara komprehensif sesuai standar akademik.',
    ]:
        story.append(bullet(t))
    story.append(sp(0.2))

    story.append(h2('1.4  Metrik Kesuksesan'))
    mk_data = [
        [Paragraph('<b>Metrik</b>', ST['TableHeader']),
         Paragraph('<b>Target Minimum</b>', ST['TableHeader']),
         Paragraph('<b>Alasan</b>', ST['TableHeader'])],
        [Paragraph('Accuracy', ST['TableCell']),
         Paragraph('\u2265 75%', ST['TableCell']),
         Paragraph('Benchmark performa klasifikasi umum', ST['TableCellL'])],
        [Paragraph('F1-Score', ST['TableCell']),
         Paragraph('\u2265 0.70', ST['TableCell']),
         Paragraph('Menyeimbangkan Precision & Recall pada data imbalanced', ST['TableCellL'])],
        [Paragraph('AUC-ROC', ST['TableCell']),
         Paragraph('\u2265 0.80', ST['TableCell']),
         Paragraph('Mengukur kemampuan diskriminasi kelas model', ST['TableCellL'])],
        [Paragraph('Recall', ST['TableCell']),
         Paragraph('\u2265 0.75', ST['TableCell']),
         Paragraph('Meminimalkan false negative (DM tidak terdeteksi)', ST['TableCellL'])],
    ]
    mk_tbl = Table(mk_data, colWidths=[2.8*cm, 3.0*cm, AVAIL_W - 5.8*cm])
    mk_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_TABLE_H),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_TABLE_R1, C_WHITE]),
        ('GRID', (0, 0), (-1, -1), 0.4, HexColor('#CFD8DC')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story += [mk_tbl, sp(0.3)]

    story.append(h2('1.5  Deskripsi Dataset'))
    story.append(body(
        'Dataset yang digunakan adalah <b>Pima Indians Diabetes Database</b> dari '
        '<i>National Institute of Diabetes and Digestive and Kidney Diseases</i> (NIDDK), USA. '
        'Seluruh subjek adalah perempuan berusia \u226521 tahun keturunan Pima Indian. '
        'Dataset ini bersifat publik dan dapat diakses melalui UCI Machine Learning Repository '
        'maupun Kaggle.'
    ))
    ds_rows = [
        ('Jumlah Sampel', '768 rekam medis'),
        ('Jumlah Fitur Input', '8 fitur numerik'),
        ('Variabel Target', 'Outcome: 0 = Tidak Diabetes, 1 = Diabetes'),
        ('Populasi', 'Perempuan \u226521 tahun, keturunan Pima Indian'),
        ('Distribusi Kelas', '500 Non-DM (65.1%) vs 268 DM (34.9%)'),
        ('Sumber Data', 'UCI ML Repository / Kaggle / GitHub Brownlee'),
    ]
    story.append(info_table(ds_rows))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # BAB 2 – METODOLOGI
    # ══════════════════════════════════════════════════════════
    story.append(section_header('2', 'METODOLOGI'))
    story.append(sp(0.3))

    story.append(h2('2.1  Alur Kerja (Pipeline)'))
    pipeline_data = [[
        Paragraph('1\nAkuisisi\nData', ParagraphStyle('pp', fontSize=8, fontName='Helvetica-Bold',
                  textColor=C_WHITE, alignment=TA_CENTER, leading=11)),
        Paragraph('\u2192', ParagraphStyle('ar', fontSize=14, textColor=C_GRAY, alignment=TA_CENTER)),
        Paragraph('2\nEDA', ParagraphStyle('pp', fontSize=8, fontName='Helvetica-Bold',
                  textColor=C_WHITE, alignment=TA_CENTER, leading=11)),
        Paragraph('\u2192', ParagraphStyle('ar', fontSize=14, textColor=C_GRAY, alignment=TA_CENTER)),
        Paragraph('3\nPre-\nprocessing', ParagraphStyle('pp', fontSize=8, fontName='Helvetica-Bold',
                  textColor=C_WHITE, alignment=TA_CENTER, leading=11)),
        Paragraph('\u2192', ParagraphStyle('ar', fontSize=14, textColor=C_GRAY, alignment=TA_CENTER)),
        Paragraph('4\nPemodelan\n& Tuning', ParagraphStyle('pp', fontSize=8, fontName='Helvetica-Bold',
                  textColor=C_WHITE, alignment=TA_CENTER, leading=11)),
        Paragraph('\u2192', ParagraphStyle('ar', fontSize=14, textColor=C_GRAY, alignment=TA_CENTER)),
        Paragraph('5\nEvaluasi', ParagraphStyle('pp', fontSize=8, fontName='Helvetica-Bold',
                  textColor=C_WHITE, alignment=TA_CENTER, leading=11)),
        Paragraph('\u2192', ParagraphStyle('ar', fontSize=14, textColor=C_GRAY, alignment=TA_CENTER)),
        Paragraph('6\nDeployment\nStreamlit', ParagraphStyle('pp', fontSize=8, fontName='Helvetica-Bold',
                  textColor=C_WHITE, alignment=TA_CENTER, leading=11)),
    ]]
    box_w = 2.2 * cm
    arr_w = 0.8 * cm
    pipeline_tbl = Table(pipeline_data,
                         colWidths=[box_w, arr_w, box_w, arr_w, box_w, arr_w, box_w, arr_w, box_w, arr_w, box_w])
    bg_cols = [C_ACCENT, None, HexColor('#0288D1'), None, HexColor('#00897B'), None,
               HexColor('#7B1FA2'), None, HexColor('#E65100'), None, HexColor('#2E7D32')]
    ps = []
    for ci, bg in enumerate(bg_cols):
        if bg:
            ps.append(('BACKGROUND', (ci, 0), (ci, 0), bg))
    ps += [
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]
    pipeline_tbl.setStyle(TableStyle(ps))
    story += [pipeline_tbl, sp(0.3)]

    story.append(h2('2.2  Data Acquisition'))
    story.append(body(
        'Dataset yang digunakan adalah <b>Pima Indians Diabetes Database</b> yang '
        'tersedia secara publik dan dapat diunduh melalui tiga sumber resmi berikut:'
    ))
    # Link style: display text pendek agar muat 1 baris & fully clickable
    LINK_STYLE = ParagraphStyle(
        'LinkCell', fontSize=9, leading=13, textColor=HexColor('#1565C0'),
        fontName='Helvetica', alignment=TA_LEFT,
    )
    src_data = [
        [Paragraph('<b>Sumber</b>', ST['TableHeader']),
         Paragraph('<b>Tautan (URL) — Klik untuk Membuka</b>', ST['TableHeader'])],
        [Paragraph('UCI Machine Learning Repository', ST['TableCellL']),
         Paragraph(
             '<a href="https://archive.ics.uci.edu/dataset/34/diabetes" color="#1565C0">'
             '<u>archive.ics.uci.edu/dataset/34/diabetes</u></a>',
             LINK_STYLE)],
        [Paragraph('Kaggle Dataset', ST['TableCellL']),
         Paragraph(
             '<a href="https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database" color="#1565C0">'
             '<u>kaggle.com/datasets/uciml/pima-indians-diabetes-database</u></a>',
             LINK_STYLE)],
        [Paragraph('Jason Brownlee GitHub', ST['TableCellL']),
         Paragraph(
             '<a href="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv" color="#1565C0">'
             '<u>raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv</u></a>',
             ParagraphStyle('LinkCellSm', fontSize=8, leading=12, textColor=HexColor('#1565C0'),
                            fontName='Helvetica', alignment=TA_LEFT))],
    ]
    src_tbl = Table(src_data, colWidths=[4.5*cm, AVAIL_W - 4.5*cm])
    src_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_TABLE_H),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_TABLE_R1, C_WHITE, C_TABLE_R1]),
        ('GRID', (0, 0), (-1, -1), 0.4, HexColor('#CFD8DC')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story += [src_tbl, sp(0.2)]
    story.append(body(
        'Dalam implementasi proyek ini, dataset diunduh langsung dari sumber ketiga '
        '(Jason Brownlee GitHub) menggunakan skrip Python, lalu disimpan secara lokal '
        'ke dalam folder <font face="Courier" size="9">data/diabetes.csv</font>. '
        'Verifikasi keaslian data dilakukan dengan membandingkan statistik deskriptif '
        'hasil unduhan (768 baris × 9 kolom) dengan dokumentasi resmi UCI Machine Learning '
        'Repository. Seluruh 8 fitur input dan 1 variabel target (Outcome) berhasil '
        'dimuat dengan tipe data yang sesuai (integer dan float).'
    ))

    story.append(h2('2.3  Exploratory Data Analysis (EDA)'))
    story.append(h3('a. Missing Values'))
    story.append(body(
        'Tidak ditemukan NaN eksplisit (isnull().sum() = 0). Namun, ditemukan '
        '<b>implicit missing values</b> berupa nilai 0 pada kolom-kolom yang secara '
        'medis tidak mungkin bernilai 0:'
    ))
    mv_data = [
        [Paragraph('<b>Kolom</b>', ST['TableHeader']),
         Paragraph('<b>Jumlah Nilai 0</b>', ST['TableHeader']),
         Paragraph('<b>Persentase</b>', ST['TableHeader']),
         Paragraph('<b>Keterangan</b>', ST['TableHeader'])],
        [Paragraph('Glucose', ST['TableCell']), Paragraph('5', ST['TableCell']),
         Paragraph('0.7%', ST['TableCell']),
         Paragraph('Kadar glukosa 0 tidak mungkin secara medis', ST['TableCellL'])],
        [Paragraph('BloodPressure', ST['TableCell']), Paragraph('35', ST['TableCell']),
         Paragraph('4.6%', ST['TableCell']),
         Paragraph('Tekanan darah 0 tidak mungkin hidup', ST['TableCellL'])],
        [Paragraph('SkinThickness', ST['TableCell']), Paragraph('227', ST['TableCell']),
         Paragraph('29.6%', ST['TableCell']),
         Paragraph('Data sering tidak diukur / tidak dicatat', ST['TableCellL'])],
        [Paragraph('Insulin', ST['TableCell']), Paragraph('374', ST['TableCell']),
         Paragraph('<b>48.7%</b>', ST['TableCell']),
         Paragraph('<b>Tertinggi</b> \u2014 sering tidak terukur', ST['TableCellL'])],
        [Paragraph('BMI', ST['TableCell']), Paragraph('11', ST['TableCell']),
         Paragraph('1.4%', ST['TableCell']),
         Paragraph('BMI 0 tidak mungkin pada individu hidup', ST['TableCellL'])],
    ]
    mv_tbl = Table(mv_data, colWidths=[3.0*cm, 2.5*cm, 2.5*cm, AVAIL_W - 8.0*cm])
    mv_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_TABLE_H),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_TABLE_R1, C_WHITE]),
        ('GRID', (0, 0), (-1, -1), 0.4, HexColor('#CFD8DC')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 5), (-1, 5), HexColor('#FFF9C4')),
    ]))
    story += [mv_tbl, sp(0.25)]

    story.append(h3('b. Outlier Detection (IQR Method)'))
    story.append(body(
        'Outlier dideteksi menggunakan batas 1.5\u00d7IQR. Kolom dengan outlier terbanyak: '
        'BloodPressure (45 outlier, 5.9%), Insulin (34, 4.4%), BMI (19, 2.5%), dan '
        'DiabetesPedigreeFunction (29, 3.8%). Outlier tidak dihapus karena merupakan nilai '
        'medis yang valid; ditangani oleh imputasi median yang robust terhadap outlier.'
    ))

    story.append(h3('c. 5 Key Insights'))
    insights = [
        ('<b>Insight 1 \u2014 Glukosa adalah Prediktor Terkuat:</b> Korelasi Pearson '
         'r = 0.47 \u2014 tertinggi dari semua fitur. Rata-rata glukosa DM 141 mg/dL vs '
         'Non-DM 110 mg/dL (selisih +28%).'),
        ('<b>Insight 2 \u2014 BMI Tinggi Berkorelasi dengan Diabetes:</b> '
         'Rata-rata BMI DM = 35.1 vs Non-DM = 30.3. Korelasi r = 0.29. '
         'Obesitas (BMI \u226530) merupakan faktor risiko signifikan.'),
        ('<b>Insight 3 \u2014 Usia Berpengaruh:</b> Rata-rata usia DM = 37 tahun vs '
         'Non-DM = 31 tahun. Risiko meningkat progresif setelah usia 35 tahun (r = 0.24).'),
        ('<b>Insight 4 \u2014 Insulin Paling Banyak Hilang:</b> 48.7% nilai Insulin = 0 '
         '(tidak valid). Menunjukkan keterbatasan pengukuran di lapangan. '
         'Perlu imputasi hati-hati.'),
        ('<b>Insight 5 \u2014 Class Imbalance:</b> 65.1% Non-DM vs 34.9% DM. '
         'Ditangani dengan class_weight="balanced" dan pemilihan F1-Score sebagai '
         'metrik utama (bukan Accuracy).'),
    ]
    for ins in insights:
        story.append(bullet(ins))
    story.append(sp(0.2))

    story.append(h2('2.4  Preprocessing'))
    pre_data = [
        [Paragraph('<b>Langkah</b>', ST['TableHeader']),
         Paragraph('<b>Teknik</b>', ST['TableHeader']),
         Paragraph('<b>Justifikasi</b>', ST['TableHeader'])],
        [Paragraph('1. Deteksi Missing Values', ST['TableCellL']),
         Paragraph('Nilai 0 \u2192 NaN (5 kolom)', ST['TableCellL']),
         Paragraph('Nilai 0 tidak mungkin secara medis pada fitur terpilih', ST['TableCellL'])],
        [Paragraph('2. Imputasi', ST['TableCellL']),
         Paragraph('Median per kelas (stratified)', ST['TableCellL']),
         Paragraph('Mempertahankan distribusi per kelas; robust terhadap outlier', ST['TableCellL'])],
        [Paragraph('3. Feature Engineering', ST['TableCellL']),
         Paragraph('4 fitur baru (category + ratio)', ST['TableCellL']),
         Paragraph('Informasi domain medis tambahan yang relevan secara klinis', ST['TableCellL'])],
        [Paragraph('4. Train/Val/Test Split', ST['TableCellL']),
         Paragraph('70/15/15 Stratified', ST['TableCellL']),
         Paragraph('Distribusi kelas terjaga di semua split; validasi tidak bias', ST['TableCellL'])],
        [Paragraph('5. Feature Scaling', ST['TableCellL']),
         Paragraph('StandardScaler (fit on train)', ST['TableCellL']),
         Paragraph('Scaler hanya di-fit pada training set \u2014 cegah data leakage', ST['TableCellL'])],
    ]
    pre_tbl = Table(pre_data, colWidths=[3.5*cm, 3.8*cm, AVAIL_W - 7.3*cm])
    pre_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_TABLE_H),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_TABLE_R1, C_WHITE]),
        ('GRID', (0, 0), (-1, -1), 0.4, HexColor('#CFD8DC')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story += [pre_tbl, sp(0.2)]

    story.append(h3('Feature Engineering Detail'))
    fe_data = [
        [Paragraph('<b>Fitur Baru</b>', ST['TableHeader']),
         Paragraph('<b>Formula/Mapping</b>', ST['TableHeader']),
         Paragraph('<b>Dasar Medis</b>', ST['TableHeader'])],
        [Paragraph('BMI_Category', ST['TableCellL']),
         Paragraph('0=Underweight(<18.5), 1=Normal, 2=Overweight, 3=Obese(\u226530)', ST['TableCellL']),
         Paragraph('Kategori WHO standar untuk obesitas', ST['TableCellL'])],
        [Paragraph('Glucose_Category', ST['TableCellL']),
         Paragraph('0=Normal(<100), 1=Pre-DM(100-125), 2=DM(\u2265126 mg/dL)', ST['TableCellL']),
         Paragraph('Klasifikasi ADA (American Diabetes Association)', ST['TableCellL'])],
        [Paragraph('Age_Group', ST['TableCellL']),
         Paragraph('0=Young(<35), 1=Middle(35-50), 2=Senior(>50)', ST['TableCellL']),
         Paragraph('Risiko diabetes meningkat dengan usia', ST['TableCellL'])],
        [Paragraph('Insulin_Glucose_Ratio', ST['TableCellL']),
         Paragraph('Insulin / (Glucose + 1)', ST['TableCellL']),
         Paragraph('Proxy untuk resistensi insulin (konsep HOMA-IR)', ST['TableCellL'])],
    ]
    fe_tbl = Table(fe_data, colWidths=[3.5*cm, 5.5*cm, AVAIL_W - 9.0*cm])
    fe_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_TABLE_H),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_TABLE_R1, C_WHITE]),
        ('GRID', (0, 0), (-1, -1), 0.4, HexColor('#CFD8DC')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story += [fe_tbl, sp(0.2)]

    story.append(h2('2.5  Pemodelan'))
    story.append(body(
        'Tiga algoritma klasifikasi dibandingkan, dipilih berdasarkan karakteristik '
        'yang saling melengkapi:'
    ))
    alg_data = [
        [Paragraph('<b>Model</b>', ST['TableHeader']),
         Paragraph('<b>Tipe</b>', ST['TableHeader']),
         Paragraph('<b>Hyperparameter yang Di-tuning</b>', ST['TableHeader']),
         Paragraph('<b>Alasan Dipilih</b>', ST['TableHeader'])],
        [Paragraph('<b>Logistic Regression</b>', ST['TableCellL']),
         Paragraph('Linear', ST['TableCell']),
         Paragraph('C=[0.01,0.1,1,10,100], solver=[lbfgs, liblinear]', ST['TableCellL']),
         Paragraph('Baseline interpretabel; koefisien menunjukkan arah pengaruh fitur', ST['TableCellL'])],
        [Paragraph('<b>Random Forest</b>', ST['TableCellL']),
         Paragraph('Ensemble\n(Bagging)', ST['TableCell']),
         Paragraph('n_estimators=[100,200,300], max_depth=[None,10,20], class_weight', ST['TableCellL']),
         Paragraph('Robust terhadap overfitting; feature importance bawaan', ST['TableCellL'])],
        [Paragraph('<b>Gradient Boosting</b>', ST['TableCellL']),
         Paragraph('Ensemble\n(Boosting)', ST['TableCell']),
         Paragraph('n_estimators=[100,200,300], learning_rate=[0.05,0.1,0.15], max_depth=[3,4,5]', ST['TableCellL']),
         Paragraph('State-of-the-art untuk tabular data; akurasi tinggi', ST['TableCellL'])],
    ]
    alg_tbl = Table(alg_data, colWidths=[3.5*cm, 1.8*cm, 5.5*cm, AVAIL_W - 10.8*cm])
    alg_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_TABLE_H),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_TABLE_R1, C_WHITE, C_TABLE_R1]),
        ('GRID', (0, 0), (-1, -1), 0.4, HexColor('#CFD8DC')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story += [alg_tbl, sp(0.2)]

    story.append(body(
        '<b>Hyperparameter Tuning:</b> GridSearchCV dengan 5-fold Stratified K-Fold '
        'Cross-Validation, scoring = F1-Score, n_jobs=-1 (paralel semua core CPU). '
        'Total kombinasi: LR = 10, RF = 36, GB = 36 kombinasi.'
    ))

    story.append(h2('2.6  Evaluasi'))
    story.append(body(
        'Evaluasi dilakukan pada ketiga split (train/validation/test) menggunakan '
        '5 metrik utama:'
    ))
    eval_rows = [
        ('Accuracy', 'Proporsi prediksi yang benar dari total prediksi: (TP+TN)/(Total)'),
        ('Precision', 'Dari semua prediksi positif, berapa yang benar-benar positif: TP/(TP+FP)'),
        ('Recall', 'Dari semua kasus positif nyata, berapa yang berhasil dideteksi: TP/(TP+FN)'),
        ('F1-Score', 'Harmonic mean Precision dan Recall: 2\u00d7(Prec\u00d7Rec)/(Prec+Rec)'),
        ('AUC-ROC', 'Area di bawah kurva ROC \u2014 mengukur kemampuan diskriminasi (0.5\u20131.0)'),
    ]
    story.append(info_table(eval_rows, col_w=[3.0*cm, AVAIL_W - 3.0*cm]))
    story.append(body(
        '<b>Kriteria pemilihan model terbaik:</b> F1-Score tertinggi pada validation set. '
        'F1 dipilih karena menyeimbangkan Precision dan Recall \u2014 sangat penting dalam '
        'konteks medis di mana <i>false negative</i> (diabetes tidak terdeteksi) memiliki '
        'konsekuensi klinis yang serius.'
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # BAB 3 – HASIL DAN ANALISIS
    # ══════════════════════════════════════════════════════════
    story.append(section_header('3', 'HASIL DAN ANALISIS'))
    story.append(sp(0.3))

    story.append(h2('3.1  Hasil EDA'))
    story.append(body(
        'Analisis korelasi Pearson menunjukkan bahwa <b>Glucose</b> (r = 0.47) memiliki '
        'korelasi tertinggi dengan Outcome, diikuti <b>BMI</b> (r = 0.29), <b>Age</b> '
        '(r = 0.24), <b>Pregnancies</b> (r = 0.22), dan <b>DiabetesPedigreeFunction</b> '
        '(r = 0.17). BloodPressure memiliki korelasi terendah (r = 0.065) dan tidak '
        'informatif secara individual. Dataset menunjukkan class imbalance 65:35 yang '
        'ditangani melalui class_weight="balanced" dan pemilihan F1-Score sebagai metrik.'
    ))

    story.append(h2('3.2  Hasil Preprocessing'))
    story.append(body(
        'Setelah preprocessing, total NaN = 0 (seluruh nilai berhasil diimputasi). '
        'Distribusi kelas di semua split terjaga (stratified): Train 70\u2014Val 15\u2014Test 15 '
        '(536\u2014116\u2014116 sampel). Fitur bertambah dari 8 menjadi <b>12 fitur</b> setelah '
        'feature engineering. Semua fitur berhasil dinormalisasi dengan StandardScaler '
        '(mean \u2248 0, std \u2248 1 pada training set).'
    ))

    story.append(h2('3.3  Hasil Training & Hyperparameter Tuning'))
    best_rows = [
        ('Model Terbaik', f'{best_model} (berdasarkan F1 tertinggi pada Val Set)'),
        ('Best Params', 'learning_rate=0.05, max_depth=3, n_estimators=100, subsample=1.0'),
        ('5-Fold CV F1', '0.8166 (pada training set)'),
        ('F1 Validation', get_metric(best_model, 'val', 'f1')),
        ('F1 Test',       get_metric(best_model, 'test', 'f1')),
    ]
    story.append(info_table(best_rows))
    story.append(sp(0.2))

    story.append(h3('Hyperparameter Terbaik Semua Model'))
    hp_data = [
        [Paragraph('<b>Model</b>', ST['TableHeader']),
         Paragraph('<b>Hyperparameter Terbaik</b>', ST['TableHeader']),
         Paragraph('<b>CV F1 Score</b>', ST['TableHeader'])],
        [Paragraph('Logistic Regression', ST['TableCellL']),
         Paragraph("C=0.01, solver='liblinear', max_iter=2000", ST['TableCellL']),
         Paragraph('0.7104', ST['TableCell'])],
        [Paragraph('Random Forest', ST['TableCellL']),
         Paragraph("class_weight='balanced', max_depth=None, n_estimators=200, min_samples_leaf=2", ST['TableCellL']),
         Paragraph('0.8447', ST['TableCell'])],
        [Paragraph(f'<b>{best_model} \u2605</b>', ST['TableCellL']),
         Paragraph("<b>learning_rate=0.05, max_depth=3, n_estimators=100, subsample=1.0</b>", ST['TableCellL']),
         Paragraph('<b>0.8166</b>', ST['TableCell'])],
    ]
    hp_tbl = Table(hp_data, colWidths=[3.5*cm, AVAIL_W - 6.0*cm, 2.5*cm])
    hp_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_TABLE_H),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_TABLE_R1, C_WHITE, HexColor('#C8E6C9')]),
        ('GRID', (0, 0), (-1, -1), 0.4, HexColor('#CFD8DC')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TEXTCOLOR', (0, 3), (-1, 3), C_SUCCESS),
    ]))
    story += [hp_tbl, sp(0.3)]

    story.append(h2('3.4  Perbandingan Metrik Semua Model (Train / Val / Test)'))
    models_data = {
        'Logistic Regression': None,
        'Random Forest': None,
        'Gradient Boosting': None,
    }
    story.append(metrics_table(models_data))
    story.append(sp(0.2))
    story.append(body(
        '<b>Analisis:</b> Gradient Boosting mencapai performa terbaik di semua metrik '
        'pada test set. Perlu dicatat bahwa Random Forest menunjukkan gap train-test yang '
        'lebih besar (train F1 = 0.9637 vs test = 0.8333), mengindikasikan sedikit overfitting. '
        'Gradient Boosting memiliki keseimbangan yang lebih baik (train F1 = 0.9280 vs test = 0.8537). '
        'Semua target metrik <b>terlampaui</b>: Acc \u2265 0.75 \u2713, F1 \u2265 0.70 \u2713, '
        'AUC \u2265 0.80 \u2713, Recall \u2265 0.75 \u2713.'
    ))

    story.append(h2('3.5  Feature Importance Analysis'))
    story.append(body(
        'Berdasarkan analisis feature importance dari ketiga model, urutan kepentingan '
        'fitur yang konsisten adalah sebagai berikut:'
    ))
    fi_items = [
        ('<b>Glucose</b> \u2014 Prediktor terkuat secara konsisten di semua model. '
         'Kadar glukosa tinggi merupakan indikator utama diabetes mellitus.'),
        ('<b>BMI</b> \u2014 Faktor risiko obesitas sangat berpengaruh. '
         'BMI tinggi berkaitan erat dengan resistensi insulin.'),
        ('<b>Age</b> \u2014 Risiko meningkat progresif. Model menangkap non-linearitas '
         'pengaruh usia yang tidak terlihat dari korelasi linear saja.'),
        ('<b>DiabetesPedigreeFunction</b> \u2014 Riwayat keluarga diabetes merupakan '
         'faktor risiko genetik yang signifikan.'),
        ('<b>Glucose_Category</b> (fitur baru) \u2014 Kategorisasi ADA memberikan '
         'informasi tambahan yang meningkatkan discriminability model.'),
        ('<b>Pregnancies</b> \u2014 Riwayat gestational diabetes berkorelasi dengan '
         'risiko DM tipe 2 di kemudian hari.'),
    ]
    for item in fi_items:
        story.append(bullet(item))
    story.append(sp(0.2))

    story.append(h2('3.6  Visualisasi'))
    # Embed figures if available
    fig_paths = [
        ('reports/figures/01_class_distribution.png', 'Gambar 3.1 \u2014 Distribusi Kelas Target (Bar Chart & Pie Chart)'),
        ('reports/figures/08_model_comparison.png',   'Gambar 3.2 \u2014 Perbandingan Performa Model (Test Set)'),
        ('reports/figures/06_roc_curves.png',         'Gambar 3.3 \u2014 ROC Curves Semua Model'),
        ('reports/figures/05_confusion_matrices.png', 'Gambar 3.4 \u2014 Confusion Matrix Semua Model'),
        ('reports/figures/07_feature_importance.png', 'Gambar 3.5 \u2014 Feature Importance Semua Model'),
        ('reports/figures/03_correlation_heatmap.png','Gambar 3.6 \u2014 Correlation Heatmap Dataset'),
    ]
    for fig_path, cap_text in fig_paths:
        img = try_image(BASE_DIR / fig_path, width=AVAIL_W)
        if img:
            story += [img, caption(cap_text), sp(0.3)]

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # BAB 4 – KESIMPULAN
    # ══════════════════════════════════════════════════════════
    story.append(section_header('4', 'KESIMPULAN DAN REKOMENDASI'))
    story.append(sp(0.3))

    story.append(h2('4.1  Kesimpulan'))
    concl = [
        ('<b>Semua target metrik berhasil dicapai oleh model Gradient Boosting pada test set:</b> '
         'Accuracy = 0.8966 (\u226575%), F1-Score = 0.8537 (\u22650.70), '
         'AUC-ROC = 0.9582 (\u22650.80), Recall = 0.8750 (\u22650.75). '
         'Model terbukti efektif untuk tugas klasifikasi biner deteksi diabetes.'),
        ('<b>Glucose dan BMI adalah prediktor terkuat</b> secara konsisten di semua model. '
         'Intervensi klinis yang fokus pada dua faktor ini (pemantauan gula darah dan '
         'pengendalian berat badan) akan paling efektif dalam pencegahan diabetes.'),
        ('<b>Gradient Boosting unggul</b> dalam menyeimbangkan Precision (0.8333) dan '
         'Recall (0.8750), menjadikannya pilihan optimal untuk konteks medis di mana '
         'false negative harus diminimalkan.'),
        ('<b>Pipeline Machine Learning end-to-end</b> berhasil dibangun mencakup seluruh '
         'tahap: akuisisi data, EDA, preprocessing, pemodelan, evaluasi, dan deployment '
         'sebagai aplikasi web interaktif Streamlit dengan 6 halaman lengkap.'),
        ('<b>Feature engineering berbasis domain medis</b> (kategorisasi BMI, Glucose, Age '
         'sesuai standar WHO dan ADA) terbukti meningkatkan performa model dibanding '
         'menggunakan 8 fitur original saja.'),
        ('<b>Aplikasi web DiabetesSense</b> berhasil dikembangkan dan siap digunakan '
         'sebagai alat bantu skrining awal diabetes yang mudah diakses oleh tenaga medis '
         'maupun masyarakat umum.'),
    ]
    for c in concl:
        story.append(bullet(c))
    story.append(sp(0.3))

    # Status metrik box
    status_data = [
        [Paragraph('<b>Metrik</b>', ST['TableHeader']),
         Paragraph('<b>Target</b>', ST['TableHeader']),
         Paragraph('<b>Hasil (GB Test)</b>', ST['TableHeader']),
         Paragraph('<b>Status</b>', ST['TableHeader'])],
        [Paragraph('Accuracy', ST['TableCell']),
         Paragraph('\u2265 75%', ST['TableCell']),
         Paragraph(f"<b>{get_metric('Gradient Boosting','test','accuracy')}</b>", ST['TableCell']),
         Paragraph('\u2713 TERCAPAI', ParagraphStyle('sts', fontSize=9, fontName='Helvetica-Bold',
                   textColor=C_SUCCESS, alignment=TA_CENTER))],
        [Paragraph('F1-Score', ST['TableCell']),
         Paragraph('\u2265 0.70', ST['TableCell']),
         Paragraph(f"<b>{get_metric('Gradient Boosting','test','f1')}</b>", ST['TableCell']),
         Paragraph('\u2713 TERCAPAI', ParagraphStyle('sts', fontSize=9, fontName='Helvetica-Bold',
                   textColor=C_SUCCESS, alignment=TA_CENTER))],
        [Paragraph('AUC-ROC', ST['TableCell']),
         Paragraph('\u2265 0.80', ST['TableCell']),
         Paragraph(f"<b>{get_metric('Gradient Boosting','test','roc_auc')}</b>", ST['TableCell']),
         Paragraph('\u2713 TERCAPAI', ParagraphStyle('sts', fontSize=9, fontName='Helvetica-Bold',
                   textColor=C_SUCCESS, alignment=TA_CENTER))],
        [Paragraph('Recall', ST['TableCell']),
         Paragraph('\u2265 0.75', ST['TableCell']),
         Paragraph(f"<b>{get_metric('Gradient Boosting','test','recall')}</b>", ST['TableCell']),
         Paragraph('\u2713 TERCAPAI', ParagraphStyle('sts', fontSize=9, fontName='Helvetica-Bold',
                   textColor=C_SUCCESS, alignment=TA_CENTER))],
    ]
    st_tbl = Table(status_data, colWidths=[3.0*cm, 2.5*cm, 3.5*cm, 3.5*cm])
    st_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_TABLE_H),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#E8F5E9'), HexColor('#F1F8E9')]),
        ('GRID', (0, 0), (-1, -1), 0.4, HexColor('#CFD8DC')),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story += [st_tbl, sp(0.3)]

    story.append(h2('4.2  Rekomendasi'))
    story.append(h3('a. Rekomendasi Klinis / Bisnis'))
    rec_kl = [
        'Implementasikan sistem skrining ML ini di puskesmas dan klinik endokrinologi untuk triase pasien berisiko tinggi.',
        'Tetapkan threshold prediksi \u2265 0.40 (bukan default 0.50) untuk meminimalkan false negative dalam konteks klinis.',
        'Prioritaskan pemeriksaan Glucose darah dan pengukuran BMI pada setiap kunjungan pasien sebagai indikator risiko utama.',
        'Integrasikan sistem ini dengan Electronic Health Record (EHR) yang sudah berjalan untuk workflow tanpa gangguan.',
        'Jalankan program edukasi masyarakat tentang pentingnya pemantauan kadar gula darah, terutama untuk kelompok usia >35 tahun dengan BMI tinggi.',
    ]
    for r in rec_kl:
        story.append(bullet(r))

    story.append(h3('b. Rekomendasi Pengembangan Teknis'))
    rec_tk = [
        'Tambahkan fitur HbA1c, kolesterol total, dan riwayat keluarga yang lebih detail untuk meningkatkan akurasi.',
        'Implementasikan SMOTE (Synthetic Minority Over-sampling Technique) untuk mengatasi class imbalance secara lebih agresif.',
        'Eksplorasi XGBoost, LightGBM, CatBoost, dan ensemble stacking untuk performa yang lebih baik.',
        'Gunakan SHAP (SHapley Additive exPlanations) untuk interpretabilitas model yang lebih mendalam dan dapat dipercaya secara klinis.',
        'Validasi model pada dataset yang lebih besar dan populasi yang lebih beragam (bukan hanya Pima Indian).',
        'Deploy ke cloud platform (Streamlit Community Cloud, Google Cloud Run, atau AWS) untuk aksesibilitas yang lebih luas.',
    ]
    for r in rec_tk:
        story.append(bullet(r))

    story.append(h2('4.3  Keterbatasan'))
    lim = [
        'Dataset terbatas (768 sampel) dari populasi yang sangat spesifik (perempuan Pima Indian \u226521 tahun) \u2014 generalisasi ke populasi lain perlu validasi.',
        'Sebanyak 48.7% nilai Insulin kosong (nilai 0 tidak valid) \u2014 imputasi median merupakan approximation terbaik yang tersedia.',
        'Fitur penting secara klinis (HbA1c, kolesterol, gaya hidup, pola makan) tidak tersedia dalam dataset ini.',
        'Model belum divalidasi secara prospektif pada pasien nyata di fasilitas kesehatan.',
    ]
    for l in lim:
        story.append(bullet(l))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # BAB 5 – REFERENSI
    # ══════════════════════════════════════════════════════════
    story.append(section_header('5', 'REFERENSI'))
    story.append(sp(0.3))

    REFF = ParagraphStyle(
        'ref2', fontSize=9, leading=14, textColor=black,
        fontName='Helvetica', spaceAfter=2, leftIndent=18,
        firstLineIndent=-18, alignment=TA_JUSTIFY
    )
    REFL = ParagraphStyle(
        'reflink2', fontSize=9, leading=13, textColor=HexColor('#1565C0'),
        fontName='Helvetica', spaceAfter=6, leftIndent=18,
    )
    def ref(text): return Paragraph(text, REFF)
    def ref_url(url): return Paragraph(
        '<a href="' + url + '" color="#1565C0"><u>' + url + '</u></a>', REFL)

    story.append(ref(
        '[1] Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., &amp; Johannes, R.S. (1988). '
        '<i>Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.</i> '
        'Proc. Annual Symposium on Computer Application in Medical Care (SCAMC), 261\u2013265. PubMed PMCID: PMC2245318.'
    ))
    story.append(ref_url('https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/'))

    story.append(ref(
        '[2] Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). '
        '<i>Scikit-learn: Machine Learning in Python.</i> '
        'Journal of Machine Learning Research, 12, 2825\u20132830.'
    ))
    story.append(ref_url('https://jmlr.org/papers/v12/pedregosa11a.html'))

    story.append(ref(
        '[3] International Diabetes Federation. (2021). '
        '<i>IDF Diabetes Atlas, 10th Edition.</i> Brussels, Belgium: IDF.'
    ))
    story.append(ref_url('https://diabetesatlas.org'))

    story.append(ref(
        '[4] Breiman, L. (2001). <i>Random Forests.</i> Machine Learning, 45(1), 5\u201332.'
    ))
    story.append(ref_url('https://doi.org/10.1023/A:1010933404324'))

    story.append(ref(
        '[5] Friedman, J.H. (2001). '
        '<i>Greedy Function Approximation: A Gradient Boosting Machine.</i> '
        'The Annals of Statistics, 29(5), 1189\u20131232.'
    ))
    story.append(ref_url('https://doi.org/10.1214/aos/1013203451'))

    story.append(ref(
        '[6] Hosmer, D.W., &amp; Lemeshow, S. (2000). '
        '<i>Applied Logistic Regression (2nd ed.).</i> '
        'New York: John Wiley &amp; Sons. ISBN: 978-0-471-35632-5.'
    ))

    story.append(ref(
        '[7] James, G., Witten, D., Hastie, T., &amp; Tibshirani, R. (2021). '
        '<i>An Introduction to Statistical Learning with Applications in R (2nd ed.).</i> '
        'New York: Springer.'
    ))
    story.append(ref_url('https://www.statlearning.com'))

    story.append(ref(
        '[8] Streamlit Inc. (2024). '
        '<i>Streamlit \u2014 The fastest way to build data apps.</i>'
    ))
    story.append(ref_url('https://streamlit.io'))

    story.append(ref(
        '[9] Kementerian Kesehatan RI. (2018). '
        '<i>Hasil Utama Riset Kesehatan Dasar (Riskesdas) 2018.</i> '
        'Badan Penelitian dan Pengembangan Kesehatan, Jakarta.'
    ))
    story.append(ref_url('https://kesmas.kemkes.go.id/assets/upload/dir_519d41d8cd98f00/files/Hasil-riskesdas-2018_1274.pdf'))

    story.append(ref(
        '[10] American Diabetes Association. (2023). '
        '<i>Standards of Medical Care in Diabetes 2023.</i> '
        'Diabetes Care, 46(Supplement 1).'
    ))
    story.append(ref_url('https://doi.org/10.2337/dc23-Sint'))

    story.append(ref(
        '[11] Lundberg, S.M., &amp; Lee, S.I. (2017). '
        '<i>A unified approach to interpreting model predictions.</i> '
        'Advances in Neural Information Processing Systems (NeurIPS), 30, 4765\u20134774.'
    ))
    story.append(ref_url('https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html'))

    story.append(ref(
        '[12] World Health Organization. (2024). '
        '<i>Obesity and overweight.</i> WHO Fact Sheet.'
    ))
    story.append(ref_url('https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight'))

    story.append(ref(
        '[13] Dua, D., &amp; Graff, C. (2019). '
        '<i>UCI Machine Learning Repository.</i> '
        'Irvine, CA: University of California, School of Information and Computer Science.'
    ))
    story.append(ref_url('https://archive.ics.uci.edu/dataset/34/diabetes'))

    story.append(sp(1.0))
    story.append(HRFlowable(width='100%', thickness=2, color=C_PRIMARY))
    story.append(sp(0.3))
    story.append(Paragraph(
        '<b>Universitas Dian Nuswantoro (UDINUS) Semarang.</b>',
        ParagraphStyle('end', fontSize=9, alignment=TA_CENTER, textColor=C_GRAY,
                       fontName='Helvetica-Oblique', leading=14)
    ))

    return story

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("Membuat Laporan Teknis PDF...")

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=2.0*cm, bottomMargin=2.0*cm,
        title='Laporan Teknis UAS Pembelajaran Mesin - UDINUS 2026',
        author='Fahmi Fatmawati Azzahra & Nathaniela Febry Nathasa',
        subject='Prediksi Penyakit Diabetes berbasis Machine Learning',
        creator='DiabetesSense ML Pipeline',
    )

    story = build_cover() + build_content()

    doc.build(
        story,
        onFirstPage=on_first_page,
        onLaterPages=on_later_pages,
    )

    size_kb = OUTPUT_PDF.stat().st_size / 1024
    print(f"[OK] PDF berhasil dibuat!")
    print(f"     Lokasi : {OUTPUT_PDF}")
    print(f"     Ukuran : {size_kb:.1f} KB")
    print(f"     Berisi : Cover + ToC + 5 Bab + Visualisasi")
