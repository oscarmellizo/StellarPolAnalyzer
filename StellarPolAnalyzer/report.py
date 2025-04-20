# File: StellarPolAnalyzer/report.py

import os
import glob
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors


def generate_pdf_report(report_dir, output_pdf, polar_results, enriched_results):
    """
    Escanea `report_dir` en busca de las imágenes generadas y arma un PDF que
    documenta todo el pipeline:

      1. Imágenes originales
      2. Imágenes alineadas
      3. Imágenes con pares identificados
      4. Fotometría
         - Imágenes de aperturas
         - Histograma de SNR
      5. Astrometría
         - Imagen sintética
         - Tabla de RA, DEC y SIMBAD
      6. Polarimetría
         - Tabla de (q, u, P, θ, error)
         - Mapa de polarización (vectores RA–DEC)
         - Histograma de P y θ
         - Diagrama Q–U

    Parámetros
    ----------
    report_dir : str
        Carpeta donde están los PNG generados por `save_plots=True`.
    output_pdf : str
        Ruta de salida del PDF (p.ej. "report.pdf").
    polar_results : list of dict
        Salida de `compute_polarimetry_for_pairs` con claves:
          'pair_index','q','u','P','theta','error'
    enriched_results : list of dict
        Salida de `annotate_with_astrometry_net` con claves:
          'pair_index','ra','dec','simbad_id'
    """
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    normal = styles["BodyText"]

    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    story = []

    def _add_section(title, images, style=h1, width=500, height=500):
        story.append(Paragraph(title, style))
        story.append(Spacer(1, 6))
        for img in images:
            story.append(Image(img, width=width, height=height))
            story.append(Spacer(1, 12))
        story.append(PageBreak())

    # 1. Imágenes originales
    patterns = ["*_ref_img.png", "*_orig.png"]
    originals = []
    for patt in patterns:
        originals.extend(glob.glob(os.path.join(report_dir, patt)))
    originals = sorted(originals)
    _add_section("1️⃣ Imágenes originales", originals)

    # 2. Imágenes alineadas
    aligned = sorted(glob.glob(os.path.join(report_dir, "*_aligned.png")))
    _add_section("2️⃣ Imágenes alineadas", aligned)

    # 3. Imágenes con pares identificados
    pairs = sorted(glob.glob(os.path.join(report_dir, "*_pairs.png")))
    _add_section("3️⃣ Imágenes con pares identificados", pairs, h1, 600, 500)

    # 4. Fotometría
    apertures = sorted(glob.glob(os.path.join(report_dir, "*_apertures.png")))
    _add_section("4️⃣ Fotometría — Aperturas", apertures)
    snr_hist = glob.glob(os.path.join(report_dir, "*snr_hist.png"))
    _add_section("4️⃣ Fotometría — Histograma de SNR", snr_hist)

    # 5. Astrometría
    syn_img = glob.glob(os.path.join(report_dir, "*_syn.png"))  # synthetic image
    _add_section("5️⃣ Astrometría — Imagen sintética", syn_img)

    # Tabla de astrometría
    story.append(Paragraph("5️⃣ Astrometría — Resultados SIMBAD", h2))
    data = [["Par", "RA (°)", "DEC (°)", "Simbad ID"]]
    for e in enriched_results:
        data.append([
            e["pair_index"],
            f"{e['ra']:.6f}",
            f"{e['dec']:.6f}",
            e["simbad_id"]
        ])
    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    story.append(tbl)
    story.append(PageBreak())

    # 6. Polarimetría
    # 6.1 tabla de resultados
    story.append(Paragraph("6️⃣ Polarimetría — Tabla de parámetros", h2))
    data2 = [["Par", "q (%)", "u (%)", "P (%)", "θ (°)", "Error (%)"]]
    for e in polar_results:
        data2.append([
            e["pair_index"],
            f"{e['q']:.2f}",
            f"{e['u']:.2f}",
            f"{e['P']:.2f}",
            f"{e['theta']:.2f}",
            f"{e['error']:.2f}"
        ])
    tbl2 = Table(data2, hAlign="LEFT")
    tbl2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    story.append(tbl2)
    story.append(Spacer(1,12))

    # 6.2 Mapa de polarización
    vec_map = sorted(glob.glob(os.path.join(report_dir, "*vector_map.png")))
    _add_section("6️⃣ Polarimetría — Mapa de polarización", vec_map)

    # 6.3 Histograma P y θ
    hist_p = glob.glob(os.path.join(report_dir, "*hist_P.png"))
    _add_section("6️⃣ Polarimetría — Histograma de P", hist_p)
    hist_th = glob.glob(os.path.join(report_dir, "*hist_theta.png"))
    _add_section("6️⃣ Polarimetría — Histograma de θ", hist_th)

    # 6.4 Diagrama Q–U
    qu = glob.glob(os.path.join(report_dir, "*qu_diagram.png"))
    _add_section("6️⃣ Polarimetría — Diagrama Q–U", qu)

    # Build PDF
    doc.build(story)
