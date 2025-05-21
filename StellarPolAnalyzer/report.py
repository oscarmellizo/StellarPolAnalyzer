"""
report.py

PDF Report Generator for StellarPolAnalyzer

This module defines a single entry point, `generate_pdf_report`, which
scans a directory of diagnostic PNGs produced during the polarimetric
pipeline and assembles them—along with tables of astrometric and
polarimetric results—into a structured PDF report.

Sections included:
  1. Original images
  2. Aligned images
  3. Paired-star visualizations
  4. Photometry: Aperture overlays & SNR histogram
  5. Astrometry: Synthetic field & SIMBAD table
  6. Polarimetry:
     • Table of q, u, P, θ, error
     • Polarization map
     • Histograms of P and θ
     • Q–U diagram

Dependencies:
  - reportlab (`platypus`, `lib.styles`, `lib.pagesizes`, `lib.colors`)
  - glob, os

Example
-------
>>> from StellarPolAnalyzer.report import generate_pdf_report
>>> generate_pdf_report(
...     report_dir="reports/assets",
...     output_pdf="reports/Polarimetric_Report.pdf",
...     polar_results=polar_results_list,
...     enriched_results=enriched_results_list
... )
"""

import os
import glob
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors


def generate_pdf_report(report_dir, output_pdf, polar_results, ism_params, enriched_results):
    """
    Assemble a PDF report from pipeline diagnostics and measurement tables.

    This function will:
      1. Gather PNG files in `report_dir` matching known patterns.
      2. Insert each group of images under a numbered heading.
      3. Add a table of SIMBAD‐matched astrometric coordinates.
      4. Add a table of polarimetric parameters (q, u, P, θ, error).
      5. Include all remaining diagnostic plots (maps, histograms, Q–U diagram).
      6. Produce `output_pdf` with letter‐sized pages.

    Parameters
    ----------
    report_dir : str
        Directory where diagnostic PNGs were saved
        (when `save_plots=True` in the pipeline).
    output_pdf : str
        File path for the generated PDF (e.g., "report.pdf").
    polar_results : list of dict
        Output of `compute_polarimetry_for_pairs` or the pipeline,
        each dict must contain keys:
          - 'pair_index' : int
          - 'q', 'u', 'P', 'theta', 'error' (floats)
    enriched_results : list of dict
        Output of `annotate_with_astrometry_net`, each dict must contain:
          - 'pair_index' : int
          - 'ra', 'dec' : floats (degrees)
          - 'simbad_id' : str

    Returns
    -------
    None
        Writes `output_pdf` to disk. Raises exceptions on I/O errors.

    Notes
    -----
    - Requires that `report_dir` contains images with suffixes:
        `_ref_img.png, _orig.png, _aligned.png, _pairs.png,
         _apertures.png, snr_hist.png, *_syn.png, *_map.png,
         *_P.png, *_theta.png, *_qu.png`
    - Tables are styled with a grey header and gridlines.
    """
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    normal = styles["BodyText"]

    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    story = []

    def _add_section(title, images, style=h1, width=500, height=500):
        """
        Helper: insert a titled section of images into the story.

        Parameters
        ----------
        title : str
            Section heading.
        images : list of str
            Filepaths to PNG images.
        style : reportlab style object
            Paragraph style for the heading.
        width : int
            Display width in PDF points.
        height : int
            Display height in PDF points.
        """
        story.append(Paragraph(title, style))
        story.append(Spacer(1, 6))
        for img_path in images:
            story.append(Image(img_path, width=width, height=height))
            story.append(Spacer(1, 12))
        story.append(PageBreak())

    # 1. Original images (_ref_img.png and _orig.png)
    patterns = ["*_ref_img.png", "*_orig.png"]
    originals = []
    for patt in patterns:
        originals.extend(glob.glob(os.path.join(report_dir, patt)))
    originals = sorted(originals)
    _add_section("1. Imágenes originales", originals)

    # 2. Aligned images
    aligned = sorted(glob.glob(os.path.join(report_dir, "*_aligned.png")))
    _add_section("2. Imágenes alineadas", aligned)

    # 3. Paired-star visualizations
    pairs = sorted(glob.glob(os.path.join(report_dir, "*_pairs.png")))
    _add_section("3. Imágenes con pares identificados", pairs, h1, width=600, height=500)

    # 4. Photometry
    apertures = sorted(glob.glob(os.path.join(report_dir, "*_apertures.png")))
    _add_section("4. Fotometría — Aperturas", apertures)
    snr_hist = sorted(glob.glob(os.path.join(report_dir, "*snr_hist.png")))
    _add_section("4. Fotometría — Histograma de SNR", snr_hist, h1, width=600, height=500)

    # 5. Astrometry: synthetic image + SIMBAD table
    syn_img = sorted(glob.glob(os.path.join(report_dir, "*_syn.png")))
    _add_section("5. Astrometría — Imagen sintética", syn_img)

    story.append(Paragraph("5. Astrometría — Resultados SIMBAD", h1))
    astro_data = [["Par", "RA (°)", "DEC (°)", "Simbad ID", "Object Type"]]
    for entry in enriched_results:
        # si ra o dec es None, usamos "N/A"
        ra_str = f"{entry['ra']:.6f}" if entry.get('ra') is not None else "N/A"
        dec_str = f"{entry['dec']:.6f}" if entry.get('dec') is not None else "N/A"
        simbad = entry.get('simbad_id', "No_ID")
        object_type = entry.get('object_type', "No_Object_Type")
        astro_data.append([
            entry["pair_index"],
            ra_str,
            dec_str,
            simbad,
            object_type
        ])
    astro_table = Table(astro_data, hAlign="LEFT")
    astro_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(astro_table)
    story.append(PageBreak())

    # 6. Polarimetry: Estimate interstellar (ISM) polarization from high-SNR q,u distributions
    story.append(Paragraph("6. Polarimetría — Estimación Interestelar (ISM)", h1))

    # extraemos todos los parámetros
    q_means   = ism_params['ism_estimation']['q_means']
    q_sigmas  = ism_params['ism_estimation']['q_sigmas']
    q_weights = ism_params['ism_estimation']['q_weights']
    u_means   = ism_params['ism_estimation']['u_means']
    u_sigmas  = ism_params['ism_estimation']['u_sigmas']
    u_weights = ism_params['ism_estimation']['u_weights']
    dominant_q = ism_params['ism_estimation']['dominant_q']
    dominant_u = ism_params['ism_estimation']['dominant_u']

    # Preparamos la tabla: encabezados
    ism_data = [[
        "Comp.",
        "Q mean", "Q σ", "Q weight",
        "U mean", "U σ", "U weight",
        "Dominante Q", "Dominante U"
    ]]

    # Rellenamos filas por cada componente
    n_comp = max(len(q_means), len(u_means))
    for i in range(n_comp):
        mu_q   = q_means[i]
        sig_q  = q_sigmas[i]
        w_q    = q_weights[i]
        mu_u   = u_means[i]
        sig_u  = u_sigmas[i]
        w_u    = u_weights[i]

        # Función auxiliar
        def fmt(x, fmt_str):
            return fmt_str.format(x) if isinstance(x, (int, float)) else str(x)
        
        row = [
            str(i),
            fmt(mu_q, "{:.3f}"),
            fmt(sig_q, "{:.3f}"),
            fmt(w_q, "{:.2f}"),
            fmt(mu_u, "{:.3f}"),
            fmt(sig_u, "{:.3f}"),
            fmt(w_u, "{:.2f}"),
            "✔" if i == dominant_q else "",
            "✔" if i == dominant_u else ""
        ]
        ism_data.append(row)

    # Dibujamos la tabla
    ism_table = Table(ism_data, hAlign="LEFT")
    ism_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.black),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
    ]))
    story.append(ism_table)
    story.append(Spacer(1, 12))


    # 6.2 Polarization map
    vec_map = sorted(glob.glob(os.path.join(report_dir, "*_map.png")))
    _add_section("6. Polarimetría — Mapa de polarización", vec_map)

    # 6.3 Histograms
    hist_p = sorted(glob.glob(os.path.join(report_dir, "*_P.png")))
    _add_section("6. Polarimetría — Histograma de P", hist_p)
    hist_th = sorted(glob.glob(os.path.join(report_dir, "*_theta.png")))
    _add_section("6. Polarimetría — Histograma de θ", hist_th)

    # 6.4 Q–U diagram
    qu = sorted(glob.glob(os.path.join(report_dir, "*_qu.png")))
    _add_section("6. Polarimetría — Diagrama Q–U", qu)

    # 6. Polarimetry: Estimate interstellar (ISM) polarization from high-SNR q,u distributions
    story.append(Paragraph("6. Polarimetría — Estimación Interestelar (ISM)", h1))

    # Extraemos parámetros
    qm  = ism_params['ism_estimation']['q_means']
    qs  = ism_params['ism_estimation']['q_sigmas']
    qw  = ism_params['ism_estimation']['q_weights']
    um  = ism_params['ism_estimation']['u_means']
    us  = ism_params['ism_estimation']['u_sigmas']
    uw  = ism_params['ism_estimation']['u_weights']
    dq  = ism_params['ism_estimation']['dominant_q']
    du  = ism_params['ism_estimation']['dominant_u']

    # Preparamos la tabla
    ism_data = [[
        "Comp.",
        "Q mean", "Q σ", "Q weight",
        "U mean", "U σ", "U weight",
        "Dom Q", "Dom U"
    ]]
    # Una fila por componente
    for i in range(len(qm)):
        mu_q   = qm[i]
        sig_q  = qs[i]
        w_q    = qw[i]
        mu_u   = um[i]
        sig_u  = us[i]
        w_u    = uw[i]

        # Función auxiliar
        def fmt(x, fmt_str):
            return fmt_str.format(x) if isinstance(x, (int, float)) else str(x)
        
        row = [
        str(i),
        fmt(mu_q, "{:.3f}"),
        fmt(sig_q, "{:.3f}"),
        fmt(w_q, "{:.2f}"),
        fmt(mu_u, "{:.3f}"),
        fmt(sig_u, "{:.3f}"),
        fmt(w_u, "{:.2f}"),
        "✔" if i == dq else "",
        "✔" if i == du else ""
    ]
    ism_data.append(row)

    # Creamos y estilizamos la tabla
    ism_table = Table(ism_data, hAlign="LEFT")
    ism_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.black),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
    ]))
    story.append(ism_table)
    story.append(Spacer(1, 12))


    # Construimos el texto HTML para la leyenda
    legend_text = (
        '<b>Explicación de columnas:</b><br/>'
        '<b>Q Means:</b> medias de la componente q estimada para la polarización interestelar.<br/>'
        '<b>Q sigmas:</b> desviaciones estándar (σ) de los componentes gaussianos ajustados a q.<br/>'
        '<b>Q weights:</b> pesos relativos de cada componente gaussiano en la mezcla de q.<br/>'
        '<b>U Means:</b> medias de la componente u estimada para la polarización interestelar.<br/>'
        '<b>U sigmas:</b> desviaciones estándar (σ) de los componentes gaussianos ajustados a u.<br/>'
        '<b>U weights:</b> pesos relativos de cada componente gaussiano en la mezcla de u.'
    )

    # Lo añadimos al flujo de document
    story.append(Paragraph(legend_text, normal))
    story.append(Spacer(1, 12))

    # Histograma de ISM con ajuste GMM Q
    q = sorted(glob.glob(os.path.join(report_dir, "*q_ism_hist.png")))
    _add_section("6. Polarimetría — Histograma de ISM con ajuste GMM Q", q)
    
    # Histograma de ISM con ajuste GMM U
    u = sorted(glob.glob(os.path.join(report_dir, "*u_ism_hist.png")))
    _add_section("6. Polarimetría — Histograma de ISM con ajuste GMM U", u)

    # Build and write the PDF
    doc.build(story)
