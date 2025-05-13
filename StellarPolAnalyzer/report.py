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


def generate_pdf_report(report_dir, output_pdf, polar_results, enriched_results):
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
    _add_section("4. Fotometría — Histograma de SNR", snr_hist)

    # 5. Astrometry: synthetic image + SIMBAD table
    syn_img = sorted(glob.glob(os.path.join(report_dir, "*_syn.png")))
    _add_section("5. Astrometría — Imagen sintética", syn_img)

    story.append(Paragraph("5. Astrometría — Resultados SIMBAD", h2))
    astro_data = [["Par", "RA (°)", "DEC (°)", "Simbad ID", "Object Type"]]
    for entry in enriched_results:
        astro_data.append([
            entry["pair_index"],
            f"{entry['ra']:.6f}",
            f"{entry['dec']:.6f}",
            entry["simbad_id"],
            entry["object_type"]
        ])
    astro_table = Table(astro_data, hAlign="LEFT")
    astro_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(astro_table)
    story.append(PageBreak())

    # 6. Polarimetry: parameters table, map, histograms, Q–U diagram
    story.append(Paragraph("6. Polarimetría — Tabla de parámetros", h2))
    polar_data = [["Par", "q (%)", "u (%)", "P (%)", "θ (°)", "Error (%)"]]
    for entry in polar_results:
        polar_data.append([
            entry["pair_index"],
            f"{entry['q']:.2f}",
            f"{entry['u']:.2f}",
            f"{entry['P']:.2f}",
            f"{entry['theta']:.2f}",
            f"{entry['error']:.2f}"
        ])
    polar_table = Table(polar_data, hAlign="LEFT")
    polar_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(polar_table)
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

    # Build and write the PDF
    doc.build(story)
