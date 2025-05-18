import json
import math
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, legal

def write_candidate_pairs_to_file(candidate_pairs, filename="candidate_pairs.txt"):
    """
    Escribe las parejas candidatas de estrellas en un archivo de texto tabulado.

    Cada línea del archivo contendrá cuatro columnas separadas por tabuladores:
      - Estrella_A: índice de la primera estrella de la pareja.
      - Estrella_B: índice de la segunda estrella de la pareja.
      - Distancia_px: separación entre ambas estrellas en píxeles, formateada a dos decimales.
      - Ángulo_deg: ángulo de emparejamiento en grados, formateado a dos decimales.

    Parámetros
    ----------
    candidate_pairs : list of tuple
        Lista de tuplas (i, j, distance, angle), donde:
          i (int)       – índice de la primera estrella,
          j (int)       – índice de la segunda estrella,
          distance (float) – distancia en píxeles,
          angle (float)    – ángulo en grados.
    filename : str, opcional
        Ruta y nombre del archivo de salida. Por defecto "candidate_pairs.txt".

    Ejemplo
    -------
    >>> pairs = [(0, 1, 36.37, 179.76), (2, 3, 36.37, 179.76)]
    >>> write_candidate_pairs_to_file(pairs, "pares.txt")
    # Crea 'pares.txt' con encabezado y los datos formateados.
    """
    with open(filename, 'w') as f:
        f.write("Estrella_A\tEstrella_B\tDistancia_px\tÁngulo_deg\n")
        for i, j, d, a in candidate_pairs:
            f.write(f"{i}\t{j}\t{d:.2f}\t{a:.2f}\n")


def generate_star_report(json_path: str,
                         simbad_id: str,
                         filter_name: str,
                         pdf_path: str,
                         json_out_path: str = None):
    """
    Genera un PDF y un JSON con el resultado polarimétrico de una estrella.

    Parámetros
    ----------
    json_path : str
        Ruta al archivo pipeline_results.json.
    simbad_id : str
        Identificador SIMBAD de la estrella (p.ej. "1H 1936+541").
    filter_name : str
        Filtro (I, R, V, B).
    pdf_path : str
        Ruta donde se escribirá el PDF final.
    json_out_path : str, opcional
        Ruta donde se escribirá el JSON. Si no se provee, se usa el mismo nombre que `pdf_path`
        reemplazando la extensión por `.json`.
    """

    # 1) Cargo el JSON de entrada
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    # 2) Busco el registro de la estrella
    rec = next((r for r in records if r.get('simbad_id') == simbad_id), None)
    if rec is None:
        raise ValueError(f"No existe simbad_id = '{simbad_id}' en {json_path}")

    # 3) Calculo P (%) y su error (%)
    P_pct   = rec['P']       # ya viene en porcentaje (p.ej. 8.076)
    err_pct = rec['error']

    # 4) Calculo θ (°) y su error (°)
    theta_deg   = rec['theta']  # ya en grados
    # σθ = (σP)/(2·P) * (180/π)
    sigma_theta = (err_pct / (2.0 * P_pct)) * (180.0 / math.pi)

    # 5) RA/Dec (°)
    ra_deg  = rec.get('ra')
    dec_deg = rec.get('dec')

    # 6) Preparo los datos
    data = {
        'simbad_id': simbad_id,
        'filter':    filter_name.upper(),
        'P_pct':     round(P_pct, 2),
        'err_pct':   round(err_pct, 2),
        'theta_deg': round(theta_deg, 2),
        'err_theta_deg': round(sigma_theta, 2),
        'ra_deg':    round(ra_deg, 6) if ra_deg is not None else None,
        'dec_deg':   round(dec_deg, 6) if dec_deg is not None else None,
    }

    # 7) Determino ruta JSON de salida
    if json_out_path is None:
        base, _ = os.path.splitext(pdf_path)
        json_out_path = f"{base}.json"

    # 8) Validación y creación de directorios
    for path in [pdf_path, json_out_path]:
        dirpath = os.path.dirname(path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
    
    # 9) Escribo el JSON
    with open(json_out_path, 'w', encoding='utf-8') as jf:
        json.dump(data, jf, ensure_ascii=False, indent=2)

    # 10) Genero el PDF
    c = canvas.Canvas(pdf_path, pagesize=landscape(legal))
    W, H = landscape(legal)
    y = H - 50

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Polarimetric Report — {simbad_id} ({filter_name.upper()})")
    y -= 30

    # Tabla
    headers = ["simbad_id", "filtro", "P (%)", "err_P (%)", "θ (°)", "err_θ (°)", "RA (°)", "Dec (°)"]
    vals    = [
        data['simbad_id'], data['filter'],
        f"{data['P_pct']}", f"{data['err_pct']}",
        f"{data['theta_deg']}", f"{data['err_theta_deg']}",
        f"{data['ra_deg']}", f"{data['dec_deg']}"
    ]
    x_positions = [50, 140, 200, 270, 350, 420, 490, 560]

    c.setFont("Helvetica-Bold", 12)
    for x, h in zip(x_positions, headers):
        c.drawString(x, y, h)
    y -= 20

    c.setFont("Helvetica", 12)
    for x, v in zip(x_positions, vals):
        c.drawString(x, y, v)
    y -= 40

    # Descripción
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Descripción de columnas:")
    y -= 20
    c.setFont("Helvetica", 11)
    bullets = [
        ("simbad_id",      "Identificador del objeto en SIMBAD."),
        ("filtro",         "Filtro fotométrico usado (I, R, V o B)."),
        ("P (%)",          "Grado de polarización lineal en %: 100·√(q²+u²)."),
        ("err_P (%)",      "Incertidumbre de P en %, tal como la entrega el pipeline."),
        ("θ (°)",          "Ángulo de polarización: θ = ½·atan2(u,q)."),
        ("err_θ (°)",      "Error de θ: σθ = (σP)/(2P)·(180/π)."),
        ("RA (°)",         "Ascensión recta (ICRS, J2000) en grados."),
        ("Dec (°)",        "Declinación (ICRS, J2000) en grados.")
    ]
    for label, desc in bullets:
        if y < 100:
            c.showPage()
            y = H - 50
            c.setFont("Helvetica", 11)
        c.drawString(60, y, f"• {label}: {desc}")
        y -= 16

    c.save()

    print(f"PDF generado en:  {pdf_path}")
    print(f"JSON generado en: {json_out_path}")


def generate_final_report(json_dir: str,
                          pdf_path: str,
                          json_out_path: str = None):
    """
    Genera un reporte final agregando múltiples archivos JSON.

    Parámetros
    ----------
    json_dir : str
        Carpeta que contiene los archivos .json individuales.
    pdf_path : str
        Ruta de salida para el PDF resumen.
    json_out_path : str, opcional
        Ruta de salida para el JSON resumen. Si no se provee, se usa mismo nombre del PDF con extensión .json.
    """
    # 1) Agrupar registros
    data = {}
    for fname in os.listdir(json_dir):
        if not fname.lower().endswith('.json'):
            continue
        print(fname)
        full = os.path.join(json_dir, fname)
        with open(full, 'r', encoding='utf-8') as f:
            rec = json.load(f)
        sid = rec.get('simbad_id')
        filt = rec.get('filter', '').upper()
        if sid is None or filt == '':
            continue
        if sid not in data:
            data[sid] = {
                'ra_deg': rec.get('ra_deg'),
                'dec_deg': rec.get('dec_deg'),
                'filters': {}
            }
        data[sid]['filters'][filt] = rec

    # 2) Construir filas para la tabla
    rows = []
    filters = ['I', 'R', 'B', 'V']
    for sid, info in data.items():
        row = {'Object': sid}
        for F in filters:
            rec = info['filters'].get(F)
            if rec:
                P = f"{rec['P_pct']:.2f} ± {rec['err_pct']:.2f}"
                A = f"{rec['theta_deg']:.2f} ± {rec['err_theta_deg']:.2f}"
            else:
                P = ''
                A = ''
            row[f"{F} (%Pol {F})"]   = P
            row[f"{F} (Angle {F})"]   = A
        row['RA']  = info.get('ra_deg')
        row['DEC'] = info.get('dec_deg')
        rows.append(row)

    # 3) Determinar JSON de salida
    if json_out_path is None:
        base, _ = os.path.splitext(pdf_path)
        json_out_path = base + '.json'

    # 4) Validar y crear directorios
    for path in [pdf_path, json_out_path]:
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    # 5) Escribir JSON resumen
    with open(json_out_path, 'w', encoding='utf-8') as jf:
        json.dump(rows, jf, ensure_ascii=False, indent=2)

    # 6) Generar PDF resumen
    c = canvas.Canvas(pdf_path, pagesize=landscape(legal))
    W, H = landscape(legal)
    y = H - 50
    # Título
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Final Polarimetric Summary")
    y -= 30

    # Encabezado tabla
    headers = [
        'Object',
        'I (%Pol I)', 'R (%Pol R)', 'B (%Pol B)', 'V (%Pol V)',
        'I (Angle I)', 'R (Angle R)', 'B (Angle B)', 'V (Angle V)',
        'RA', 'DEC'
    ]
    x_pos = [
      50,   # Object
      200,  # I (%Pol I)
      280,  # R (%Pol R)
      360,  # B (%Pol B)
      440,  # V (%Pol V)
      520,  # I (Angle I)
      600,  # R (Angle R)
      680,  # B (Angle B)
      760,  # V (Angle V)
      840,  # RA
      920   # DEC
    ]
    c.setFont("Helvetica-Bold", 10)
    for x, h in zip(x_pos, headers):
        c.drawString(x, y, h)
    y -= 15

    # Filas
    c.setFont("Helvetica", 9)
    for row in rows:
        for x, h in zip(x_pos, headers):
            v = row.get(h, '')
            c.drawString(x, y, str(v))
        y -= 12
        if y < 100:
            c.showPage()
            y = H - 50
            c.setFont("Helvetica-Bold", 10)
            for x, h in zip(x_pos, headers):
                c.drawString(x, y, h)
            y -= 15
            c.setFont("Helvetica", 9)

    # Descripción de columnas
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Column Descriptions:")
    y -= 20
    c.setFont("Helvetica", 10)
    descs = [
        ("Object",          "Identificador SIMBAD de la fuente."),
        ("X (%Pol X)",      "Polarización P ± error en %% en banda X."),
        ("X (Angle X)",     "Ángulo θ ± error en grados en banda X."),
        ("RA",              "Ascensión recta ICRS J2000 en grados."),
        ("DEC",             "Declinación ICRS J2000 en grados.")
    ]
    # repetir descripción para cada banda X
    for F in filters:
        descs.insert(1, (f"{F} (%Pol {F})", f"P ± error en banda {F}"))
        descs.insert(2, (f"{F} (Angle {F})", f"θ ± error en grados en banda {F}"))

    for label, text in descs:
        if y < 50:
            c.showPage()
            y = H - 50
            c.setFont("Helvetica", 10)
        c.drawString(60, y, f"• {label}: {text}")
        y -= 14

    c.save()
    print(f"PDF generado: {pdf_path}")
    print(f"JSON generado: {json_out_path}")

def run_star_report(json_path: str, simbad_id: str, filter_name: str, output_dir: str):
    """
    Genera PDF y JSON de reporte para una estrella específica.

    Parámetros
    ----------
    json_path : str
        Ruta al archivo JSON del pipeline.
    simbad_id : str
        Identificador SIMBAD de la estrella.
    filter_name : str
        Filtro (I, R, V, B).
    output_dir : str
        Carpeta donde se guardarán los PDF y JSON de salida.
    """
    # Asegurar que el directorio de salida existe
    os.makedirs(output_dir, exist_ok=True)

    # Limpiar simbad_id para usar en nombre de archivo: reemplazar '*' por '_'
    safe_id = simbad_id.replace('*', '_').replace(' ', '').replace('/', '_')
    pdf_path = os.path.join(output_dir, f"{safe_id}_{filter_name}.pdf")
    json_out_path = os.path.join(output_dir, f"{safe_id}_{filter_name}.json")

    # Generar reporte
    generate_star_report(
        json_path=json_path,
        simbad_id=simbad_id,
        filter_name=filter_name,
        pdf_path=pdf_path,
        json_out_path=json_out_path
    )

    print(f"Reporte generado: PDF → {pdf_path} | JSON → {json_out_path}")