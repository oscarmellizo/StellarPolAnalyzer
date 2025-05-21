import json
import math
import os
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, legal
from reportlab.lib.units import mm
from sklearn.mixture import GaussianMixture

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
    Genera un PDF y un JSON con el resultado polarimétrico de una estrella,
    incluyendo la estimación ISM de q y u, con tamaño de página adaptado
    al número de columnas.
    """
    # 1) Cargo el JSON de entrada
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    # 2) Busco el registro de la estrella
    rec = next((r for r in records if r.get('simbad_id') == simbad_id), None)
    if rec is None:
        raise ValueError(f"No existe simbad_id = '{simbad_id}' en {json_path}")

    # 3) Calculo P (%) y su error (%)
    P_pct   = rec['P']
    err_pct = rec['error']

    # 4) Calculo θ (°) y su error (°)
    theta_deg   = rec['theta']
    sigma_theta = (err_pct / (2.0 * P_pct)) * (180.0 / math.pi)

    # 5) RA/Dec (°)
    ra_deg  = rec.get('ra')
    dec_deg = rec.get('dec')

    # 6) Preparo los datos
    data = {
        'simbad_id':     simbad_id,
        'filter':        filter_name.upper(),
        'P_pct':         round(P_pct, 2),
        'err_pct':       round(err_pct, 2),
        'theta_deg':     round(theta_deg, 2),
        'err_theta_deg': round(sigma_theta, 2),
        'ra_deg':        round(ra_deg, 6) if ra_deg is not None else None,
        'dec_deg':       round(dec_deg, 6) if dec_deg is not None else None,
    }

    # 7) Cargo estimación ISM
    ism_file = os.path.join(os.path.dirname(json_path), 'ism_estimation.json')
    if os.path.exists(ism_file):
        ism = json.load(open(ism_file, 'r')).get('ism_estimation', {})
        qm = ism['q_means']; qs = ism['q_sigmas']; dq = ism['dominant_q']
        um = ism['u_means']; us = ism['u_sigmas']; du = ism['dominant_u']
        data.update({
            'Q_ISM':     round(qm[dq], 3),
            'err_Q_ISM': round(qs[dq], 3),
            'ISM_dom_q': dq,
            'U_ISM':     round(um[du], 3),
            'err_U_ISM': round(us[du], 3),
            'ISM_dom_u': du
        })
    else:
        for key in ('Q_ISM','err_Q_ISM','ISM_dom_q','U_ISM','err_U_ISM','ISM_dom_u'):
            data[key] = None

    # 8) JSON de salida
    if json_out_path is None:
        base, _ = os.path.splitext(pdf_path)
        json_out_path = f"{base}.json"
    os.makedirs(os.path.dirname(json_out_path), exist_ok=True)
    with open(json_out_path, 'w', encoding='utf-8') as jf:
        json.dump(data, jf, ensure_ascii=False, indent=2)

    # 9) Preparar PDF con ancho dinámico
    headers = [
        "simbad_id","Filtro",
        "P (%)","err_P (%)",
        "θ (°)","err_θ (°)",
        "Q_ISM","err_Q_ISM","ISM_dom_q",
        "U_ISM","err_U_ISM","ISM_dom_u",
        "RA","Dec"
    ]
    vals = [
        data['simbad_id'], data['filter'],
        f"{data['P_pct']}", f"{data['err_pct']}",
        f"{data['theta_deg']}", f"{data['err_theta_deg']}",
        f"{data['Q_ISM']}"     if data['Q_ISM']     is not None else "",
        f"{data['err_Q_ISM']}" if data['err_Q_ISM'] is not None else "",
        f"{data['ISM_dom_q']}" if data['ISM_dom_q'] is not None else "",
        f"{data['U_ISM']}"     if data['U_ISM']     is not None else "",
        f"{data['err_U_ISM']}" if data['err_U_ISM'] is not None else "",
        f"{data['ISM_dom_u']}" if data['ISM_dom_u'] is not None else "",
        f"{data['ra_deg']}", f"{data['dec_deg']}"
    ]

    # Parámetros de layout
    margin    = 20 * mm
    n_cols    = len(headers)
    col_width = 40 * mm  # ajusta este valor si necesitas más o menos ancho
    page_w    = margin*2 + col_width * n_cols
    page_h    = 80 * mm  # altura suficiente
    c = canvas.Canvas(pdf_path, pagesize=(page_w, page_h))
    W, H = page_w, page_h
    y = H - 20 * mm

    # Título
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, f"Polarimetric Report — {simbad_id} ({data['filter']})")
    y -= 12 * mm

    # Calcular posiciones X
    x_positions = [margin + i*col_width for i in range(n_cols)]

    # Dibujar cabeceras
    c.setFont("Helvetica-Bold", 10)
    for x, h in zip(x_positions, headers):
        c.drawString(x, y, h)
    y -= 8 * mm

    # Dibujar valores
    c.setFont("Helvetica", 10)
    for x, v in zip(x_positions, vals):
        c.drawString(x, y, v)
    y -= 20 * mm

    # Descripción de columnas
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Descripción de columnas:")
    y -= 6 * mm
    c.setFont("Helvetica", 9)
    bullets = [
        ("simbad_id",    "Identificador del objeto en SIMBAD."),
        ("Filtro",       "Filtro fotométrico (I, R, V o B)."),
        ("P (%)",        "Grado de polarización lineal en %."),
        ("err_P (%)",    "Error de P en %."),
        ("θ (°)",        "Ángulo de polarización en grados."),
        ("err_θ (°)",    "Error de θ en grados."),
        ("Q_ISM",        "Media de q del ISM."),
        ("err_Q_ISM",    "σ de q del ISM."),
        ("ISM_dom_q",    "Índice de componente dominante q (0 o 1)."),
        ("U_ISM",        "Media de u del ISM."),
        ("err_U_ISM",    "σ de u del ISM."),
        ("ISM_dom_u",    "Índice de componente dominante u (0 o 1)."),
        ("RA",           "Ascensión recta (°)."),
        ("Dec",          "Declinación (°).")
    ]
    for label, desc in bullets:
        if y < 10 * mm:
            c.showPage()
            y = H - 20 * mm
            c.setFont("Helvetica", 9)
        c.drawString(margin + 5*mm, y, f"• {label}: {desc}")
        y -= 5 * mm

    c.save()
    print(f"PDF generado en:  {pdf_path}")
    print(f"JSON generado en: {json_out_path}")

def generate_final_report(json_dir: str,
                          pdf_path: str,
                          json_out_path: str = None):
    """
    Genera un reporte final con ancho de página dinámico
    para alojar todas las columnas, incluyendo las componentes
    dominantes de ISM para q y u en cada filtro.
    """
    # 1) Agrupar registros…
    data = {}
    for fname in os.listdir(json_dir):
        if not fname.lower().endswith('.json'):
            continue
        full = os.path.join(json_dir, fname)
        rec = json.load(open(full, 'r', encoding='utf-8'))
        sid  = rec.get('simbad_id')
        filt = rec.get('filter',    '').upper()
        if not sid or not filt:
            continue
        data.setdefault(sid, {
            'ra_deg':  rec.get('ra_deg'),
            'dec_deg': rec.get('dec_deg'),
            'filters': {}
        })['filters'][filt] = rec

    # 2) Definir filtros y encabezados
    filters = ['I','R','B','V']
    headers = ['Object']
    for F in filters:
        headers += [
            f"{F} (%Pol {F})",
            f"{F} (Angle {F})",
            f"{F} Q_ISM",
            f"{F} U_ISM",
            f"{F} dom_q",
            f"{F} dom_u"
        ]
    headers += ['RA','DEC']

    # 3) Construir filas
    rows = []
    for sid, info in data.items():
        row = {'Object': sid}
        for F in filters:
            rec = info['filters'].get(F, {})
            if rec:
                row[f"{F} (%Pol {F})"]  = f"{rec['P_pct']:.2f} ± {rec['err_pct']:.2f}"
                row[f"{F} (Angle {F})"] = f"{rec['theta_deg']:.2f} ± {rec['err_theta_deg']:.2f}"
                row[f"{F} Q_ISM"]       = f"{rec.get('Q_ISM',0):.3f} ± {rec.get('err_Q_ISM',0):.3f}"
                row[f"{F} U_ISM"]       = f"{rec.get('U_ISM',0):.3f} ± {rec.get('err_U_ISM',0):.3f}"
                row[f"{F} dom_q"]       = str(rec.get('ISM_dom_q', ''))
                row[f"{F} dom_u"]       = str(rec.get('ISM_dom_u', ''))
            else:
                # si no hay rec, dejo todas las columnas vacías
                for col in [
                    f"{F} (%Pol {F})",
                    f"{F} (Angle {F})",
                    f"{F} Q_ISM",
                    f"{F} U_ISM",
                    f"{F} dom_q",
                    f"{F} dom_u"
                ]:
                    row[col] = ''
        row['RA']  = info.get('ra_deg')
        row['DEC'] = info.get('dec_deg')
        rows.append(row)

    # 4) JSON de salida
    if json_out_path is None:
        base, _ = os.path.splitext(pdf_path)
        json_out_path = f"{base}.json"
    os.makedirs(os.path.dirname(json_out_path), exist_ok=True)
    with open(json_out_path, 'w', encoding='utf-8') as jf:
        json.dump(rows, jf, ensure_ascii=False, indent=2)

    # 5) Configurar canvas con ancho dinámico
    margin = 20 * mm
    n_cols = len(headers)
    col_width = 40 * mm  # ajusta este ancho si necesitas más o menos espacio
    page_width = margin*2 + col_width * n_cols
    page_height = 180 * mm
    c = canvas.Canvas(pdf_path, pagesize=(page_width, page_height))
    W, H = page_width, page_height
    y = H - 20 * mm

    # 6) Título
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Final Polarimetric Summary")
    y -= 10 * mm

    # 7) Calcular posiciones X
    x_positions = [margin + i*col_width for i in range(n_cols)]

    # 8) Dibujar encabezados
    c.setFont("Helvetica-Bold", 10)
    for x, h in zip(x_positions, headers):
        c.drawString(x, y, h)
    y -= 8 * mm

    # 9) Dibujar filas de datos
    c.setFont("Helvetica", 9)
    for row in rows:
        for x, h in zip(x_positions, headers):
            v = row.get(h, '')
            c.drawString(x, y, str(v))
        y -= 6 * mm
        if y < 30 * mm:
            c.showPage()
            y = H - 20 * mm
            c.setFont("Helvetica-Bold", 10)
            for x, h in zip(x_positions, headers):
                c.drawString(x, y, h)
            y -= 8 * mm
            c.setFont("Helvetica", 9)

    # 10) Descripción de columnas
    y -= 10 * mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Column Descriptions:")
    y -= 8 * mm
    c.setFont("Helvetica", 10)
    descs = [("Object","Identificador SIMBAD.")]
    for F in filters:
        descs += [
            (f"{F} (%Pol {F})",  "Polarización P ± error en %"),
            (f"{F} (Angle {F})", "Ángulo θ ± error en °"),
            (f"{F} Q_ISM",       "q_ISM ± σ"),
            (f"{F} U_ISM",       "u_ISM ± σ"),
            (f"{F} dom_q",       "Índice de componente dominante de q (0 o 1)"),
            (f"{F} dom_u",       "Índice de componente dominante de u (0 o 1)")
        ]
    descs += [("RA","Ascensión recta (°)"),("DEC","Declinación (°)")]
    for label, text in descs:
        if y < 20 * mm:
            c.showPage()
            y = H - 20 * mm
            c.setFont("Helvetica", 10)
        c.drawString(margin + 5*mm, y, f"• {label}: {text}")
        y -= 5 * mm

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
    
def estimate_ism_from_histograms(q_vals, u_vals,
                                 n_components=2,
                                 random_state=0):
    """
    Ajusta GaussianMixture a las distribuciones de q y u.
    Siempre devuelve listas de longitud n_components:
      - Si hay <2 muestras, la segunda componente es '-' (placeholder).
    Retorna dict con:
      q_means, q_sigmas, q_weights,
      u_means, u_sigmas, u_weights,
      dominant_q, dominant_u
    """
    def _safe_fit(data):
        # Cálculo básico de media y sigma
        mean  = float(np.mean(data)) if len(data) > 0 else 0.0
        sigma = float(np.std(data, ddof=0)) if len(data) > 0 else 0.0

        # Si no hay al menos 2 muestras, devolvemos placeholder en segunda posición
        if len(data) < 2:
            means   = [mean, "-"]
            sigmas  = [sigma, "-"]
            weights = [1.0, "-"]
            return means, sigmas, weights

        # Ajuste GMM normal
        X = np.array(data).reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_components,
                              random_state=random_state)
        gmm.fit(X)
        means   = gmm.means_.flatten().tolist()
        sigmas  = np.sqrt(gmm.covariances_).flatten().tolist()
        weights = gmm.weights_.flatten().tolist()
        return means, sigmas, weights

    # Ajuste seguro para q y u
    q_means, q_sigmas, q_weights = _safe_fit(q_vals)
    u_means, u_sigmas, u_weights = _safe_fit(u_vals)

    # Índice de componente dominante (ponderado con '1.0' vs '-' → argmax queda en 0)
    dominant_q = int(np.nanargmax([w if isinstance(w, float) else 0 for w in q_weights]))
    dominant_u = int(np.nanargmax([w if isinstance(w, float) else 0 for w in u_weights]))
    
    inner = {
        'q_means':   q_means,
        'q_sigmas':  q_sigmas,
        'q_weights': q_weights,
        'u_means':   u_means,
        'u_sigmas':  u_sigmas,
        'u_weights': u_weights,
        'dominant_q': dominant_q,
        'dominant_u': dominant_u
    }
    # Devolver un solo elemento llamado 'ism_estimation'
    return {'ism_estimation': inner}