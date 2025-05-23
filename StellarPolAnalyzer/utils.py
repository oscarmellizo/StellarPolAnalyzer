import json
import math
import os
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, legal
from reportlab.lib.units import mm
from sklearn.mixture import GaussianMixture
from decimal import Decimal, InvalidOperation

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
        'P_pct':         P_pct,
        'err_pct':       err_pct,
        'theta_deg':     theta_deg,
        'err_theta_deg': sigma_theta,
        'ra_deg':        ra_deg if ra_deg is not None else None,
        'dec_deg':       dec_deg if dec_deg is not None else None,
    }

    # 7) Cargo estimación ISM
    ism_file = os.path.join(os.path.dirname(json_path), 'ism_estimation.json')
    if os.path.exists(ism_file):
        ism = json.load(open(ism_file, 'r')).get('ism_estimation', {})
        qm = ism['q_means']; qs = ism['q_sigmas']; dq = ism['dominant_q']
        um = ism['u_means']; us = ism['u_sigmas']; du = ism['dominant_u']
        q = qm[dq]
        q_err = qs[dq]
        u = um[du]
        u_err = us[du]
         # calculamos P_ISM y θ_ISM
        P_ISM, err_P_ISM, θ_ISM, err_θ_ISM = compute_ism_pol_angle(q, u, q_err, u_err)
        data.update({
            'p_ism':     P_ISM,
            'err_p_ism': err_P_ISM,
            'θ_ism':     θ_ISM,
            'err_θ_ism': err_θ_ISM
        })
    else:
        for key in ('p_ism','err_p_ism','θ_ism','err_θ_ism'):
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
        "Simbad id","Filtro",
        "Polarization P (%)",
        "Angle θ (°)",
        "P ISM (%)",
        "θ ISM (°)",
        "RA","Dec"
    ]
    
    vals = [
        data['simbad_id'], data['filter'],
        fme([(data['P_pct'], data['err_pct'])]),
        fme([(data['theta_deg'], data['err_theta_deg'])]),
        fme([(data['p_ism'], data['err_p_ism'])]) if data['p_ism'] is not None and data['err_p_ism'] is not None else "",
        fme([(data['θ_ism'], data['err_θ_ism'])]) if data['θ_ism'] is not None and data['err_θ_ism'] is not None else "",
        f"{data['ra_deg']}", f"{data['dec_deg']}"
    ]

    # Parámetros de layout
    margin    = 20 * mm
    n_cols    = len(headers)
    col_width = 36 * mm  # ajusta este valor si necesitas más o menos ancho
    page_w    = margin*2 + col_width * n_cols
    page_h    = 140 * mm  # altura suficiente
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
        ("Simbad id",    "Identificador del objeto en SIMBAD."),
        ("Filtro",       "Filtro fotométrico (I, R, V o B)."),
        ("Polarization P (%)","Grado de polarización lineal en % ± error en %"),
        ("Angle θ (°)", "Ángulo de polarización en grados ± error en °"),
        ("P ISM (%)",   "Polarización del medio interestelar en %"),
        ("θ ISM (°)",   "Ángulo del medio interestelar en grados"),
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
            f"Polarization {F}",
            f"Angle {F}",
            f"P ISM - {F}",
            f"θ ISM - {F}",
        ]
    headers += ['RA','DEC']

    # 3) Construir filas
    rows = []
    for sid, info in data.items():
        row = {'object': sid}
        for F in filters:
            rec = info['filters'].get(F, {})
            if rec:
                row[f"pol_{F}"]  = rec['P_pct']
                row[f"err_pol_{F}"]  = rec['err_pct']
                row[f"angle_{F}"] = rec['theta_deg']
                row[f"err_angle_{F}"] = rec['err_theta_deg']
                row[f"P_ism_{F}"] = rec.get('p_ism',0)
                row[f"err_P_ism_{F}"] = rec.get('err_p_ism',0)
                row[f"θ_ism_{F}"] = rec.get('θ_ism',0)
                row[f"err_θ_ism_{F}"] = rec.get('err_θ_ism',0)
            else:
                # si no hay rec, dejo todas las columnas vacías
                for col in [
                    f"pol_{F}",
                    f"err_pol_{F}",
                    f"angle_{F}",
                    f"err_angle_{F}"
                    f"P_ism_{F}",
                    f"err_P_ism_{F}",
                    f"θ_ism_{F}",
                    f"err_θ_ism_{F}"
                ]:
                    row[col] = ''
        row['RA']  = info.get('ra_deg')
        row['DEC'] = info.get('dec_deg')
        rows.append(row)
    
    # 3) Construir filas
    rows_pdf = []
    for sid, info in data.items():
        row = {'Object': sid}
        for F in filters:
            rec = info['filters'].get(F, {})
            if rec:
                row[f"Polarization {F}"]  = fme([(rec['P_pct'], rec['err_pct'])])
                row[f"Angle {F}"] = fme([(rec['theta_deg'], rec['err_theta_deg'])])
                row[f"P ISM - {F}"] = fme([(rec.get('p_ism',0), rec.get('err_p_ism',0))])
                row[f"θ ISM - {F}"] = fme([(rec.get('θ_ism',0), rec.get('err_θ_ism',0))])
            else:
                # si no hay rec, dejo todas las columnas vacías
                for col in [
                    f"Polarization {F}",
                    f"Angle {F}",
                    f"P ISM - {F}",
                    f"θ ISM - {F}"
                ]:
                    row[col] = ''
        row['RA']  = info.get('ra_deg')
        row['DEC'] = info.get('dec_deg')
        rows_pdf.append(row)

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
    col_width = 36 * mm  # ajusta este ancho si necesitas más o menos espacio
    page_width = margin*2 + col_width * n_cols
    page_height = 250 * mm
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
    for row in rows_pdf:
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
            (f"Polarization {F}",  "Polarización P ± error en %"),
            (f"Angle {F}", "Ángulo θ ± error en °"),
            (f"P ISM - {F}",       "Polarización del medio (%) ± error (%)"),
            (f"θ ISM - {F}",       "Ángulo del medio (°) ± error (°)")
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
  
def fme(pairs):
    """
    Format a (measurement, error) pair so that both numbers:
     - Have at least 3 decimal places.
     - Extend to the first non-zero digit in either the measurement or the error,
       if that occurs beyond the 3rd decimal.

    Args:
        pairs (tuple or list of tuple): Either a single (measurement, error) tuple
                                        or a list containing one such tuple.

    Returns:
        str: A string "measurement ± error" with the appropriate number of decimals.
    """
    # accept either a single tuple or a list of tuples
    if isinstance(pairs, tuple) and not isinstance(pairs[0], (list, tuple)):
        measurement, error = pairs
    else:
        measurement, error = pairs[0]

    # convert to Decimal
    try:
        m = Decimal(str(measurement))
        e = Decimal(str(error))
    except InvalidOperation:
        print(f"Cannot convert {measurement} or {error} to Decimal")
        return "-"

    def decimals_needed(x: Decimal):
        """
        Return the number of decimal places you must show to reach
        the first non-zero digit in x’s fractional part.
        If x == 0, returns 0.
        
        Examples:
        0.00432 → 3   (first non-zero is the '4' in 0.00[4]32)
        0.020   → 2   (0.[0]2)
        3.14159 → 1   (0.[1]4159)
        1.2300  → 1   (effectively 1.23 → 1st place is non-zero)
        """
        if x == 0:
            return 0

        # Remove trailing zeros and get exponent + digits
        t = x.normalize().as_tuple()
        digits = t.digits
        exp = t.exponent  # negative for fractional

        n_frac = -exp       # how many total fractional places
        n_sig  = len(digits)  # how many significant digits after
        # position of first non-zero = total_frac - sig_digits + 1
        pos = n_frac - n_sig + 1

        # at least 1 decimal needed if there's any fractional part
        return max(pos, 1)

    # compute needed decimals for measurement and error
    dm = decimals_needed(m)
    de = decimals_needed(e)

    # at least 3 decimals, but extend to whichever needs more
    nd = max(3, dm, de)

    # build format string
    fmt = f"{{:.{nd}f}}"

    return f"{fmt.format(m)} ± {fmt.format(e)}"

def fmt(val):
    return val if isinstance(val, (int, float)) else str(val)

def compute_ism_pol_angle(q, u, sigma_q, sigma_u):
    """
    Dadas las componentes Q_ISM y U_ISM (floats o strings), 
    devuelve (P_ISM, theta_ISM):
      P_ISM = sqrt(q² + u²)
      theta_ISM = 0.5 * atan2(u, q) en grados
    Si q o u no son convertibles a float, devuelve (None, None).
    """
    # polarización P y ángulo θ
    P      = math.hypot(q, u)
    theta  = 0.5 * math.degrees(math.atan2(u, q))

    # σ_P
    if P > 0:
        sigma_P = math.sqrt((q/P)**2 * sigma_q**2 +
                            (u/P)**2 * sigma_u**2)
    else:
        sigma_P = 0.0

    # σ_theta (rad)
    denom = (q**2 + u**2)
    if denom > 0:
        var_theta_rad = 0.25 * ((u/denom)**2 * sigma_q**2 +
                                (q/denom)**2 * sigma_u**2)
        sigma_theta = math.degrees(math.sqrt(var_theta_rad))
    else:
        sigma_theta = 0.0

    return P, sigma_P, theta, sigma_theta
