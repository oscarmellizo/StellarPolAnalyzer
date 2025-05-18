"""
photometry.py

Aperture photometry and polarimetric analysis for paired stars across multi-angle images.

Este módulo realiza:
  - Fotometría de apertura con resta de fondo para cada componente ordinaria/extraordinaria.
  - Cálculo de SNR y, si SNR_threshold=-1, selección automática del umbral basado en la mediana de SNR.
  - Generación de histograma de SNR (opcional) guardado en report_dir.
  - Cómputo de diferencias normalizadas NDθ.
  - Cálculo de parámetros de Stokes q, u, grado de polarización P, ángulo θ y propagación de errores.
"""

import os
import numpy as np
from astropy.io import fits
from photutils import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def compute_polarimetry_for_pairs(final_image_paths, sources, final_pairs,
                                  aperture_radius=5, r_in=7, r_out=10,
                                  SNR_threshold=5,
                                  save_histogram=False, report_dir=None,
                                  hist_filename="snr_hist.png"):
    """
    Perform aperture photometry on each star pair across four polarization images
    and compute Stokes parameters q, u, degree of polarization P, and polarization angle θ,
    con gestión dinámica de umbral de SNR si SNR_threshold=-1.

    Parameters
    ----------
    final_image_paths : list of str
        Paths a las cuatro imágenes FITS alineadas en orden [0°, 22.5°, 45°, 67.5°].
    sources : astropy.table.Table o lista de dict-like
        Fuentes detectadas en la imagen de referencia, con 'xcentroid' y 'ycentroid'.
    final_pairs : list of tuple
        Parejas finales (i, j, distance, angle) referenciando índices en `sources`.
    aperture_radius : float
        Radio de apertura (px) para fotometría (default: 5).
    r_in, r_out : float
        Radios interior y exterior (px) del anillo de fondo.
    SNR_threshold : float
        Umbral mínimo de SNR; si es -1, se calcula automáticamente como mediana de SNR.
    save_histogram : bool
        Si True, genera y guarda un histograma de la distribución de SNR en report_dir.
    report_dir : str or None
        Directorio donde guardar el histograma y el archivo de umbral (si se calcula).
    hist_filename : str
        Nombre del PNG de histograma de SNR (default: "snr_hist.png").

    Returns
    -------
    results : list of dict
        Lista de resultados por par, cada dict con:
          - 'pair_index'
          - 'fluxes': dict {angle: {ord_flux, ext_flux, ord_bkg, ext_bkg, ord_snr, ext_snr, ND, error}}
          - 'q','u','P','theta','error'
    """

    # 1) Preparar posiciones ordinarias/extraordinarias
    coords = np.array([(s['xcentroid'], s['ycentroid']) for s in sources])
    pair_positions = []
    for (i, j, *_ ) in final_pairs:
        p1, p2 = coords[i], coords[j]
        pair_positions.append((p1, p2) if p1[0] < p2[0] else (p2, p1))

    if len(pair_positions) == 0:
        return []

    angles = [0.0, 22.5, 45.0, 67.5]
    # Acumulador de SNR de todas las aperturas
    all_snr = []

    # Primera pasada: recabar todos los SNR
    for path in final_image_paths:
        data = fits.getdata(path)
        ord_pos = np.array([pp[0] for pp in pair_positions])
        ext_pos = np.array([pp[1] for pp in pair_positions])
        ord_ap = CircularAperture(ord_pos, r=aperture_radius)
        ext_ap = CircularAperture(ext_pos, r=aperture_radius)
        ord_ann = CircularAnnulus(ord_pos, r_in=r_in, r_out=r_out)
        ext_ann = CircularAnnulus(ext_pos, r_in=r_in, r_out=r_out)

        ord_stats = ApertureStats(data, ord_ann)
        ext_stats = ApertureStats(data, ext_ann)
        ord_bkg = ord_stats.mean * ord_ap.area
        ext_bkg = ext_stats.mean * ext_ap.area

        ord_tab = aperture_photometry(data, ord_ap)
        ext_tab = aperture_photometry(data, ext_ap)
        ord_flux = ord_tab['aperture_sum'] - ord_bkg
        ext_flux = ext_tab['aperture_sum'] - ext_bkg

        ord_snr = ord_flux / np.sqrt(ord_flux + ord_ap.area * ord_stats.std**2)
        ext_snr = ext_flux / np.sqrt(ext_flux + ext_ap.area * ext_stats.std**2)

        # Guardar todos los valores de SNR
        for idx in range(len(pair_positions)):
            if ord_flux[idx] > 0 and ext_flux[idx] > 0:
                all_snr.append(ord_snr[idx])
                all_snr.append(ext_snr[idx])

    # 2) Determinar umbral efectivo de SNR
    if SNR_threshold < 0:
        eff_SNR = calcula_umbral_snr_gmm(all_snr)
        
        # Guardar el umbral calculado
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
            with open(os.path.join(report_dir, "snr_threshold.txt"), "w") as f:
                f.write(f"Auto-calculated SNR_threshold = {eff_SNR:.2f}\n")
    else:
        eff_SNR = SNR_threshold

    # 3) Segunda pasada: llenar flux_tables con el umbral eff_SNR
    flux_tables = [{} for _ in pair_positions]
    for path, angle in zip(final_image_paths, angles):
        data = fits.getdata(path)
        ord_pos = np.array([pp[0] for pp in pair_positions])
        ext_pos = np.array([pp[1] for pp in pair_positions])
        ord_ap = CircularAperture(ord_pos, r=aperture_radius)
        ext_ap = CircularAperture(ext_pos, r=aperture_radius)
        ord_ann = CircularAnnulus(ord_pos, r_in=r_in, r_out=r_out)
        ext_ann = CircularAnnulus(ext_pos, r_in=r_in, r_out=r_out)

        ord_stats = ApertureStats(data, ord_ann)
        ext_stats = ApertureStats(data, ext_ann)
        ord_bkg = ord_stats.mean * ord_ap.area
        ext_bkg = ext_stats.mean * ext_ap.area

        ord_tab = aperture_photometry(data, ord_ap)
        ext_tab = aperture_photometry(data, ext_ap)
        ord_flux = ord_tab['aperture_sum'] - ord_bkg
        ext_flux = ext_tab['aperture_sum'] - ext_bkg

        ord_snr = ord_flux / np.sqrt(ord_flux + ord_ap.area * ord_stats.std**2)
        ext_snr = ext_flux / np.sqrt(ext_flux + ext_ap.area * ext_stats.std**2)

        for idx in range(len(pair_positions)):
            if (ord_flux[idx] > 0 and ext_flux[idx] > 0 and
                ord_snr[idx] >= eff_SNR and ext_snr[idx] >= eff_SNR):

                ND = (ext_flux[idx] - ord_flux[idx]) / (ext_flux[idx] + ord_flux[idx])
                err = 0.5 / np.sqrt(ord_flux[idx] + ext_flux[idx])

                flux_tables[idx][angle] = {
                    'ord_flux': float(ord_flux[idx]),
                    'ext_flux': float(ext_flux[idx]),
                    'ord_bkg':  float(ord_bkg[idx]),
                    'ext_bkg':  float(ext_bkg[idx]),
                    'ord_snr':  float(ord_snr[idx]),
                    'ext_snr':  float(ext_snr[idx]),
                    'ND':       float(ND),
                    'error':    float(err)
                }

    # 4) Histograma de SNR 
    if save_histogram and report_dir:
        os.makedirs(report_dir, exist_ok=True)
        plt.figure(figsize=(6, 4))
        filtered_snr = [snr for snr in all_snr if snr >= eff_SNR]
        plt.hist(filtered_snr, bins=30, color='gray', edgecolor='black')
        plt.xlabel('SNR')
        plt.ylabel('Count')
        plt.title('Distribution of Photometry SNR')
        out = os.path.join(report_dir, hist_filename)
        plt.savefig(out, bbox_inches='tight')
        plt.close()

    # 5) Cálculo de Stokes para pares completos
    results = []
    for idx, table in enumerate(flux_tables):
        if all(angle in table for angle in angles):
            ND0, ND45 = table[0.0]['ND'], table[45.0]['ND']
            ND22, ND67 = table[22.5]['ND'], table[67.5]['ND']
            err_avg = np.mean([table[ang]['error'] for ang in angles])

            q = ((ND0 - ND45) / 2.0) * 100.0
            u = ((ND22 - ND67) / 2.0) * 100.0
            P = np.hypot(q, u)
            theta = 0.5 * np.degrees(np.arctan2(u, q))

            results.append({
                'pair_index': idx,
                'fluxes'   : table,
                'q'        : q,
                'u'        : u,
                'P'        : P,
                'theta'    : theta,
                'error'    : err_avg
            })

    return results

def calcula_umbral_snr_gmm(snr_vals):
    """
    Ajusta un GMM de 2 componentes en log(SNR) y devuelve el valor de SNR
    donde las dos gaussianas tienen la misma probabilidad (el punto de corte).
    """
    snr = np.array(snr_vals)
    # Tomamos log para estabilizar varianza y manejar muy distintos rangos
    log_snr = np.log(snr[snr > 0]).reshape(-1,1)

    # Ajustamos GMM de 2 componentes
    gmm = GaussianMixture(n_components=2, random_state=0).fit(log_snr)
    means = gmm.means_.ravel()
    covs  = gmm.covariances_.ravel()
    weights = gmm.weights_.ravel()

    # P(x) = w1 N(x|m1,c1) - w2 N(x|m2,c2)  = 0 → punto de cruce
    # Resolvemos en x (en log-Escala) la ecuación:
    #   w1/(sqrt(c1)) * exp(-(x-m1)^2/(2c1)) = w2/(sqrt(c2)) * exp(-(x-m2)^2/(2c2))
    # Llevando al log y reorganizando:
    a = 1/(2*covs[0]) - 1/(2*covs[1])
    b = means[1]/covs[1] - means[0]/covs[0]
    c = (means[0]**2)/(2*covs[0]) - (means[1]**2)/(2*covs[1]) + \
        0.5*np.log((covs[0]*weights[1]**2)/(covs[1]*weights[0]**2))
    # La ecuación es: a x^2 + b x + c = 0
    disc = b**2 - 4*a*c
    if disc < 0:
        # No hay cruce real: nos quedamos con el percentil 90
        thresh_log = np.percentile(log_snr, 90)
    else:
        x1 = (-b + np.sqrt(disc)) / (2*a)
        x2 = (-b - np.sqrt(disc)) / (2*a)
        # Elegimos el que esté entre las dos medias
        candidates = [x for x in (x1, x2) if min(means)<x<max(means)]
        thresh_log = candidates[0] if candidates else np.percentile(log_snr, 90)

    return float(np.exp(thresh_log))