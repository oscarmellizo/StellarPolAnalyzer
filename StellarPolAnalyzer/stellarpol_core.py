"""
stellarpol_core.py

Este módulo contiene la lógica principal para el análisis de imágenes polarimétricas.
Se incluyen funciones para:
- Detectar estrellas.
- Calcular la distancia y el ángulo entre estrellas.
- Generar parejas candidatas y filtrarlas.
- Dibujar los resultados.
- Función de alto nivel: process_image.
"""

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.stats import sigma_clipped_stats
from astropy.modeling.models import Gaussian2D
from astropy.wcs import WCS
from astropy import coordinates as coord
from astroquery.astrometry_net import AstrometryNet
from astroquery.simbad import Simbad
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from photutils import DAOStarFinder, CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from sklearn.neighbors import NearestNeighbors
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from collections import Counter

def detect_stars(image_data, fwhm=3.0, threshold_multiplier=5.0):
    """
    Detecta estrellas en la imagen usando DAOStarFinder.
    Permite ajustar fwhm y threshold_multiplier.
    Retorna una lista de fuentes con atributos (como xcentroid y ycentroid).
    """
    mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_multiplier * std)
    sources = daofind(image_data - median)
    print(f"Se detectaron {len(sources)} estrellas.")
    return sources

def compute_distance_angle(p1, p2):
    """
    Calcula la distancia y el ángulo (en grados) entre dos puntos p1 y p2.
    Se devuelve el ángulo normalizado al "ángulo mínimo", es decir,
    se toma el valor absoluto y, si es mayor a 90°, se usa 180° menos el valor.
    Esto hace que ángulos complementarios se conviertan en el mismo valor.
    """
    x1, y1 = p1
    x2, y2 = p2
    distance = np.hypot(x2 - x1, y2 - y1)
    raw_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    angle = abs(raw_angle)
    if angle > 90:
        angle = 180 - angle
    return distance, angle

def find_candidate_pairs(sources, max_distance=75):
    """
    Para cada estrella, encuentra todos los vecinos dentro de un radio 'max_distance'.
    Genera parejas candidatas (i, j, distance, angle) para cada par (i, j) con i < j, 
    evitando duplicados.
    """
    coords = np.array([(s['xcentroid'], s['ycentroid']) for s in sources])
    nn = NearestNeighbors(radius=max_distance, algorithm='ball_tree')
    nn.fit(coords)
    distances_list, indices_list = nn.radius_neighbors(coords, return_distance=True)
    
    candidate_pairs = []
    for i in range(len(coords)):
        for j, d in zip(indices_list[i], distances_list[i]):
            if j <= i:  # Considerar solo pares con j > i
                continue
            p1 = coords[i]
            p2 = coords[j]
            distance, angle = compute_distance_angle(p1, p2)
            candidate_pairs.append((i, j, distance, angle))
    return candidate_pairs

def filter_pairs_by_mode(candidate_pairs, tol_distance=0.52, tol_angle=0.30):
    """
    Redondea a dos decimales las distancias y ángulos de los pares candidatos.
    Calcula la moda de las distancias y ángulos y filtra los pares cuyos valores
    redondeados estén dentro de la tolerancia:
      |distancia - moda_distancia| <= tol_distance y |ángulo - moda_ángulo| <= tol_angle.
    Retorna una tupla: (final_pairs, distance_mode, angle_mode)
    """
    if not candidate_pairs:
        return [], None, None
    
    distances = [round(p[2], 2) for p in candidate_pairs]
    angles = [round(p[3], 2) for p in candidate_pairs]
    
    distance_mode = Counter(distances).most_common(1)[0][0]
    angle_mode = Counter(angles).most_common(1)[0][0]
    
    print(f"Modo de distancia: {distance_mode} px, Modo de ángulo: {angle_mode}°")
    
    final_pairs = []
    for (i, j, d, a) in candidate_pairs:
        if abs(round(d, 2) - distance_mode) <= tol_distance and abs(round(a, 2) - angle_mode) <= tol_angle:
            final_pairs.append((i, j, d, a))
    
    star_counts = Counter()
    for (i, j, d, a) in final_pairs:
        star_counts[i] += 1
        star_counts[j] += 1
    for star, count in star_counts.items():
        if count > 1:
            print(f"La estrella {star} aparece en {count} parejas.")
            
    return final_pairs, distance_mode, angle_mode

def draw_pairs(image_data, sources, pairs, num_stars, mode_distance, mode_angle, tol_distance, tol_angle):
    """
    Dibuja la imagen y las parejas encontradas. Se coloca un punto rojo para cada estrella.
    Para cada pareja se dibuja una línea lime (lw=0.5) entre los centros, un círculo azul
    alrededor de la estrella con menor X y un círculo rojo alrededor de la estrella con mayor X.
    Se añade una leyenda fuera de la gráfica (margen derecho) con el número de estrellas,
    parejas finales y los parámetros de polarimetría.
    """
    interval = ZScaleInterval()
    z1, z2 = interval.get_limits(image_data)
    
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(image_data, cmap='gray', origin='lower', vmin=z1, vmax=z2)
    ax.set_title('StellarPol Analyzer')
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')
    
    coords = np.array([(s['xcentroid'], s['ycentroid']) for s in sources])
    
    for idx, (x, y) in enumerate(coords):
        ax.plot(x, y, marker='o', markersize=1, color='red')
        ax.text(x + 2, y + 2, str(idx), color='blue', fontsize=6)
    
    for (i, j, d, a) in pairs:
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        ax.plot([x1, x2], [y1, y2], color='lime', lw=0.5)
        if x1 < x2:
            left_idx, right_idx = i, j
        else:
            left_idx, right_idx = j, i
        x_left, y_left = coords[left_idx]
        x_right, y_right = coords[right_idx]
        circ_left = Circle((x_left, y_left), radius=5, edgecolor='blue', facecolor='none', lw=0.5)
        circ_right = Circle((x_right, y_right), radius=5, edgecolor='red', facecolor='none', lw=0.5)
        ax.add_patch(circ_left)
        ax.add_patch(circ_right)
        print(f"Pareja ({i}, {j}): Distancia = {d:.2f} px, Ángulo = {a:.2f}°")
    
    plt.subplots_adjust(right=0.7)
    info_text = (f"Estrellas detectadas: {num_stars}\n"
                 f"Parejas finales: {len(pairs)}\n"
                 f"Distancia: {mode_distance} ± {tol_distance} px\n"
                 f"Ángulo: {mode_angle} ± {tol_angle}°")
    plt.figtext(0.72, 0.5, info_text, fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    plt.show()

def write_candidate_pairs_to_file(candidate_pairs, filename="candidate_pairs.txt"):
    """
    Escribe la lista de pares candidatos en un archivo de texto.
    Cada línea contendrá: Estrella A, Estrella B, Distancia (px) y Ángulo (°).
    """
    with open(filename, "w") as f:
        f.write("Estrella_A\tEstrella_B\tDistancia_px\tÁngulo_deg\n")
        for (i, j, d, a) in candidate_pairs:
            f.write(f"{i}\t{j}\t{d:.2f}\t{a:.2f}\n")
    print(f"Se han escrito {len(candidate_pairs)} candidatos en el archivo '{filename}'.")

def process_image(image_path, fwhm=3.0, threshold_multiplier=5.0, tol_distance=0.52, tol_angle=0.30, max_distance=50):
    """Procesa la imagen y retorna los resultados."""
    with fits.open(image_path) as hdul:
        image_data = hdul[0].data
    sources = detect_stars(image_data, fwhm=fwhm, threshold_multiplier=threshold_multiplier)
    candidate_pairs = find_candidate_pairs(sources, max_distance=max_distance)
    final_pairs, mode_distance, mode_angle = filter_pairs_by_mode(candidate_pairs, tol_distance=tol_distance, tol_angle=tol_angle)
    return image_data, sources, candidate_pairs, final_pairs, mode_distance, mode_angle


def align_images(reference_image, image_to_align):
    """
    Calcula la traslación necesaria para alinear image_to_align con reference_image
    utilizando phase_cross_correlation y luego aplica la traslación con scipy.ndimage.shift.
    Retorna la imagen alineada y el vector de desplazamiento.
    """
    shift_estimation, error, diffphase = phase_cross_correlation(reference_image, image_to_align, upsample_factor=10)
    aligned_image = shift(image_to_align, shift=shift_estimation)
    return aligned_image, shift_estimation

def save_fits_with_same_headers(original_filename, new_image, output_filename):
    """
    Guarda la imagen en un nuevo archivo FITS conservando el header original.
    """
    with fits.open(original_filename) as hdul:
        header = hdul[0].header
    hdu = fits.PrimaryHDU(data=new_image, header=header)
    hdu.writeto(output_filename, overwrite=True)
    print(f"Se ha guardado el archivo: {output_filename}")

def process_four_polarimetric_images(ref_path, other_paths,
                                     fwhm=3.0, threshold_multiplier=5.0,
                                     tol_distance=0.52, tol_angle=0.30,
                                     max_distance=75):
    """
    Procesa un conjunto de 4 imágenes polarimétricas correspondientes a diferentes ángulos.
    
    Parámetros:
      ref_path: string
          Path de la imagen de referencia (por ejemplo, la imagen tomada a 0°).
      other_paths: list of strings
          Lista con los paths de las otras 3 imágenes.
      fwhm: float, opcional (default=3.0)
          Valor de FWHM para la detección de estrellas.
      threshold_multiplier: float, opcional (default=5.0)
          Multiplicador del threshold para la detección de estrellas.
      tol_distance: float, opcional (default=0.52)
          Tolerancia en la distancia (en píxeles) para filtrar las parejas.
      tol_angle: float, opcional (default=0.30)
          Tolerancia en el ángulo (en grados) para filtrar las parejas.
      max_distance: float, opcional (default=75)
          Radio máximo (en píxeles) para encontrar parejas, usado en process_image.
      align_max_distance: float, opcional (default=75)
          Radio máximo (en píxeles) para la alineación (usado en align_images).
    
    Flujo:
      1. Se carga la imagen de referencia (ref_path).
      2. Para cada uno de los paths en other_paths:
         - Se carga la imagen.
         - Se alinea con la imagen de referencia usando `align_images`.
         - Se guarda la imagen alineada utilizando `save_fits_with_same_headers`,
           generando un nuevo archivo (por ejemplo, con sufijo "-aligned.fits").
      3. Se crea una lista final de 4 paths: el ref_path y los 3 archivos alineados.
      4. Para cada imagen en esta lista, se llama a `process_image` utilizando los parámetros especificados.
    
    Retorna:
      final_image_paths: list of strings
          Lista de paths de las imágenes finales (la de referencia y las 3 alineadas).
      results: list
          Lista con los resultados de `process_image` para cada imagen.
    """
    
    # Cargar la imagen de referencia
    ref_data = fits.getdata(ref_path)
    
    # Lista para almacenar los paths finales
    final_image_paths = [ref_path]
    
    # Para cada imagen de las otras 3, alinear a la imagen de referencia y guardar la imagen alineada.
    for path in other_paths:
        img_data = fits.getdata(path)
        aligned_image, shift_est = align_images(ref_data, img_data)
        # Generamos el nombre del archivo alineado; por ejemplo, reemplazamos '.fits' por '-aligned.fits'
        output_filename = path.replace(".fits", "-aligned.fits")
        save_fits_with_same_headers(path, aligned_image, output_filename)
        final_image_paths.append(output_filename)
    
    # Procesar cada imagen utilizando process_image
    results = []
    for path in final_image_paths:
        result = process_image(path, fwhm=fwhm, threshold_multiplier=threshold_multiplier,
                                tol_distance=tol_distance, tol_angle=tol_angle, max_distance=max_distance)
        results.append(result)
    
    return final_image_paths, results

def compute_polarimetry_for_pairs(final_image_paths, sources, final_pairs,
                                  aperture_radius=5, r_in=7, r_out=10, SNR_threshold=5):
    """
    Calcula la fotometría y los parámetros de polarimetría para cada par de estrellas.

    Parámetros:
      final_image_paths : list of str
          Lista de 4 paths a las imágenes polarimétricas (ya alineadas) en el orden:
          [imagen_0°, imagen_22.5°, imagen_45°, imagen_67.5°].
      sources : list or Table
          Lista de fuentes detectadas en la imagen de referencia, con campos 'xcentroid' y 'ycentroid'.
      final_pairs : list of tuples
          Lista de parejas finales, cada una del tipo (i, j, distance, angle), donde i y j son índices en `sources`.
      aperture_radius : float, opcional
          Radio de la apertura (en píxeles) para realizar la fotometría de apertura.
      r_in : float, opcional
          Radio interior para el anillo de fondo.
      r_out : float, opcional
          Radio exterior para el anillo de fondo.
      SNR_threshold : float, opcional
          Valor mínimo de SNR para considerar la medición válida.

    Retorna:
      results : list of dict
          Cada elemento corresponde a un par de estrellas y contiene:
            - 'pair_indices' : (i, j)
            - 'fluxes' : {angle: {'ord_flux': ..., 'ext_flux': ..., 'ND': ..., 'error': ...}}
                          para cada uno de los 4 ángulos.
            - 'q' : Valor calculado de q (en porcentaje)
            - 'u' : Valor calculado de u (en porcentaje)
            - 'P' : Grado de polarización
            - 'theta' : Ángulo de polarización (en grados)
    """

    # Obtener las posiciones para cada par usando la imagen de referencia
    # Se asume que final_pairs contiene índices (i, j) que refieren a elementos en `sources`
    # y que para cada par, el de menor x se considera "ordinario" y el de mayor x, "extraordinario".
    source_positions = np.array([(src['xcentroid'], src['ycentroid']) for src in sources])

    # Para cada par, determina la posición de cada miembro.
    pair_positions = []
    for (i, j, dist, ang) in final_pairs:
        pos_i = source_positions[i]
        pos_j = source_positions[j]
        if pos_i[0] < pos_j[0]:
            # pos_i es ordinario y pos_j es extraordinario
            pair_positions.append({'ord': pos_i, 'ext': pos_j})
        else:
            pair_positions.append({'ord': pos_j, 'ext': pos_i})

    # Inicializa un diccionario para almacenar las fotometrías para cada imagen.
    # Usaremos el ángulo de cada imagen según el orden de final_image_paths.
    # Asumimos: final_image_paths[0] corresponde a 0°, [1] a 22.5°, [2] a 45° y [3] a 67.5°.
    image_angles = [0.0, 22.5, 45.0, 67.5]
    # Guardaremos para cada par un diccionario: fluxes[angle] = {'ord_flux': valor, 'ext_flux': valor, 'ND': valor, 'error': valor}
    pair_fluxes = [dict() for _ in range(len(pair_positions))]

    # Para cada imagen de polarización
    for path, pol_angle in zip(final_image_paths, image_angles):
        with fits.open(path) as hdul:
            data = hdul[0].data

        # Realizar la fotometría para cada par
        # Se crearán aperturas para la posición ordinaria y extraordinaria de cada par.
        ord_positions = np.array([pair['ord'] for pair in pair_positions])
        ext_positions = np.array([pair['ext'] for pair in pair_positions])
        ord_apertures = CircularAperture(ord_positions, r=aperture_radius)
        ext_apertures = CircularAperture(ext_positions, r=aperture_radius)
        # Define anillos de fondo, opcionalmente
        ord_annulus = CircularAnnulus(ord_positions, r_in=r_in, r_out=r_out)
        ext_annulus = CircularAnnulus(ext_positions, r_in=r_in, r_out=r_out)

        # Obtener estadísticas de fondo
        ord_stats = ApertureStats(data, ord_annulus)
        ext_stats = ApertureStats(data, ext_annulus)
        ord_bkg = ord_stats.mean * ord_apertures.area
        ext_bkg = ext_stats.mean * ext_apertures.area

        # Fotometría en ambas aperturas
        ord_phot_table = aperture_photometry(data, ord_apertures)
        ext_phot_table = aperture_photometry(data, ext_apertures)
        # Resta del fondo
        ord_flux = ord_phot_table['aperture_sum'] - ord_bkg
        ext_flux = ext_phot_table['aperture_sum'] - ext_bkg

        # Calcular SNR para cada medición (usando una aproximación)
        ord_SNR = ord_flux / np.sqrt(ord_flux + ord_apertures.area * ord_stats.std**2)
        ext_SNR = ext_flux / np.sqrt(ext_flux + ext_apertures.area * ext_stats.std**2)

        # Para cada par, guardar fotometría si SNR > SNR_threshold, de lo contrario se marca como no fiable.
        for idx in range(len(pair_positions)):
            # Si cualquiera de las mediciones es negativa o con SNR insuficiente, se omite.
            if (ord_flux[idx] <= 0) or (ext_flux[idx] <= 0):
                continue
            if (ord_SNR[idx] < SNR_threshold) or (ext_SNR[idx] < SNR_threshold):
                continue

            # Calcular el valor normalizado de diferencia ND para este ángulo:
            # ND = (ext - ord) / (ext + ord)
            ND = (ext_flux[idx] - ord_flux[idx]) / (ext_flux[idx] + ord_flux[idx])
            # Se estima un error simple para ND (esto puede refinarse)
            # Por ejemplo, error = 0.5/sqrt(flux * factor) como en tu código, aquí se usa un factor arbitrario.
            error = 0.5 / np.sqrt(abs(ord_flux[idx] + ext_flux[idx]))  # ejemplo simple

            # Almacena los resultados para este par para el ángulo actual.
            pair_fluxes[idx][pol_angle] = {
                'ord_flux': ord_flux[idx],
                'ext_flux': ext_flux[idx],
                'ND': ND,
                'error': error
            }
    
    # Ahora, para cada par, se deben combinar las mediciones para calcular los parámetros de polarimetría.
    # Suponiendo que usamos la convención:
    # q = ((ND[0°] - ND[45°]) / 2) * 100 
    # u = ((ND[22.5°] - ND[67.5°]) / 2) * 100
    # (Aquí se puede ajustar según tu calibración)
    polarimetry_results = []
    for idx, flux_dict in enumerate(pair_fluxes):
        # Verificar que se tienen mediciones para todos los ángulos necesarios
        if (0.0 not in flux_dict) or (45.0 not in flux_dict) or (22.5 not in flux_dict) or (67.5 not in flux_dict):
            continue

        ND0 = flux_dict[0.0]['ND']
        ND22 = flux_dict[22.5]['ND']
        ND45 = flux_dict[45.0]['ND']
        ND67 = flux_dict[67.5]['ND']
        # Promedio simple de los errores:
        err_total = np.mean([flux_dict[angle]['error'] for angle in [0.0, 22.5, 45.0, 67.5]])

        q = ((ND0 - ND45) / 2.0) * 100.0
        u = ((ND22 - ND67) / 2.0) * 100.0
        P = np.sqrt(q**2 + u**2)
        theta = 0.5 * np.degrees(np.arctan2(u, q))
        # Propagación de errores simple
        err_theta = (0.5 * err_total) * 100 if P != 0 else 0.0

        polarimetry_results.append({
            'pair_index': idx,
            'q': q,
            'u': u,
            'P': P,
            'theta': theta,
            'error': err_total,
            'fluxes': flux_dict
        })
    
    return polarimetry_results

def compute_full_polarimetry(ref_path, other_paths,
                             fwhm=3.0, threshold_multiplier=5.0,
                             tol_distance=0.52, tol_angle=0.30, max_distance=75,
                             phot_aperture_radius=5, r_in=7, r_out=10, SNR_threshold=5):
    """
    Integra el procesamiento de 4 imágenes polarimétricas y realiza la fotometría
    para calcular los parámetros de polarimetría para cada pareja de estrellas detectadas.

    Parámetros:
      - ref_path: string
          Path de la imagen de referencia (por ejemplo, la imagen a 0°).
      - other_paths: list of strings
          Lista de paths de las otras 3 imágenes (por ejemplo, las tomadas a 22.5°, 45° y 67.5°).
      - fwhm: float (default=3.0)
          Parámetro FWHM para la detección de estrellas.
      - threshold_multiplier: float (default=5.0)
          Multiplicador del umbral para la detección de estrellas.
      - tol_distance: float (default=0.52)
          Tolerancia en la distancia (en píxeles) para el filtrado de parejas.
      - tol_angle: float (default=0.30)
          Tolerancia en el ángulo (en grados) para el filtrado de parejas.
      - max_distance: float (default=75)
          Radio máximo (en píxeles) para buscar parejas en el método process_image.
      - phot_aperture_radius: float (default=5)
          Radio de la apertura para la fotometría.
      - r_in: float (default=7)
          Radio interior para el anillo de fondo.
      - r_out: float (default=10)
          Radio exterior para el anillo de fondo.
      - SNR_threshold: float (default=5)
          Valor mínimo de SNR para considerar válida la medición fotométrica.
      
    Flujo:
      1. Se llama al método process_four_polarimetric_images (que debe formar parte de la librería)
         para alinear y procesar las 4 imágenes. Esto retorna (final_image_paths, results), donde 
         results es una lista con, para cada imagen, una tupla:
         (image_data, sources, candidate_pairs, final_pairs, mode_distance, mode_angle).
      2. Se toma la lista de final_image_paths y se extrae, de la imagen de referencia (por ejemplo, results[0]),
         los parámetros sources y final_pairs.
      3. Se llama a compute_polarimetry_for_pairs (otro método de la librería) pasando:
            - final_image_paths,
            - sources (de la imagen de referencia),
            - final_pairs,
            - y los parámetros de fotometría (aperture_radius, r_in, r_out, SNR_threshold).
      4. Se retorna el resultado de compute_polarimetry_for_pairs, que es una lista de diccionarios
         con los parámetros de polarimetría (q, u, P, theta, errores, etc.) para cada par.

    Retorna:
      - process_four_polarimetric_images_results: list
          Lista con los resultados de `process_image` para cada imagen.
      - polarimetry_results: list of dict
          Lista con los parámetros de polarimetría calculados para cada par.
      - final_image_paths: list of strings
          Lista de paths de las imágenes finales (la de referencia y las 3 alineadas).
    """
    # Primero, procesar las 4 imágenes con el método existente
    final_image_paths, process_four_polarimetric_images_results = process_four_polarimetric_images(ref_path, other_paths,
                                                                   fwhm=fwhm,
                                                                   threshold_multiplier=threshold_multiplier,
                                                                   tol_distance=tol_distance,
                                                                   tol_angle=tol_angle,
                                                                   max_distance=max_distance)
    # Supongamos que la imagen de referencia es la primera de results.
    # Se extraen 'sources' y 'final_pairs' desde la primera imagen.
    # Cada elemento de results es una tupla:
    # (image_data, sources, candidate_pairs, final_pairs, mode_distance, mode_angle)
    if not process_four_polarimetric_images_results:
        raise ValueError("No se obtuvieron resultados del procesamiento de imágenes.")
    
    ref_result = process_four_polarimetric_images_results[0]
    sources = ref_result[1]
    final_pairs = ref_result[3]
    
    # Llamar al método de fotometría que calcula los parámetros de polarimetría para los pares
    # Este método debe estar definido en tu librería, por ejemplo: compute_polarimetry_for_pairs
    polarimetry_results = compute_polarimetry_for_pairs(final_image_paths, sources, final_pairs,
                                                        aperture_radius=phot_aperture_radius,
                                                        r_in=r_in, r_out=r_out,
                                                        SNR_threshold=SNR_threshold)
    return process_four_polarimetric_images_results, polarimetry_results, final_image_paths

def annotate_with_astrometry_net(ref_path, sources, final_pairs, polarimetry_results,
                                 fwhm=3.0, api_key=None, simbad_radius=0.01*u.deg,
                                 synthetic_name="synthetic.fits"):
    """
    A partir de los resultados de compute_full_polarimetry:
      - genera una imagen sintética con Gaussian2D en las posiciones ordinarias
        de cada par (con amplitud = suma de flujos ord+ext en 0°),
      - guarda el FITS sintético,
      - lo resuelve con Astrometry.Net,
      - convierte esas posiciones pixel -> sky,
      - consulta SIMBAD para cada coordenada,
      - y anota en cada dict de polarimetry_results los campos:
         'ra', 'dec', 'simbad_id'.

    Parámetros
    ----------
    ref_path : str
        Path al FITS de referencia (0°), se usa su header para tamaño y metadatos.
    sources : list/Table
        Fuentes detectadas en esa imagen, con campos 'xcentroid','ycentroid'.
    final_pairs : list of tuples
        Lista de pares (i, j, dist, ang) según compute_full_polarimetry.
    polarimetry_results : list of dict
        Salida de compute_polarimetry_for_pairs, cada dict debe incluir:
          - 'pair_index'  → índice en final_pairs,
          - 'fluxes'[0.0]['ord_flux'] y ['ext_flux'].
    fwhm : float
        FWHM (px) para convertir a sigma y generar los Gaussianos.
    api_key : str
        Tu API key de Astrometry.Net.
    simbad_radius : Quantity
        Radio de búsqueda en SIMBAD alrededor de cada posición resuelta.
    synthetic_name : str
        Nombre de archivo donde se escribirá el FITS sintético.

    Retorna
    -------
    wcs : astropy.wcs.WCS
        La solución WCS obtenida de Astrometry.Net.
    enriched_results : list of dict
        Mismo lista de polarimetry_results, pero cada dict ahora incluye:
          - 'ra', 'dec'        : coordenadas ICRS en grados,
          - 'simbad_id'        : nombre SIMBAD o "No_ID".
    """
    # 1) Leemos header de la imagen de referencia
    with fits.open(ref_path) as hdul:
        hdr = hdul[0].header
    nx = hdr['NAXIS1']; ny = hdr['NAXIS2']
    
    # 2) Convertir FWHM a sigma
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    # 3) Construir la imagen sintética
    syn = np.zeros((ny, nx))
    positions = []
    for entry in polarimetry_results:
        idx = entry['pair_index']
        i, j, _, _ = final_pairs[idx]
        x = sources[i]['xcentroid']
        y = sources[i]['ycentroid']
        positions.append((x, y))
        # amplitud = suma de ord + ext en 0°
        amp = entry['fluxes'][0.0]['ord_flux'] + entry['fluxes'][0.0]['ext_flux']
        g = Gaussian2D(amplitude=amp, x_mean=x, y_mean=y,
                       x_stddev=sigma, y_stddev=sigma)
        y_grid, x_grid = np.mgrid[0:ny, 0:nx]
        syn += g(x_grid, y_grid)
    
    # 4) Guardar FITS sintético
    syn_hdr = fits.Header()
    for key in ('RA','DEC','OBJECT'):
        if key in hdr:
            syn_hdr[key] = hdr[key]
    fits.PrimaryHDU(syn, header=syn_hdr).writeto(synthetic_name, overwrite=True)
    
    # 5) Resolver con Astrometry.Net
    ast = AstrometryNet()
    ast.api_key = api_key
    sol = ast.solve_from_image(synthetic_name)
    wcs_hdr = fits.Header()
    for k,v in sol.items():
        wcs_hdr[k] = v
    wcs = WCS(wcs_hdr)
    
    # 6) Pixel → sky
    pix = np.array(positions)
    world = wcs.all_pix2world(pix, 1)  # array Nx2: [ra,dec]
    
    # 7) Consultar SIMBAD para cada coordenada
    Simbad.reset_votable_fields()
    Simbad.add_votable_fields('otype')
    enriched = []
    for entry, (ra, dec) in zip(polarimetry_results, world):
        sc = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        res = Simbad.query_region(sc, radius=simbad_radius)
        if res and len(res)>0:
            obj = res['MAIN_ID'][0].decode('utf-8') if isinstance(res['MAIN_ID'][0], bytes) else res['MAIN_ID'][0]
        else:
            obj = "No_ID"
        e = entry.copy()
        e['ra'] = ra
        e['dec'] = dec
        e['simbad_id'] = obj
        enriched.append(e)
    
    return wcs, enriched

def run_complete_polarimetric_pipeline(ref_path,
                                        other_paths,
                                        pol_angles,
                                        fwhm=3.0,
                                        threshold_multiplier=5.0,
                                        tol_distance=0.52,
                                        tol_angle=0.30,
                                        max_distance=75,
                                        phot_aperture_radius=5,
                                        r_in=7,
                                        r_out=10,
                                        SNR_threshold=5,
                                        astrometry_api_key=None,
                                        simbad_radius=0.01*u.deg,
                                        synthetic_name="synthetic.fits"):
    """
    Ejecuta en secuencia todo el flujo de trabajo polarimétrico y de astrometría:
      1) Alinea y procesa las 4 imágenes polarimétricas, detecta fuentes y empareja estrellas.
      2) Realiza la fotometría sobre cada par y calcula q, u, P y theta.
      3) Crea una imagen sintética de las componentes ordinarias de cada par.
      4) Envía la sintética a Astrometry.Net, resuelve WCS y actualiza header.
      5) Convierte las posiciones pixel → sky y consulta SIMBAD para anotar nombres.

    Parámetros
    ----------
    ref_path : str
        Path de la imagen de referencia (por ejemplo, la de 0°).
    other_paths : list of str
        Paths de las otras 3 imágenes.
    pol_angles : list of float
        Ángulos de polarización en grados [0°, 22.5°, 45°, 67.5°].
    fwhm, threshold_multiplier, tol_distance, tol_angle, max_distance : float
        Parámetros para el emparejamiento de estrellas.
    phot_aperture_radius, r_in, r_out, SNR_threshold : float
        Parámetros para la fotometría de cada par.
    astrometry_api_key : str
        API key para Astrometry.Net (si None, no resuelve).
    simbad_radius : Quantity
        Radio para consulta SIMBAD.
    synthetic_name : str
        Nombre de fichero FITS sintético.

    Retorna
    -------
    final_image_paths : list of str
        Paths de las 4 imágenes procesadas (alineadas) usadas.
    polar_results : list of dict
        Lista con q, u, P, theta, errores y flujos para cada par.
    wcs : astropy.wcs.WCS
        Solución WCS obtenida de Astrometry.Net.
    enriched_results : list of dict
        Lista `polar_results` enriquecida con 'ra', 'dec' y 'simbad_id' por par.
    """
    # Paso 1–2: polarimetría
    proc_results, polar_results, final_image_paths = compute_full_polarimetry(
        ref_path,
        other_paths,
        fwhm=fwhm,
        threshold_multiplier=threshold_multiplier,
        tol_distance=tol_distance,
        tol_angle=tol_angle,
        max_distance=max_distance,
        phot_aperture_radius=phot_aperture_radius,
        r_in=r_in,
        r_out=r_out,
        SNR_threshold=SNR_threshold
    )

    # Extraer sources y final_pairs desde el resultado de la imagen de referencia
    _, sources, _, final_pairs, _, _ = proc_results[0]

    # Paso 3–5: astrometría + SIMBAD
    wcs, enriched = annotate_with_astrometry_net(
        ref_path,
        sources,
        final_pairs,
        polar_results,
        fwhm=fwhm,
        api_key=astrometry_api_key,
        simbad_radius=simbad_radius,
        synthetic_name=synthetic_name
    )

    return final_image_paths, polar_results, wcs, enriched