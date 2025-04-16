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

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ZScaleInterval
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import Circle
from collections import Counter
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

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

if __name__ == '__main__':
    image_path = 'caf-20231114-22_35_48-sci-blap_b_f.fits'
    image_data, sources, candidate_pairs, final_pairs, mode_distance, mode_angle = process_image(
        image_path, fwhm=3.0, threshold_multiplier=5.0, tol_distance=1.44, tol_angle=1.2, max_distance=75)
    print("Número de parejas candidatas:", len(candidate_pairs))
    print("Número de parejas finales:", len(final_pairs))
    draw_pairs(image_data, sources, final_pairs, len(sources), mode_distance, mode_angle, 1.44, 1.2)
