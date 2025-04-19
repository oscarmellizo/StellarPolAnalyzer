"""
pipeline.py

Funciones de orquestación para StellarPolAnalyzer:
- compute_full_polarimetry: procesa 4 imágenes polarimétricas y devuelve resultados de polarimetría
- run_complete_polarimetric_pipeline: ejecuta pipeline completo, incluyendo astrometría y cruces SIMBAD
"""
from .detection import process_image
from .alignment import align_images, save_fits_with_same_headers
from .photometry import compute_polarimetry_for_pairs
from .astrometry import annotate_with_astrometry_net
import astropy.units as u


def compute_full_polarimetry(ref_path, other_paths,
                             fwhm=3.0, threshold_multiplier=5.0,
                             tol_distance=0.52, tol_angle=0.30,
                             max_distance=75,
                             phot_aperture_radius=5, r_in=7, r_out=10, SNR_threshold=5):
    """
    Realiza polarimetría en 4 imágenes (ref + tres ángulos).

    Parámetros: idénticos a la documentación.

    Retorna:
      - process_results: lista con tuplas de process_image para cada FITS
      - polar_results  : lista de dicts con q,u,P,theta para cada par
      - final_paths    : lista de los 4 paths usados (ref + alineados)
    """
    # 1) Alinear imágenes
    ref_data = __import__('astropy.io.fits').io.fits.getdata(ref_path)
    final_paths = [ref_path]
    for path in other_paths:
        img = __import__('astropy.io.fits').io.fits.getdata(path)
        aligned, _ = align_images(ref_data, img)
        out = path.replace('.fits', '-aligned.fits')
        save_fits_with_same_headers(path, aligned, out)
        final_paths.append(out)

    # 2) Procesar cada imagen: detección + emparejado
    process_results = []
    for path in final_paths:
        res = process_image(path,
                            fwhm=fwhm,
                            threshold_multiplier=threshold_multiplier,
                            tol_distance=tol_distance,
                            tol_angle=tol_angle,
                            max_distance=max_distance)
        process_results.append(res)

    # 3) Fotometría y polarimetría sobre la imagen de referencia
    # Extraer fuentes y parejas finales
    _, sources, _, final_pairs, _, _ = process_results[0]
    polar_results = compute_polarimetry_for_pairs(
        final_paths, sources, final_pairs,
        aperture_radius=phot_aperture_radius,
        r_in=r_in, r_out=r_out,
        SNR_threshold=SNR_threshold)

    return process_results, polar_results, final_paths


def run_complete_polarimetric_pipeline(ref_path, other_paths, pol_angles,
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
    Ejecuta pipeline completo de polarimetría + astrometría.

    Parámetros adicionales:
      - astrometry_api_key: clave API para Astrometry.Net
      - simbad_radius     : radio de consulta en SIMBAD
      - synthetic_name    : nombre FITS sintético

    Retorna:
      - final_paths    : lista de paths de FITS usados
      - polar_results  : salida de compute_polarimetry_for_pairs
      - wcs            : astropy.wcs.WCS de la solución astrométrica
      - enriched       : lista de dicts con RA/DEC/SIMBAD añadidos
    """
    # Descartar args que no corresponden a compute_full_polarimetry
    process_results, polar_results, final_paths = compute_full_polarimetry(
        ref_path, other_paths,
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

    # Preparar datos para astrometría
    _, sources, _, final_pairs, _, _ = process_results[0]

    # Resolver y enriquecer con astrometry_net + SIMBAD
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
    return final_paths, polar_results, wcs, enriched
