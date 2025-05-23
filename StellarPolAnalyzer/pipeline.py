"""
pipeline.py

Pipeline orchestration for StellarPolAnalyzer: alignment, pairing, photometry, and astrometry.

This module exposes two high-level functions:

* `compute_full_polarimetry`:
    - Aligns a set of 4 polarimetric FITS images.
    - Detects stars and pairs them based on a fixed distance and angle pattern.
    - Performs aperture photometry on the reference image's pairs.

* `run_complete_polarimetric_pipeline`:
    - Executes `compute_full_polarimetry` for polarimetric processing.
    - Generates synthetic image, solves WCS via Astrometry.Net, and annotates SIMBAD IDs.
    - Optionally saves intermediate diagnostic plots.

Usage:
```python
from StellarPolAnalyzer.pipeline import run_complete_polarimetric_pipeline

final_paths, polar_results, wcs, enriched = run_complete_polarimetric_pipeline(
    ref_path="ref_0deg.fits",
    other_paths=["img22.fits","img45.fits","img67.fits"],
    pol_angles=[0,22.5,45,67.5],
    save_plots=True,
    report_dir="reports/assets",
    astrometry_api_key="YOUR_KEY"
)
```
"""
import os
import astropy.units as u
import json
import math
from astropy.io import fits
from .detection import process_image
from .alignment import align_images, save_fits_with_same_headers
from .photometry import compute_polarimetry_for_pairs
from .astrometry import annotate_with_astrometry_net
from .visualization import draw_pairs, save_plot, draw_apertures, plot_polarization_errors, plot_polarization_map, plot_histogram_P, plot_histogram_theta, plot_qu_diagram, plot_ism_gmm_histogram
from .report import generate_pdf_report
from .utils import estimate_ism_from_histograms

# Ángulos de referencia para Hiltner 960 (Schmidt et al. 1992), en grados
REF_ANGLES = {
  'B': 55.06, 'V': 54.79, 'R': 54.54, 'I': 53.96
}

# Polarization instrumental offsets (medidos sobre HD 154892; Q,U en %)
INST_OFFSETS = {
    'B': {'q': 3.4291041672626603, 'u': -4.88708822945143},
    'V': {'q': 2.5074183794553737, 'u': -5.272479544544009},
    'R': {'q': 1.4703385381906753, 'u': -4.967341304186951},
    'I': {'q': 1.1189172334736106, 'u': -4.125623240742233},
}

#INST_OFFSETS = {
#    'B': {'q': 0.0, 'u': 0.0},
#    'V': {'q': 0.0, 'u': 0.0},
#    'R': {'q': 0.0, 'u': 0.0},
#    'I': {'q': 0.0, 'u': 0.0},
#}

def compute_full_polarimetry(
    ref_path,
    other_paths,
    pol_angles,
    filter,
    fwhm=3.0,
    threshold_multiplier=5.0,
    tol_distance=0.52,
    tol_angle=0.30,
    max_distance=75,
    phot_aperture_radius=5,
    r_in=7,
    r_out=10,
    SNR_threshold=5,
    save_plots=False,
    report_dir=None
):
    """
    Perform polarimetric processing on four FITS images.

    Steps:
      1. Load and optionally display/save the reference image.
      2. Align the three other polarimetric images to the reference.
      3. Detect stars and find pairs in each aligned image.
      4. Conduct aperture photometry on the reference image's pairs.

    Parameters
    ----------
    ref_path : str
        Path to the reference FITS (e.g., 0° retarder angle).
    other_paths : list of str
        Paths to the other three FITS images (e.g., 22.5°, 45°, 67.5°).
    fwhm : float, optional
        Full-width at half-maximum for DAOStarFinder (default=3.0 px).
    threshold_multiplier : float, optional
        Detection threshold in sigma above background (default=5.0).
    tol_distance : float, optional
        Distance tolerance (px) for pairing (default=0.52 px).
    tol_angle : float, optional
        Angle tolerance (deg) for pairing (default=0.30°).
    max_distance : float, optional
        Maximum neighbor search radius (px) (default=75 px).
    phot_aperture_radius : float, optional
        Radius (px) for photometric aperture (default=5 px).
    r_in : float, optional
        Inner radius (px) for background annulus (default=7 px).
    r_out : float, optional
        Outer radius (px) for background annulus (default=10 px).
    SNR_threshold : float, optional
        Minimum signal-to-noise ratio to accept a photometric pair (default=5).
    save_plots : bool, optional
        If True, saves intermediate plots under `report_dir`.
    report_dir : str, optional
        Directory to store diagnostic PNGs (created if needed).

    Returns
    -------
    process_results : list of tuple
        For each of the four images (ref + 3 aligned), returns:
        `(image_data, sources, candidate_pairs, final_pairs, mode_distance, mode_angle)`.
    polar_results : list of dict
        Polarimetric parameters (q, u, P, theta, errors, fluxes) per star pair.
    final_paths : list of str
        Paths of the reference and the three aligned FITS files.
    """
    # Create report directory if requested
    if save_plots and report_dir:
        os.makedirs(report_dir, exist_ok=True)

    # Step 1: reference image
    ref_data = fits.getdata(ref_path)
    if save_plots and report_dir:
        save_plot(ref_data, 'lower', os.path.basename(ref_path), report_dir,
                  title="Reference Image " + ref_path, filename_suffix="_ref_img")

    # Step 1 & 2: alignment
    final_paths = [ref_path]
    for path in other_paths:
        img_data = fits.getdata(path)
        aligned, _ = align_images(ref_data, img_data)
        out_fits = path.replace('.fits', '-aligned.fits')
        save_fits_with_same_headers(path, aligned, out_fits)
        final_paths.append(out_fits)
        if save_plots and report_dir:
            save_plot(img_data, 'lower', os.path.basename(path), report_dir,
                      title="Original Image " + path, filename_suffix="_orig")
            save_plot(aligned, 'lower', os.path.basename(path), report_dir,
                      title="Aligned Image " + path, filename_suffix="_aligned")

    # Step 3: detect & pair
    process_results = []
    for path in final_paths:
        result = process_image(
            path,
            fwhm=fwhm,
            threshold_multiplier=threshold_multiplier,
            tol_distance=tol_distance,
            tol_angle=tol_angle,
            max_distance=max_distance
        )
        process_results.append(result)
        if save_plots and report_dir:
            img_data, sources, _, pairs, d_mode, a_mode = result
            draw_apertures(
                image_data=img_data,
                sources=sources,
                aperture_radius=phot_aperture_radius,
                annulus_radii=(r_in, r_out),
                original_name=os.path.basename(path),
                filename_suffix='_apertures',
                report_dir=report_dir
            )
            draw_pairs(
                img_data,
                sources,
                pairs,
                num_stars=len(sources),
                mode_distance=d_mode,
                mode_angle=a_mode,
                tol_distance=tol_distance,
                tol_angle=tol_angle,
                original_name=os.path.basename(path),
                filename_suffix="_pairs",
                report_dir=report_dir
            )

    # Step 4: photometry & polarimetry on reference
    _, sources, _, final_pairs, _, _ = process_results[0]
    polar_results, q_vals, u_vals = compute_polarimetry_for_pairs(
        final_paths,
        sources,
        final_pairs,
        aperture_radius=phot_aperture_radius,
        r_in=r_in,
        r_out=r_out,
        SNR_threshold=SNR_threshold,
        save_histogram=save_plots, 
        report_dir=report_dir, 
        hist_filename="snr_hist.png"
    )
    
    for rec in polar_results:
        print("#################################################")
        print(ref_path)
        print("FILTER" + filter)
        print("q" + str(rec['q']))
        print("u" + str(rec['u']))
        print("theta" + str(rec['theta']))
        print("P" + str(rec['P']))
        
        # 1) Restar offset instrumental
        q_corr = rec['q'] - INST_OFFSETS[filter]['q']
        u_corr = rec['u'] - INST_OFFSETS[filter]['u']
    
        # 2) Recalcular P y θ desde q_corr/u_corr
        P = math.hypot(q_corr, u_corr)
        θ = 0.5 * math.degrees(math.atan2(u_corr, q_corr))
    
        # resto instrumental
        #rec['q'] -= INST_OFFSETS[filter]['q']
        #rec['u'] -= INST_OFFSETS[filter]['u']
        # recalcular P, θ
        #P = math.hypot(rec['q'],rec['u'])
        #θ = 0.5*math.deg(math.atan2(rec['u'],rec['q']))
        # calibrar θ
        Δθ = θ - REF_ANGLES[filter]
        rec['theta'] = θ - Δθ
        rec['P'] = P
        print("q" + str(rec['q']))
        print("u" + str(rec['u']))
        print("theta" + str(rec['theta']))
        print("P" + str(rec['P']))
        print("#################################################")
    
    # Estimate interstellar (ISM) polarization from high-SNR q,u distributions
    ism_params = estimate_ism_from_histograms(q_vals, u_vals, n_components=2)
    
    if save_plots and report_dir:
        plot_polarization_errors(
            polar_results,
            report_dir,
            filename="polar_errors.png"
        )
        #plot_ism_histogram(q_vals, 'q', report_dir + '/q_ism_hist.png')
        #plot_ism_histogram(u_vals, 'u', report_dir + '/u_ism_hist.png')
        
        # ism_params = estimate_ism_from_histograms(...)
        q_hist_png = plot_ism_gmm_histogram(
            q_vals,
            ism_params['ism_estimation']['q_means'],
            ism_params['ism_estimation']['q_sigmas'],
            ism_params['ism_estimation']['q_weights'],
            label='q',
            outfile=report_dir + '/q_ism_hist.png',
            bins=30
        )
        u_hist_png = plot_ism_gmm_histogram(
            u_vals,
            ism_params['ism_estimation']['u_means'],
            ism_params['ism_estimation']['u_sigmas'],
            ism_params['ism_estimation']['u_weights'],
            label='u',
            outfile=report_dir + '/u_ism_hist.png',
            bins=30
        )

    return process_results, polar_results, final_paths, ism_params


def run_complete_polarimetric_pipeline(
    ref_path,
    other_paths,
    pol_angles,
    filter,
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
    synthetic_name="synthetic.fits",
    save_plots=False,
    report_dir=None
):
    """
    Execute the full polarimetric and astrometric analysis pipeline.

    This high-level function chains together:
      - compute_full_polarimetry: alignment, detection, pairing, photometry.
      - annotate_with_astrometry_net: synthetic image creation, WCS solution,
        pixel-to-world conversion, SIMBAD querying.
      - Optional saving of the synthetic image.

    Parameters
    ----------
    ref_path : str
        Path to the 0° FITS image (reference).
    other_paths : list of str
        Paths to the other three polarimetric FITS images.
    pol_angles : list of float
        Polarization angles corresponding to each input (0,22.5,45,67.5).
    fwhm, threshold_multiplier, tol_distance, tol_angle, max_distance : float
        Parameters for star detection and pairing.
    phot_aperture_radius, r_in, r_out, SNR_threshold : float
        Photometry parameters (aperture and background annulus).
    astrometry_api_key : str or None
        API key for Astrometry.Net. If None, skipping WCS solve.
    simbad_radius : astropy.units.Quantity
        Search radius for SIMBAD queries (default 0.01°).
    synthetic_name : str
        Output filename for the synthetic FITS (default "synthetic.fits").
    save_plots : bool
        Whether to save diagnostic plots to `report_dir`.
    report_dir : str or None
        Directory to collect all PNG outputs. Created if needed.

    Returns
    -------
    final_paths : list of str
        Paths to the four processed FITS files (reference + aligned).
    polar_results : list of dict
        Computed polarimetric metrics per star pair.
    wcs : astropy.wcs.WCS
        World-coordinate solution for the synthetic image.
    enriched : list of dict
        Polarimetric results augmented with 'ra', 'dec', and 'simbad_id' per pair.
    """
    
    report_directory_assets = report_dir + "/assets"
    
    # 1) Polarimetric processing
    process_results, polar_results, final_paths, ism_params = compute_full_polarimetry(
        ref_path,
        other_paths,
        pol_angles,
        filter,
        fwhm=fwhm,
        threshold_multiplier=threshold_multiplier,
        tol_distance=tol_distance,
        tol_angle=tol_angle,
        max_distance=max_distance,
        phot_aperture_radius=phot_aperture_radius,
        r_in=r_in,
        r_out=r_out,
        SNR_threshold=SNR_threshold,
        save_plots=save_plots,
        report_dir=report_directory_assets
    )

    # 2) Astrometry + SIMBAD annotation
    synthetic_path_name = report_dir + '/' + synthetic_name
    _, sources, _, final_pairs, _, _ = process_results[0]
    wcs, enriched = annotate_with_astrometry_net(
        ref_path,
        sources,
        final_pairs,
        polar_results,
        fwhm=fwhm,
        api_key=astrometry_api_key,
        simbad_radius=simbad_radius,
        synthetic_name=synthetic_path_name
    )

    # 3) Save synthetic image (optional)
    if save_plots and report_dir:
        syn_data = fits.getdata(synthetic_path_name)
        # 1. Imagen syntetica
        save_plot(syn_data, 'upper', os.path.basename(synthetic_path_name), report_directory_assets,
                  title="Synthetic Image", filename_suffix="_syn")
        # 2. Mapa de polarización
        plot_polarization_map(
            ref_path,
            enriched,
            wcs,
            report_directory_assets,
            filename="polarization_map.png"
        )
        # 3. Histograma de P
        plot_histogram_P(
            enriched,
            report_directory_assets,
            filename="histogram_P.png"
        )
        # 4. Histograma de θ
        plot_histogram_theta(
            enriched,
            report_directory_assets,
            filename="histogram_theta.png"
        )
        # 5. Diagrama Q–U
        plot_qu_diagram(
            enriched,
            report_directory_assets,
            filename="diagram_qu.png"
        )

    # 4) Put results in a JSON file
    elementos = [s for s in enriched]
    with open(report_dir + '/pipeline_results.json', 'w', encoding='utf-8') as f:
        json.dump(elementos, f, indent=4, ensure_ascii=False)

    with open(report_dir + '/ism_estimation.json', 'w', encoding='utf-8') as f:
        json.dump(ism_params, f, indent=4, ensure_ascii=False)

    generate_pdf_report(
        report_dir=report_directory_assets,
        output_pdf=report_dir + "/Polarimetric_Report.pdf",
        polar_results=polar_results,
        ism_params=ism_params,
        enriched_results=enriched
    )
    return final_paths, polar_results, wcs, enriched
