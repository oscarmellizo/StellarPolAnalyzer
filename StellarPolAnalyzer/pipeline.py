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
from astropy.io import fits
from .detection import process_image
from .alignment import align_images, save_fits_with_same_headers
from .photometry import compute_polarimetry_for_pairs
from .astrometry import annotate_with_astrometry_net
from .visualization import draw_pairs, save_plot


def compute_full_polarimetry(
    ref_path,
    other_paths,
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
        save_plot(ref_data, os.path.basename(ref_path), report_dir,
                  title="Reference Image", filename_suffix="_refimg")

    # Step 1 & 2: alignment
    final_paths = [ref_path]
    for path in other_paths:
        img_data = fits.getdata(path)
        aligned, _ = align_images(ref_data, img_data)
        out_fits = path.replace('.fits', '-aligned.fits')
        save_fits_with_same_headers(path, aligned, out_fits)
        final_paths.append(out_fits)
        if save_plots and report_dir:
            save_plot(img_data, os.path.basename(path), report_dir,
                      title="Original Image", filename_suffix="_orig")
            save_plot(aligned, os.path.basename(path), report_dir,
                      title="Aligned Image", filename_suffix="_aligned")

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
    polar_results = compute_polarimetry_for_pairs(
        final_paths,
        sources,
        final_pairs,
        aperture_radius=phot_aperture_radius,
        r_in=r_in,
        r_out=r_out,
        SNR_threshold=SNR_threshold
    )

    return process_results, polar_results, final_paths


def run_complete_polarimetric_pipeline(
    ref_path,
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
    # 1) Polarimetric processing
    process_results, polar_results, final_paths = compute_full_polarimetry(
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
        SNR_threshold=SNR_threshold,
        save_plots=save_plots,
        report_dir=report_dir
    )

    # 2) Astrometry + SIMBAD annotation
    _, sources, _, final_pairs, _, _ = process_results[0]
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

    # 3) Save synthetic image (optional)
    if save_plots and report_dir:
        syn_data = fits.getdata(synthetic_name)
        save_plot(syn_data,
                  os.path.basename(synthetic_name),
                  report_dir,
                  title="Synthetic Image",
                  filename_suffix="_syn")

    return final_paths, polar_results, wcs, enriched
