"""
photometry.py

Aperture photometry and polarimetric analysis for paired stars across multi‑angle images.

This module provides functionality to:
  - Perform background‑subtracted aperture photometry on each member of star pairs
    detected in polarimetric observations.
  - Compute normalized differences for each polarization angle.
  - Derive Stokes parameters q, u, the degree of polarization P, and the polarization
    angle θ, along with error estimates.

Key formulae:
  NDθ = (F_ext(θ) - F_ord(θ)) / (F_ext(θ) + F_ord(θ))
  q = (ND₀° - ND₄₅°) / 2 × 100%
  u = (ND₂₂.₅° - ND₆₇.₅°) / 2 × 100%
  P = √(q² + u²)
  θ = ½ × arctan2(u, q) (in degrees)
"""

import os
import numpy as np
from astropy.io import fits
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
import matplotlib.pyplot as plt


def compute_polarimetry_for_pairs(final_image_paths, sources, final_pairs,
                                  aperture_radius=5, r_in=7, r_out=10, SNR_threshold=5,
                                  save_histogram=False, report_dir=None, hist_filename="snr_hist.png"):
    """
    Perform aperture photometry on each star pair across four polarization images,
    compute Stokes parameters q, u, degree of polarization P, and polarization angle θ,
    and optionally generate a histogram of all SNR measurements.

    Parameters
    ----------
    final_image_paths : list of str
        Paths to the four aligned FITS images, ordered as:
        [image at 0°, image at 22.5°, image at 45°, image at 67.5°].
    sources : astropy.table.Table or list
        Detected sources from the reference image, each with 'xcentroid' and 'ycentroid'.
    final_pairs : list of tuple
        List of star pairs, each tuple: (i, j, distance_px, angle_deg).
    aperture_radius : float, optional
        Radius of the circular aperture in pixels (default: 5).
    r_in : float, optional
        Inner radius of the background annulus in pixels (default: 7).
    r_out : float, optional
        Outer radius of the background annulus in pixels (default: 10).
    SNR_threshold : float, optional
        Minimum signal‑to‑noise ratio to accept a measurement (default: 5).
    save_histogram : bool, optional
        If True, saves a histogram of all SNR values to `report_dir/hist_filename`.
    report_dir : str, optional
        Directory where to save the histogram (created if needed).
    hist_filename : str, optional
        Filename for the saved histogram (default: "snr_hist.png").

    Returns
    -------
    results : list of dict
        One dict per valid star pair, containing:
          - 'pair_index' : int
          - 'fluxes' : dict mapping each angle to:
                * 'ord_flux'  : float, background‑subtracted ordinary flux
                * 'ext_flux'  : float, background‑subtracted extraordinary flux
                * 'ord_bkg'   : float, estimated background for ordinary aperture
                * 'ext_bkg'   : float, estimated background for extraordinary aperture
                * 'ord_snr'   : float, SNR of ordinary flux
                * 'ext_snr'   : float, SNR of extraordinary flux
                * 'ND'        : float, normalized difference
                * 'error'     : float, error on ND
          - 'q' : float, Stokes q (%)
          - 'u' : float, Stokes u (%)
          - 'P' : float, degree of polarization (%)
          - 'theta' : float, polarization angle (°)
          - 'error' : float, averaged propagated error
    """
    # Prepare source coordinates
    coords = np.array([(s['xcentroid'], s['ycentroid']) for s in sources])

    # Assign ordinary vs extraordinary positions (ord = leftmost)
    pair_positions = []
    for (i, j, *_ ) in final_pairs:
        p1, p2 = coords[i], coords[j]
        pair_positions.append((p1, p2) if p1[0] < p2[0] else (p2, p1))

    angles = [0.0, 22.5, 45.0, 67.5]
    flux_tables = [{} for _ in pair_positions]

    # Collect all SNR values for histogram
    snr_values = []

    # Photometry per image
    for path, angle in zip(final_image_paths, angles):
        data = fits.getdata(path)

        # apertures and annuli
        ord_pos = np.array([pp[0] for pp in pair_positions])
        ext_pos = np.array([pp[1] for pp in pair_positions])
        ord_ap = CircularAperture(ord_pos, r=aperture_radius)
        ext_ap = CircularAperture(ext_pos, r=aperture_radius)
        ord_ann = CircularAnnulus(ord_pos, r_in=r_in, r_out=r_out)
        ext_ann = CircularAnnulus(ext_pos, r_in=r_in, r_out=r_out)

        # background estimates (per pair)
        ord_stats = ApertureStats(data, ord_ann)
        ext_stats = ApertureStats(data, ext_ann)
        ord_bkg = ord_stats.mean * ord_ap.area
        ext_bkg = ext_stats.mean * ext_ap.area

        # raw aperture sums
        ord_tab = aperture_photometry(data, ord_ap)
        ext_tab = aperture_photometry(data, ext_ap)
        ord_flux = ord_tab['aperture_sum'] - ord_bkg
        ext_flux = ext_tab['aperture_sum'] - ext_bkg

        # compute SNR
        ord_snr = ord_flux / np.sqrt(ord_flux + ord_ap.area * ord_stats.std**2)
        ext_snr = ext_flux / np.sqrt(ext_flux + ext_ap.area * ext_stats.std**2)

        # store valid measurements
        for idx in range(len(pair_positions)):
            if (ord_flux[idx] > 0 and ext_flux[idx] > 0 and
                ord_snr[idx] >= SNR_threshold and ext_snr[idx] >= SNR_threshold):

                ND = (ext_flux[idx] - ord_flux[idx]) / (ext_flux[idx] + ord_flux[idx])
                err = 0.5 / np.sqrt(ord_flux[idx] + ext_flux[idx])

                flux_tables[idx][angle] = {
                    'ord_flux' : float(ord_flux[idx]),
                    'ext_flux' : float(ext_flux[idx]),
                    'ord_bkg'  : float(ord_bkg[idx]),
                    'ext_bkg'  : float(ext_bkg[idx]),
                    'ord_snr'  : float(ord_snr[idx]),
                    'ext_snr'  : float(ext_snr[idx]),
                    'ND'       : float(ND),
                    'error'    : float(err)
                }
                # collect for histogram
                snr_values.append(ord_snr[idx])
                snr_values.append(ext_snr[idx])

    # Optionally save SNR histogram
    if save_histogram and report_dir:
        os.makedirs(report_dir, exist_ok=True)
        plt.figure(figsize=(6,4))
        plt.hist(snr_values, bins=30, color='gray', edgecolor='black')
        plt.xlabel('SNR')
        plt.ylabel('Count')
        plt.title('Distribution of Photometry SNR')
        hist_path = os.path.join(report_dir, hist_filename)
        plt.savefig(hist_path, bbox_inches='tight')
        plt.close()

    # Compute Stokes parameters
    results = []
    for idx, table in enumerate(flux_tables):
        if all(angle in table for angle in angles):
            ND0, ND45 = table[0.0]['ND'], table[45.0]['ND']
            ND22, ND67 = table[22.5]['ND'], table[67.5]['ND']
            err_avg = np.mean([table[a]['error'] for a in angles])

            q = ((ND0 - ND45) / 2.0) * 100.0
            u = ((ND22 - ND67) / 2.0) * 100.0
            P = np.hypot(q, u)
            theta = 0.5 * np.degrees(np.arctan2(u, q))

            results.append({
                'pair_index': idx,
                'fluxes': table,
                'q': q,
                'u': u,
                'P': P,
                'theta': theta,
                'error': err_avg
            })

    return results
