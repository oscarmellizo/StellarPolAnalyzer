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

import numpy as np
from astropy.io import fits
from photutils import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats


def compute_polarimetry_for_pairs(final_image_paths, sources, final_pairs,
                                  aperture_radius=5, r_in=7, r_out=10, SNR_threshold=5):
    """
    Perform aperture photometry on each star pair across four polarization images
    and compute Stokes parameters q, u, degree of polarization P, and polarization angle θ.

    Parameters
    ----------
    final_image_paths : list of str
        Paths to the four aligned FITS images, ordered as:
        [image at 0°, image at 22.5°, image at 45°, image at 67.5°].

    sources : astropy.table.Table or list of dict-like
        Detected sources from the reference image, each with fields:
        'xcentroid' and 'ycentroid'.

    final_pairs : list of tuple
        List of star pairs, each tuple:
        (i, j, distance_px, angle_deg)
        where i, j are indices into `sources`.

    aperture_radius : float, optional
        Radius of the circular aperture in pixels for flux measurement (default: 5).

    r_in : float, optional
        Inner radius of the background annulus in pixels (default: 7).

    r_out : float, optional
        Outer radius of the background annulus in pixels (default: 10).

    SNR_threshold : float, optional
        Minimum signal‑to‑noise ratio for accepting a measurement (default: 5).

    Returns
    -------
    results : list of dict
        One entry per star pair that has valid measurements at all four angles.
        Each dict contains:
          - 'pair_index' : int
              Index of the pair in `final_pairs`.
          - 'fluxes' : dict of dict
              Keys are polarization angles [0.0, 22.5, 45.0, 67.5], each mapping to:
                * 'ord_flux' : float
                    Background‑subtracted ordinary‑beam flux.
                * 'ext_flux' : float
                    Background‑subtracted extraordinary‑beam flux.
                * 'ND' : float
                    Normalized difference NDθ.
                * 'error' : float
                    Estimated error on NDθ.
          - 'q' : float
              Stokes q parameter (%).
          - 'u' : float
              Stokes u parameter (%).
          - 'P' : float
              Degree of polarization (%) = √(q² + u²).
          - 'theta' : float
              Polarization angle (degrees) = ½·arctan2(u, q).
          - 'error' : float
              Propagated error, averaged over the four angles.

    Notes
    -----
    1. Aperture photometry is carried out using `CircularAperture` for stellar flux
       and `CircularAnnulus` for local background estimation.
    2. Background flux is estimated as mean(inner_annulus) × aperture_area.
    3. SNR is computed as:
         SNR = flux / √(flux + aperture_area × σ_bkg²),
       where σ_bkg is the annular background standard deviation.
    4. Only flux values > 0 and with SNR ≥ SNR_threshold are kept.
    5. Polarimetric parameters follow standard dual‑beam formulae.

    Example
    -------
    >>> from StellarPolAnalyzer.photometry import compute_polarimetry_for_pairs
    >>> img_paths = ["field_0-aligned.fits",
    ...              "field_22.5-aligned.fits",
    ...              "field_45-aligned.fits",
    ...              "field_67.5-aligned.fits"]
    >>> results = compute_polarimetry_for_pairs(
    ...     img_paths,
    ...     sources_table,
    ...     final_pairs,
    ...     aperture_radius=5,
    ...     r_in=8, r_out=12,
    ...     SNR_threshold=10
    ... )
    >>> for entry in results:
    ...     print(f"Pair {entry['pair_index']}: P={entry['P']:.2f}%, θ={entry['theta']:.1f}°")
    """
    # Prepare source coordinates
    coords = np.array([(s['xcentroid'], s['ycentroid']) for s in sources])

    # Assign ordinary vs. extraordinary positions (ord = leftmost)
    pair_positions = []
    for (i, j, *_ ) in final_pairs:
        p1, p2 = coords[i], coords[j]
        pair_positions.append((p1, p2) if p1[0] < p2[0] else (p2, p1))

    angles = [0.0, 22.5, 45.0, 67.5]
    flux_tables = [{} for _ in pair_positions]

    # Perform photometry on each image/angle
    for path, angle in zip(final_image_paths, angles):
        data = fits.getdata(path)
        ord_pos = np.array([pp[0] for pp in pair_positions])
        ext_pos = np.array([pp[1] for pp in pair_positions])

        ord_ap = CircularAperture(ord_pos, r=aperture_radius)
        ext_ap = CircularAperture(ext_pos, r=aperture_radius)
        ord_ann = CircularAnnulus(ord_pos, r_in=r_in, r_out=r_out)
        ext_ann = CircularAnnulus(ext_pos, r_in=r_in, r_out=r_out)

        ord_bkg = ApertureStats(data, ord_ann).mean * ord_ap.area
        ext_bkg = ApertureStats(data, ext_ann).mean * ext_ap.area

        ord_tab = aperture_photometry(data, ord_ap)
        ext_tab = aperture_photometry(data, ext_ap)

        ord_flux = ord_tab['aperture_sum'] - ord_bkg
        ext_flux = ext_tab['aperture_sum'] - ext_bkg

        ord_snr = ord_flux / np.sqrt(ord_flux + ord_ap.area * ApertureStats(data, ord_ann).std**2)
        ext_snr = ext_flux / np.sqrt(ext_flux + ext_ap.area * ApertureStats(data, ext_ann).std**2)

        for idx in range(len(pair_positions)):
            if (ord_flux[idx] > 0 and ext_flux[idx] > 0 and
                ord_snr[idx] >= SNR_threshold and ext_snr[idx] >= SNR_threshold):

                ND = (ext_flux[idx] - ord_flux[idx]) / (ext_flux[idx] + ord_flux[idx])
                err = 0.5 / np.sqrt(ord_flux[idx] + ext_flux[idx])

                flux_tables[idx][angle] = {
                    'ord_flux': float(ord_flux[idx]),
                    'ext_flux': float(ext_flux[idx]),
                    'ND': float(ND),
                    'error': float(err)
                }

    # Compute Stokes parameters for pairs with measurements at all angles
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
                'fluxes': table,
                'q': q,
                'u': u,
                'P': P,
                'theta': theta,
                'error': err_avg
            })

    return results
