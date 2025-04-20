"""
detection.py

Module for detecting point sources in polarimetric FITS images and generating
star‐pair candidates for subsequent polarimetric analysis.

Provides:
  - detect_stars: run DAOStarFinder on image data to locate stars.
  - process_image: high‐level wrapper that loads a FITS file, detects stars,
                   finds all candidate star‐pairs within a radius, and filters
                   them by modal distance and angle.
"""

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder


def detect_stars(image_data, fwhm=3.0, threshold_multiplier=5.0):
    """
    Detect point sources in a 2D image using DAOStarFinder.

    This function computes robust background statistics with sigma‐clipping,
    subtracts the median background, and then runs DAOStarFinder to locate
    stars as local maxima convolved with a Gaussian kernel.

    Parameters
    ----------
    image_data : 2D array_like
        The pixel data from a FITS image (e.g., as returned by `fits.getdata`).
    fwhm : float, optional
        The full width at half maximum (in pixels) of the Gaussian kernel,
        corresponding roughly to the typical stellar PSF. Default is 3.0 px.
    threshold_multiplier : float, optional
        The detection threshold expressed as a multiple of the background
        standard deviation. DAOStarFinder will search for peaks with
        `threshold = threshold_multiplier * sigma`. Default is 5.0.

    Returns
    -------
    sources : astropy.table.Table
        Table of detected sources. Typical columns include:
          - xcentroid, ycentroid : subpixel star centroid coordinates
          - sharpness, roundness : shape diagnostics
          - flux                 : summed counts within the PSF kernel
        If no sources are found, returns an empty Table.

    Notes
    -----
    - Uses `sigma_clipped_stats` with sigma=3.0 to estimate background mean,
      median, and standard deviation.
    - Subtracts the median background before star detection.
    """
    # Estimate background statistics
    mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
    # Instantiate DAOStarFinder with the given FWHM and threshold
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_multiplier * std)
    # Return the catalog of sources found on the background‐subtracted image
    return daofind(image_data - median)


def process_image(image_path, fwhm=3.0, threshold_multiplier=5.0,
                  tol_distance=0.52, tol_angle=0.30, max_distance=50):
    """
    Load a FITS image, detect stars, find candidate pairs, and filter them.

    This convenience function ties together detection and pairing for a single
    FITS file. It returns all intermediate and final products needed for
    downstream polarimetric pipeline steps.

    Parameters
    ----------
    image_path : str
        Filesystem path to the FITS image to process.
    fwhm : float, optional
        FWHM parameter passed to `detect_stars`. Default is 3.0 px.
    threshold_multiplier : float, optional
        Threshold multiplier passed to `detect_stars`. Default is 5.0.
    tol_distance : float, optional
        Absolute tolerance (in pixels) around the modal pairing distance
        for filtering candidate pairs. Default is 0.52 px.
    tol_angle : float, optional
        Absolute tolerance (in degrees) around the modal pairing angle
        for filtering candidate pairs. Default is 0.30°.
    max_distance : float, optional
        Maximum pixel radius used when searching for candidate neighbors.
        Default is 50 px.

    Returns
    -------
    image_data : 2D ndarray
        The raw pixel array loaded from the FITS file.
    sources : astropy.table.Table
        Catalog of detected stars (see `detect_stars`).
    candidate_pairs : list of tuple
        All raw candidate pairs within `max_distance`. Each tuple is
        (i, j, distance, angle), where i and j index into `sources`.
    final_pairs : list of tuple
        Subset of `candidate_pairs` whose distance and angle lie within
        the tolerances around the field’s modal values.
    mode_distance : float
        The modal (most frequent) pairing distance, rounded to two decimals.
    mode_angle : float
        The modal pairing angle, rounded to two decimals.

    Example
    -------
    >>> img, src, cand, final, d0, a0 = process_image(
    ...     "field_0.fits",
    ...     fwhm=3.5,
    ...     threshold_multiplier=4.0,
    ...     tol_distance=1.0,
    ...     tol_angle=0.5,
    ...     max_distance=60
    ... )
    >>> print(f"Detected {len(src)} stars, {len(final)} pairs at {d0}±1.0 px, {a0}±0.5°")
    """
    # Load image data
    image_data = fits.getdata(image_path)

    # Lazy‐import pairing functions to avoid circular imports
    from .pairing import find_candidate_pairs, filter_pairs_by_mode

    # Detect stars
    sources = detect_stars(image_data, fwhm=fwhm, threshold_multiplier=threshold_multiplier)

    # Generate all neighbor pairs within max_distance
    candidate_pairs = find_candidate_pairs(sources, max_distance=max_distance)

    # Filter to those matching the modal distance/angle
    final_pairs, mode_distance, mode_angle = filter_pairs_by_mode(
        candidate_pairs,
        tol_distance=tol_distance,
        tol_angle=tol_angle
    )

    return image_data, sources, candidate_pairs, final_pairs, mode_distance, mode_angle
