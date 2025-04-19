import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder


def detect_stars(image_data, fwhm=3.0, threshold_multiplier=5.0):
    """
    Detect stars in image_data using DAOStarFinder.

    Parameters
    ----------
    image_data          : 2D array
    fwhm                : float
    threshold_multiplier: float

    Returns
    -------
    sources : astropy Table
    """
    mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_multiplier * std)
    return daofind(image_data - median)


def process_image(image_path, fwhm=3.0, threshold_multiplier=5.0,
                  tol_distance=0.52, tol_angle=0.30, max_distance=50):
    """
    Detect stars and find filtered pairs in a single image.

    Parameters
    ----------
    image_path          : str
    fwhm                : float
    threshold_multiplier: float
    tol_distance        : float
    tol_angle           : float
    max_distance        : float

    Returns
    -------
    image_data      : 2D array
    sources         : Table
    candidate_pairs : list of tuples
    final_pairs     : list of tuples
    mode_distance   : float
    mode_angle      : float
    """
    image_data = fits.getdata(image_path)
    from .pairing import find_candidate_pairs, filter_pairs_by_mode
    sources = detect_stars(image_data, fwhm=fwhm, threshold_multiplier=threshold_multiplier)
    candidate_pairs = find_candidate_pairs(sources, max_distance=max_distance)
    final_pairs, mode_distance, mode_angle = filter_pairs_by_mode(
        candidate_pairs, tol_distance=tol_distance, tol_angle=tol_angle)
    return image_data, sources, candidate_pairs, final_pairs, mode_distance, mode_angle