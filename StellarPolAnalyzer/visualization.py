"""
visualization.py

Plotting Utilities for StellarPolAnalyzer

This module provides functions to visualize and save diagnostic figures at various
stages of the polarimetric pipeline:

1. **draw_pairs**: Annotate star pair detections on an image.
2. **save_plot**: Save any 2D array as a Z‑scale grayscale PNG.
3. **draw_apertures**: Overlay photometric apertures and background annuli.
4. **plot_polarization_errors**: Create a Q–U scatter plot with error bars.
5. **plot_polarization_map**: Draw polarization vectors over a FITS background.
6. **plot_histogram_P**: Histogram of polarization degree P.
7. **plot_histogram_theta**: Histogram of polarization angle θ.
8. **plot_qu_diagram**: Q–U scatter plot (with optional box‑and‑whisker style).

These utilities can both display figures interactively and write them to disk
in a reproducible “report” folder structure.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.visualization import ZScaleInterval
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import CircularAperture, CircularAnnulus


def draw_pairs(
    image_data,
    sources,
    pairs,
    num_stars,
    mode_distance,
    mode_angle,
    tol_distance,
    tol_angle,
    original_name,
    filename_suffix,
    report_dir
):
    """
    Display and optionally save a plot of detected stellar pairs with annotations.

    Visual elements:
      - Red markers for each detected star centroid.
      - Lime lines connecting each paired star.
      - Blue circles around the 'ordinary' (left) component of each pair.
      - Red circles around the 'extraordinary' (right) component of each pair.
      - A legend box showing:
        • Total stars detected
        • Total pairs identified
        • Modal distance ± tolerance (pixels)
        • Modal angle ± tolerance (degrees)

    If `original_name` is non-empty, saves the figure to:
        <report_dir>/<base_of_original_name><filename_suffix>.png
    Otherwise, shows the plot interactively.

    Parameters
    ----------
    image_data : 2D numpy.ndarray
        Pixel array of the FITS image to display.
    sources : astropy.table.Table or sequence
        Table of detected sources; each must have 'xcentroid' and 'ycentroid'.
    pairs : list of tuple
        Each tuple (i, j, dist, angle) indexes into `sources`.
    num_stars : int
        Number of detected stars.
    mode_distance : float
        Modal pairing distance in pixels.
    mode_angle : float
        Modal pairing angle in degrees.
    tol_distance : float
        Distance tolerance in pixels.
    tol_angle : float
        Angle tolerance in degrees.
    original_name : str
        Base FITS filename (e.g. 'field_22.fits'), used for naming.
    filename_suffix : str
        Suffix appended to the base filename before '.png'.
    report_dir : str
        Directory to save PNG (created if it does not exist).

    Returns
    -------
    None
    """
    os.makedirs(report_dir, exist_ok=True)

    base = os.path.splitext(original_name)[0]
    png_name = f"{base}{filename_suffix}.png"
    output_path = os.path.join(report_dir, png_name)

    interval = ZScaleInterval()
    z1, z2 = interval.get_limits(image_data)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image_data, cmap='gray', origin='lower', vmin=z1, vmax=z2)
    ax.set_title(png_name)
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')

    coords = np.array([(s['xcentroid'], s['ycentroid']) for s in sources])

    for x, y in coords:
        ax.plot(x, y, 'ro', markersize=2)

    for i, j, _, _ in pairs:
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        ax.plot([x1, x2], [y1, y2], color='lime', lw=0.5)
        left_idx, right_idx = (i, j) if x1 < x2 else (j, i)
        x_left, y_left = coords[left_idx]
        x_right, y_right = coords[right_idx]
        ax.add_patch(Circle((x_left, y_left), radius=5, edgecolor='blue', facecolor='none', lw=0.5))
        ax.add_patch(Circle((x_right, y_right), radius=5, edgecolor='red', facecolor='none', lw=0.5))

    plt.subplots_adjust(right=0.7)
    summary = (
        f"Stars: {num_stars}\n"
        f"Pairs: {len(pairs)}\n"
        f"Distance: {mode_distance} ± {tol_distance}\n"
        f"Angle: {mode_angle} ± {tol_angle}"
    )
    plt.figtext(0.75, 0.5, summary, bbox=dict(facecolor='white', alpha=0.7), fontsize=10)

    if original_name:
        fig.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def save_plot(
    data,
    origin,
    original_name,
    report_dir,
    title=None,
    filename_suffix=None
):
    """
    Render and save a generic image plot (e.g., original or aligned frame).

    Uses Z-scale contrast interval for astrophotography‑style display.

    Parameters
    ----------
    data : 2D numpy.ndarray
        Pixel array to visualize.
    original_name : str
        Base FITS filename for naming (e.g. 'field_22.fits').
    report_dir : str
        Directory to save the PNG (created if needed).
    title : str, optional
        Custom plot title; defaults to the generated filename.
    filename_suffix : str, optional
        Suffix to append to the base filename before '.png'.

    Returns
    -------
    None
    """
    os.makedirs(report_dir, exist_ok=True)
    base = os.path.splitext(original_name)[0]
    suffix = filename_suffix or ''
    png_name = f"{base}{suffix}.png"
    output_path = os.path.join(report_dir, png_name)
    plot_title = title or png_name

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(data, cmap='gray', origin=origin, vmin=vmin, vmax=vmax)
    ax.set_title(plot_title)
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def draw_apertures(
    image_data,
    sources,
    aperture_radius,
    annulus_radii=None,
    original_name=None,
    filename_suffix='_apertures',
    report_dir=None
):
    """
    Overlay photometric apertures and optional background annuli on each star.

    Parameters
    ----------
    image_data : 2D numpy.ndarray
        Pixel values of the FITS image.
    sources : astropy.table.Table or sequence
        Detected sources with 'xcentroid' and 'ycentroid'.
    aperture_radius : float
        Radius of the measurement aperture (pixels).
    annulus_radii : tuple (r_in, r_out), optional
        Inner/outer radii of the background annulus (pixels).
    original_name : str, optional
        Base FITS filename; if provided, used for naming saved PNG.
    filename_suffix : str, optional
        Suffix appended to base filename before '.png'.
    report_dir : str, optional
        Directory to save the PNG (created if needed).

    Returns
    -------
    None
    """
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    coords = [(s['xcentroid'], s['ycentroid']) for s in sources]
    apertures = CircularAperture(coords, r=aperture_radius)

    fig, ax = plt.subplots(figsize=(8, 8))
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(image_data)
    ax.imshow(image_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title(f"Apertures {original_name} (r={aperture_radius}px)")
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')

    apertures.plot(ax=ax, edgecolor='yellow', lw=1.0, alpha=0.7)

    if annulus_radii:
        r_in, r_out = annulus_radii
        annuli = CircularAnnulus(coords, r_in=r_in, r_out=r_out)
        annuli.plot(ax=ax, edgecolor='cyan', lw=0.8, alpha=0.5)

    if original_name and report_dir:
        base = os.path.splitext(original_name)[0]
        png_name = f"{base}{filename_suffix}.png"
        out_path = os.path.join(report_dir, png_name)
        fig.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def plot_polarization_errors(polar_results, report_dir, filename="polar_errors.png"):
    """
    Plot q vs. u with error bars and save the figure.

    This diagram highlights the scatter of polarization measurements and
    their uncertainties for each star pair.

    Parameters
    ----------
    polar_results : list of dict
        Output from compute_polarimetry_for_pairs or compute_full_polarimetry;
        each dict must include 'q', 'u', and 'error'.
    report_dir : str
        Directory in which to save the PNG (created if needed).
    filename : str, optional
        Name of the output PNG (default: 'polar_errors.png').

    Returns
    -------
    None
    """
    os.makedirs(report_dir, exist_ok=True)

    q_vals = [entry['q'] for entry in polar_results]
    u_vals = [entry['u'] for entry in polar_results]
    err_vals = [entry['error'] for entry in polar_results]

    plt.figure(figsize=(6, 6))
    plt.errorbar(q_vals, u_vals, xerr=err_vals, yerr=err_vals,
                 fmt='o', ecolor='gray', capsize=3, alpha=0.8)
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.xlabel('q (%)')
    plt.ylabel('u (%)')
    plt.title('Polarization Q–U with Errors')
    plt.grid(True, ls='--', alpha=0.5)

    out_path = os.path.join(report_dir, filename)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def plot_polarization_map(
    ref_fits_path,
    polar_results,
    wcs,
    report_dir,
    filename="polarization_map.png"
):
    """
    Plot a polarization vector map over the reference FITS image.

    Overlays quiver arrows whose length scales with P and orientation with θ
    in RA/DEC coordinates.

    Parameters
    ----------
    ref_fits_path : str
        Path to the reference FITS (e.g., 0° image) for background.
    polar_results : list of dict
        Each dict must contain 'ra', 'dec', 'P', and 'theta'.
    wcs : astropy.wcs.WCS
        WCS solution for converting sky to pixel.
    report_dir : str
        Directory to save the PNG (created if needed).
    filename : str, optional
        Output PNG filename.

    Returns
    -------
    None
    """
    os.makedirs(report_dir, exist_ok=True)
    from astropy.io import fits

    data = fits.getdata(ref_fits_path)
    vmin, vmax = ZScaleInterval().get_limits(data)

    ras = np.array([e['ra'] for e in polar_results])
    decs = np.array([e['dec'] for e in polar_results])
    Ps = np.array([e['P'] for e in polar_results])
    thetas = np.array([e['theta'] for e in polar_results])

    sky = SkyCoord(ras * u.deg, decs * u.deg, frame='icrs')
    xs, ys = wcs.world_to_pixel(sky)
    us = Ps * np.cos(np.deg2rad(thetas))
    vs = Ps * np.sin(np.deg2rad(thetas))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=wcs)
    ax.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax.quiver(xs, ys, us, vs, Ps,
              transform=ax.get_transform('pixel'),
              cmap='viridis', scale=50, width=0.002, pivot='mid')
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    ax.set_title('Polarization Map')
    plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', label='P (%)')

    out = os.path.join(report_dir, filename)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def plot_histogram_P(polar_results, report_dir, filename="histogram_P.png", bins=20):
    """
    Plot a histogram of the degree of polarization P across all pairs.

    Parameters
    ----------
    polar_results : list of dict
        Each dict must contain 'P'.
    report_dir : str
        Directory to save the PNG (created if needed).
    filename : str, optional
        Output PNG filename.
    bins : int, optional
        Number of histogram bins (default: 20).

    Returns
    -------
    None
    """
    os.makedirs(report_dir, exist_ok=True)
    Ps = [e['P'] for e in polar_results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(Ps, bins=bins, edgecolor='black', alpha=0.8)
    ax.set_xlabel('P (%)')
    ax.set_ylabel('Count')
    ax.set_title('Histogram of Polarization Degree')
    out = os.path.join(report_dir, filename)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def plot_histogram_theta(polar_results, report_dir, filename="histogram_theta.png", bins=20):
    """
    Plot a histogram of the polarization angle θ across all pairs.

    Parameters
    ----------
    polar_results : list of dict
        Each dict must contain 'theta'.
    report_dir : str
        Directory to save the PNG (created if needed).
    filename : str, optional
        Output PNG filename.
    bins : int, optional
        Number of histogram bins (default: 20).

    Returns
    -------
    None
    """
    os.makedirs(report_dir, exist_ok=True)
    thetas = [e['theta'] for e in polar_results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(thetas, bins=bins, edgecolor='black', alpha=0.8)
    ax.set_xlabel('θ (deg)')
    ax.set_ylabel('Count')
    ax.set_title('Histogram of Polarization Angle')
    out = os.path.join(report_dir, filename)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def plot_qu_diagram(polar_results, report_dir, filename="qu_diagram.png"):
    """
    Plot a Q–U scatter diagram, optionally with error bars.

    Parameters
    ----------
    polar_results : list of dict
        Each dict must include 'q', 'u', and 'error'.
    report_dir : str
        Directory to save the PNG (created if needed).
    filename : str, optional
        Output PNG filename.

    Returns
    -------
    None
    """
    os.makedirs(report_dir, exist_ok=True)
    qs = [e['q'] for e in polar_results]
    us = [e['u'] for e in polar_results]
    errs = [e['error'] for e in polar_results]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(qs, us, c='darkgreen', s=30, alpha=0.8, label='pairs')
    ax.errorbar(qs, us, xerr=errs, yerr=errs, fmt='none', ecolor='gray', alpha=0.5)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('q (%)')
    ax.set_ylabel('u (%)')
    ax.set_title('Q–U Diagram')
    ax.legend()

    out = os.path.join(report_dir, filename)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
