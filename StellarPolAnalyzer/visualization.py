"""
visualization.py

Plotting Utilities for StellarPolAnalyzer 🚀

This module provides functions to visualize and save diagnostic figures at various
stages of the polarimetric pipeline:

1. **draw_pairs**: Annotate star pair detections on an image.
2. **save_plot**: Save any 2D array as a Z‑scale grayscale PNG.
3. **draw_apertures**: Overlay photometric apertures and background annuli.
4. **plot_polarization_errors**: Create a Q–U scatter plot with error bars.

These utilities can both display figures interactively and write them to disk
in a reproducible “report” folder structure.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.visualization import ZScaleInterval
import numpy as np
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

    This function visualizes:
      - All detected star centroids as small red markers.
      - Lime‐colored lines connecting each paired star.
      - Blue circles around the “ordinary” (left) beam components.
      - Red circles around the “extraordinary” (right) beam components.
      - A summary legend outside the image showing:
        • Total stars detected
        • Total pairs identified
        • Modal distance ± tolerance (pixels)
        • Modal angle ± tolerance (degrees)

    If `original_name` is non‐empty, the plot is saved to:
        `<report_dir>/<base_of_original_name><filename_suffix>.png`
    Otherwise, the figure is shown interactively.

    Parameters
    ----------
    image_data : 2D numpy.ndarray
        Pixel values of the FITS image to display.
    sources : astropy.table.Table or sequence
        Detected sources table; each entry must have 'xcentroid' and 'ycentroid'.
    pairs : list of tuple
        Each (i, j, distance, angle) indexes into `sources`.
    num_stars : int
        Number of detected sources.
    mode_distance : float
        Most frequent pairing distance (px).
    mode_angle : float
        Most frequent pairing angle (deg).
    tol_distance : float
        Allowed deviation from `mode_distance` (px).
    tol_angle : float
        Allowed deviation from `mode_angle` (deg).
    original_name : str
        Base FITS filename (e.g. 'field_22.fits') used to build output name.
    filename_suffix : str
        Suffix appended to the base filename before '.png'.
    report_dir : str
        Directory in which to save the PNG (will be created if needed).
    """
    # Ensure report directory exists
    os.makedirs(report_dir, exist_ok=True)

    # Build output filename
    base = os.path.splitext(original_name)[0]
    png_name = f"{base}{filename_suffix}.png"
    output_path = os.path.join(report_dir, png_name)

    # Z‑scale contrast limits
    interval = ZScaleInterval()
    z1, z2 = interval.get_limits(image_data)

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image_data, cmap='gray', origin='lower', vmin=z1, vmax=z2)
    ax.set_title(png_name)
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')

    coords = np.array([(s['xcentroid'], s['ycentroid']) for s in sources])

    # Draw star centroids
    for x, y in coords:
        ax.plot(x, y, 'ro', markersize=2)

    # Draw each pair
    for i, j, _, _ in pairs:
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        ax.plot([x1, x2], [y1, y2], color='lime', lw=0.5)

        # Identify left/right component
        left_idx, right_idx = (i, j) if x1 < x2 else (j, i)
        x_left, y_left = coords[left_idx]
        x_right, y_right = coords[right_idx]

        ax.add_patch(Circle((x_left, y_left), radius=5,
                            edgecolor='blue', facecolor='none', lw=0.5))
        ax.add_patch(Circle((x_right, y_right), radius=5,
                            edgecolor='red', facecolor='none', lw=0.5))

    # Summary legend
    plt.subplots_adjust(right=0.7)
    summary = (
        f"Stars: {num_stars}\n"
        f"Pairs: {len(pairs)}\n"
        f"Distance: {mode_distance} ± {tol_distance}\n"
        f"Angle: {mode_angle} ± {tol_angle}"
    )
    plt.figtext(0.75, 0.5, summary,
                bbox=dict(facecolor='white', alpha=0.7), fontsize=10)

    # Save or show
    if original_name:
        fig.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def save_plot(
    data,
    original_name,
    report_dir,
    title=None,
    filename_suffix=None
):
    """
    Render and save a generic image plot (e.g., original or aligned frame).

    Uses Z‑scale contrast for astrophotography‐style display.

    Parameters
    ----------
    data : 2D numpy.ndarray
        Pixel array to visualize.
    original_name : str
        Base FITS filename (e.g. 'field_22.fits') for output naming.
    report_dir : str
        Directory in which to save the PNG (created if it does not exist).
    title : str, optional
        Custom plot title; defaults to `<base><filename_suffix>.png`.
    filename_suffix : str, optional
        Suffix appended to `<base>` before the `.png` extension.
    """
    os.makedirs(report_dir, exist_ok=True)
    base = os.path.splitext(original_name)[0]
    suffix = filename_suffix or ''
    png_name = f"{base}{suffix}.png"
    output_path = os.path.join(report_dir, png_name)
    plot_title = title or png_name

    # Z‑scale contrast
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)

    # Plot and save
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
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
        Detected sources, each with 'xcentroid' and 'ycentroid'.
    aperture_radius : float
        Radius (px) of the measurement aperture.
    annulus_radii : tuple (r_in, r_out), optional
        Inner and outer radii (px) of the background annulus.
    original_name : str, optional
        Base FITS filename for PNG naming; if None, displays interactively.
    filename_suffix : str, optional
        Suffix to append to the base filename before `.png`.
    report_dir : str, optional
        Directory to save the figure; created if it does not exist.
    """
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    coords = [(s['xcentroid'], s['ycentroid']) for s in sources]
    apertures = CircularAperture(coords, r=aperture_radius)

    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 8))
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(image_data)
    ax.imshow(image_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title(f"Apertures (r={aperture_radius}px)")
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')

    # Draw apertures
    apertures.plot(ax=ax, edgecolor='yellow', lw=1.0, alpha=0.7)

    # Draw background annuli
    if annulus_radii:
        r_in, r_out = annulus_radii
        annuli = CircularAnnulus(coords, r_in=r_in, r_out=r_out)
        annuli.plot(ax=ax, edgecolor='cyan', lw=0.8, alpha=0.5)

    # Save or show
    if original_name and report_dir:
        base = os.path.splitext(original_name)[0]
        png_name = f"{base}{filename_suffix}.png"
        out_path = os.path.join(report_dir, png_name)
        fig.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def plot_polarization_errors(
    polar_results,
    report_dir,
    filename="polar_errors.png"
):
    """
    Plot q vs u with error bars and save the figure.

    This diagram highlights the scatter of polarization measurements and
    their uncertainties for each star pair.

    Parameters
    ----------
    polar_results : list of dict
        Output from `compute_polarimetry_for_pairs` or `compute_full_polarimetry`,
        each dict must include:
          - 'q'    : Stokes q (%)
          - 'u'    : Stokes u (%)
          - 'error': symmetric error on both q and u (%)
    report_dir : str
        Directory in which to save the PNG; created if it does not exist.
    filename : str, optional
        Name of the output PNG file (default: 'polar_errors.png').
    """
    os.makedirs(report_dir, exist_ok=True)

    # Extract values
    q_vals   = [entry['q']     for entry in polar_results]
    u_vals   = [entry['u']     for entry in polar_results]
    err_vals = [entry['error'] for entry in polar_results]

    # Plot
    plt.figure(figsize=(6, 6))
    plt.errorbar(q_vals, u_vals,
                 xerr=err_vals, yerr=err_vals,
                 fmt='o', ecolor='gray', capsize=3, alpha=0.8)
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.xlabel('q (%)')
    plt.ylabel('u (%)')
    plt.title('Polarization Q–U with Errors')
    plt.grid(True, ls='--', alpha=0.5)

    # Save figure
    out_path = os.path.join(report_dir, filename)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
