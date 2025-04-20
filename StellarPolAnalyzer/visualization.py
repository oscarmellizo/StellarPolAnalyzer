"""
visualization.py

This module provides plotting utilities for StellarPolAnalyzer, focused on visualizing
detected star pairs and alignment steps in polarimetric image processing.

Functions:
-----------
- draw_pairs: Display or save an annotated image showing star centroids,
  pair connections, and a summary legend.
- save_plot: Render and save a generic image (e.g., original or aligned frame)
  with customizable title and filename suffix.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.visualization import ZScaleInterval
import numpy as np


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
      - All detected star centroids as small red circles.
      - Connecting lines (in lime) between each paired star.
      - Highlighted "ordinary" (left) components in blue and "extraordinary"
        (right) components in red for each pair.
      - A summary legend outside the main image area with:
        * Total number of stars detected.
        * Total number of pairs identified.
        * Modal distance ± tolerance.
        * Modal angle ± tolerance.

    If `original_name` is provided, the figure is saved in PNG format under
    `report_dir` with filename `<base><filename_suffix>.png`. Otherwise, the
    plot is shown interactively.

    Parameters
    ----------
    image_data : 2D numpy.ndarray
        Pixel array of the FITS image to display.
    sources : astropy.table.Table or sequence of dict-like
        Detected sources, each containing 'xcentroid' and 'ycentroid' for star positions.
    pairs : list of tuple
        List of (i, j, distance, angle) tuples, where i and j index into `sources`.
    num_stars : int
        Number of detected stars (len(sources)).
    mode_distance : float
        The most frequent pairing distance (pixels).
    mode_angle : float
        The most frequent pairing angle (degrees).
    tol_distance : float
        Allowed deviation (pixels) from `mode_distance` for filtered pairs.
    tol_angle : float
        Allowed deviation (degrees) from `mode_angle` for filtered pairs.
    original_name : str
        Base FITS filename (e.g., 'field_22.fits') used to derive output PNG name.
    filename_suffix : str
        Suffix appended to the base filename (before '.png') for the saved figure.
    report_dir : str
        Directory path where the PNG will be saved; created if it does not exist.

    Behavior
    --------
    1. Creates `report_dir` if missing.
    2. Computes display limits via ZScaleInterval for appropriate contrast.
    3. Plots centroids, pair lines, and colored circles.
    4. Adds a summary text block outside the image on the right.
    5. Saves to `<report_dir>/<base><filename_suffix>.png` or displays interactively.
    """
    # Ensure report directory exists
    os.makedirs(report_dir, exist_ok=True)

    # Build output filename and title
    base = os.path.splitext(original_name)[0]
    png_name = f"{base}{filename_suffix}.png"
    output_path = os.path.join(report_dir, png_name)
    title = png_name

    # Compute display limits
    interval = ZScaleInterval()
    z1, z2 = interval.get_limits(image_data)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image_data, cmap='gray', origin='lower', vmin=z1, vmax=z2)
    ax.set_title(title)
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')

    # Extract coordinates
    coords = np.array([(s['xcentroid'], s['ycentroid']) for s in sources])

    # Plot all star centroids
    for x, y in coords:
        ax.plot(x, y, 'ro', markersize=2)

    # Plot each pair with line and colored circles
    for (i, j, _, _) in pairs:
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        ax.plot([x1, x2], [y1, y2], color='lime', lw=0.5)

        # Determine left vs right component
        left_idx, right_idx = (i, j) if x1 < x2 else (j, i)
        x_left, y_left = coords[left_idx]
        x_right, y_right = coords[right_idx]

        # Add colored circles
        ax.add_patch(Circle((x_left, y_left), radius=5,
                            edgecolor='blue', facecolor='none', lw=0.5))
        ax.add_patch(Circle((x_right, y_right), radius=5,
                            edgecolor='red', facecolor='none', lw=0.5))

    # Place summary legend outside plot
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


def save_plot(data, original_name, report_dir, title=None, filename_suffix=None):
    """
    Render and save a generic image plot with optional custom title.

    This function is suitable for saving intermediate steps such as the
    original or aligned frames in the pipeline. It uses Z-scale display
    limits for consistent visualization.

    Parameters
    ----------
    data : 2D numpy.ndarray
        Pixel array to visualize.
    original_name : str
        Base FITS filename (e.g., 'field_22.fits') used to derive the PNG name.
    report_dir : str
        Directory path where the PNG file will be saved; created if needed.
    title : str, optional
        Custom figure title; defaults to the generated filename.
    filename_suffix : str, optional
        Suffix appended to the base filename before '.png'.

    Behavior
    --------
    1. Ensures `report_dir` exists.
    2. Computes Z-scale display limits via ZScaleInterval.
    3. Sets up a 6×6" figure showing `data` in grayscale.
    4. Uses `title` or filename as the plot title.
    5. Saves the plot to `<report_dir>/<base><filename_suffix>.png`.
    """
    # Ensure report directory exists
    os.makedirs(report_dir, exist_ok=True)

    # Construct output filename
    base = os.path.splitext(original_name)[0]
    suffix = filename_suffix or ''
    png_name = f"{base}{suffix}.png"
    output_path = os.path.join(report_dir, png_name)
    plot_title = title or png_name

    # Determine display limits
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
