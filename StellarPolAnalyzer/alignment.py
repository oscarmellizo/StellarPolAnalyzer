"""
alignment.py

Utilities for registering and aligning astronomical FITS images.

This module provides functions to:
  - Compute the pixel shift needed to align one image to a reference
    using phase cross‑correlation.
  - Apply the computed shift and save the resulting aligned image
    to a new FITS file while preserving the original header metadata.

Functions
---------
align_images(reference_image, image_to_align)
save_fits_with_same_headers(original_filename, new_image, output_filename)
"""

from astropy.io import fits
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift


def align_images(reference_image, image_to_align):
    """
    Compute the optimal translation to align one image to a reference, then apply it.

    Uses subpixel phase cross‑correlation to estimate the 2D shift
    (dy, dx) that best aligns `image_to_align` to `reference_image`.
    The shift is then applied via a translational shift on the image array.

    Parameters
    ----------
    reference_image : 2D array_like
        Pixel data of the reference image. Typically a NumPy array from FITS.
    image_to_align : 2D array_like
        Pixel data of the image to be aligned to the reference.

    Returns
    -------
    aligned_image : 2D ndarray
        The input `image_to_align`, shifted by the computed (dy, dx), with the same
        shape as the original.
    shift_estimation : tuple of float
        The estimated shift in pixels, as (shift_y, shift_x). Positive values
        indicate that `image_to_align` must be moved in the +Y or +X direction
        to best match the reference.

    Notes
    -----
    - The `upsample_factor=10` parameter in `phase_cross_correlation` provides
      subpixel accuracy by upsampling the cross‑correlation peak.
    - The returned `aligned_image` may contain edge artifacts where data is shifted
      in from outside the original frame.

    Example
    -------
    >>> from astropy.io import fits
    >>> ref = fits.getdata('ref.fits')
    >>> img = fits.getdata('to_align.fits')
    >>> aligned, shift_vec = align_images(ref, img)
    >>> print("Applied shift (y, x):", shift_vec)
    """
    shift_estimation, _, _ = phase_cross_correlation(
        reference_image, image_to_align,
        upsample_factor=10
    )
    aligned_image = shift(image_to_align, shift=shift_estimation)
    return aligned_image, shift_estimation


def save_fits_with_same_headers(original_filename, new_image, output_filename):
    """
    Save a shifted image array to a new FITS file, preserving the original header.

    Opens the FITS file at `original_filename`, reads its primary header,
    and writes `new_image` data into a new PrimaryHDU with that header.

    Parameters
    ----------
    original_filename : str
        Path to the existing FITS file whose header will be reused.
    new_image : 2D array_like
        The image data to write. Must be broadcastable to a 2D array.
    output_filename : str
        Desired path for the output FITS file. If the file exists, it will be overwritten.

    Returns
    -------
    None

    Side Effects
    ------------
    - Writes (or overwrites) the file at `output_filename` on disk.
    - Preserves the original FITS header, ensuring that metadata
      (WCS, instrument settings, etc.) remain intact.

    Example
    -------
    >>> aligned_data = ...  # obtained from align_images
    >>> save_fits_with_same_headers(
    ...     original_filename='to_align.fits',
    ...     new_image=aligned_data,
    ...     output_filename='to_align-aligned.fits'
    ... )
    >>> # Now 'to_align-aligned.fits' contains aligned_data with the same header.
    """
    with fits.open(original_filename) as hdul:
        header = hdul[0].header
    hdu = fits.PrimaryHDU(data=new_image, header=header)
    hdu.writeto(output_filename, overwrite=True)
