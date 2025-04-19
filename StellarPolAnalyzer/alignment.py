from astropy.io import fits
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift


def align_images(reference_image, image_to_align):
    """
    Compute and apply the shift to align image_to_align to reference_image.

    Parameters
    ----------
    reference_image : 2D array
    image_to_align   : 2D array

    Returns
    -------
    aligned_image   : 2D array
    shift_estimation: tuple of (shift_y, shift_x)
    """
    shift_estimation, _, _ = phase_cross_correlation(reference_image, image_to_align, upsample_factor=10)
    aligned_image = shift(image_to_align, shift=shift_estimation)
    return aligned_image, shift_estimation


def save_fits_with_same_headers(original_filename, new_image, output_filename):
    """
    Save new_image to output_filename, preserving header from original_filename.

    Parameters
    ----------
    original_filename : str
    new_image         : 2D array
    output_filename   : str
    """
    with fits.open(original_filename) as hdul:
        header = hdul[0].header
    hdu = fits.PrimaryHDU(data=new_image, header=header)
    hdu.writeto(output_filename, overwrite=True)