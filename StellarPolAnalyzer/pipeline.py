from .detection import process_image
from .alignment import align_images, save_fits_with_same_headers
from .photometry import compute_polarimetry_for_pairs
from .astrometry import annotate_with_astrometry_net


def compute_full_polarimetry(ref_path, other_paths,
                             fwhm=3.0, threshold_multiplier=5.0,
                             tol_distance=0.52, tol_angle=0.30,
                             max_distance=75,
                             phot_aperture_radius=5, r_in=7, r_out=10, SNR_threshold=5):
    """
    Process 4 polarimetric images to get polarimetry results.
    """
    # align & process images
    ref_data = __import__('astropy.io.fits').io.fits.getdata(ref_path)
    final_paths=[ref_path]
    for p in other_paths:
        img = __import__('astropy.io.fits').io.fits.getdata(p)
        aligned,_ = align_images(ref_data,img)
        out = p.replace('.fits','-aligned.fits')
        save_fits_with_same_headers(p,aligned,out)
        final_paths.append(out)
    # run detection+pairing on each
    results=[process_image(fp,fwhm,threshold_multiplier,tol_distance,tol_angle,max_distance)
             for fp in final_paths]
    # extract reference
    _,sources,_,pairs,_,_=results[0]
    # photometry
    pol = compute_polarimetry_for_pairs(final_paths,sources,pairs,
                                         phot_aperture_radius,r_in,r_out,SNR_threshold)
    return results, pol, final_paths


def run_complete_polarimetric_pipeline(ref_path, other_paths, pol_angles,
                                        **kwargs):
    """
    Full pipeline: polarimetry + astrometry.
    Returns final_paths, polar_results, wcs, enriched.
    """
    results, polar, final = compute_full_polarimetry(ref_path,other_paths,**kwargs)
    _,sources,_,pairs,_,_=results[0]
    wcs,enr = annotate_with_astrometry_net(ref_path,sources,pairs,polar,
                                           fwhm=kwargs.get('fwhm',3.0),
                                           api_key=kwargs.get('astrometry_api_key'),
                                           simbad_radius=kwargs.get('simbad_radius',0.01*u.deg),
                                           synthetic_name=kwargs.get('synthetic_name','synthetic.fits'))
    return final, polar, wcs, enr