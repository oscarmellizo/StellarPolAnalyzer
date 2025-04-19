import numpy as np
from astropy.io import fits
from photutils import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats


def compute_polarimetry_for_pairs(final_image_paths, sources, final_pairs,
                                  aperture_radius=5, r_in=7, r_out=10, SNR_threshold=5):
    """
    Perform aperture photometry on each star pair across 4 polarization images and compute Stokes parameters.

    Returns
    -------
    results: list of dict with keys:
      'pair_index', 'fluxes', 'q','u','P','theta','error'
    """
    # Prepare source positions
    coords = np.array([(s['xcentroid'], s['ycentroid']) for s in sources])
    # Determine ordinary/extreme positions
    pair_positions = []
    for (i,j,_,_) in final_pairs:
        p1, p2 = coords[i], coords[j]
        pair_positions.append((p1,p2) if p1[0]<p2[0] else (p2,p1))
    angles = [0.0, 22.5, 45.0, 67.5]
    flux_tables = [{} for _ in pair_positions]
    # Photometry per image
    for path, ang in zip(final_image_paths, angles):
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
            if ord_flux[idx]>0 and ext_flux[idx]>0 and ord_snr[idx]>=SNR_threshold and ext_snr[idx]>=SNR_threshold:
                ND = (ext_flux[idx]-ord_flux[idx])/(ext_flux[idx]+ord_flux[idx])
                err = 0.5/np.sqrt(ord_flux[idx]+ext_flux[idx])
                flux_tables[idx][ang] = {
                    'ord_flux': float(ord_flux[idx]),
                    'ext_flux': float(ext_flux[idx]),
                    'ND': float(ND),
                    'error': float(err)
                }
    # Compute Stokes parameters
    results=[]
    for idx, table in enumerate(flux_tables):
        if all(a in table for a in angles):
            ND0, ND45 = table[0.0]['ND'], table[45.0]['ND']
            ND22,ND67 = table[22.5]['ND'], table[67.5]['ND']
            err_avg = np.mean([table[a]['error'] for a in angles])
            q = ((ND0-ND45)/2)*100
            u = ((ND22-ND67)/2)*100
            P = np.hypot(q,u)
            theta = 0.5*np.degrees(np.arctan2(u,q))
            results.append({
                'pair_index': idx,
                'fluxes': table,
                'q':q,'u':u,'P':P,'theta':theta,'error':err_avg
            })
    return results