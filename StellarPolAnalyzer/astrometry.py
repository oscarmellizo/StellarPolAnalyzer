import numpy as np
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from astropy.wcs import WCS
from astroquery.astrometry_net import AstrometryNet
from astroquery.simbad import Simbad
from astropy import coordinates as coord
import astropy.units as u


def annotate_with_astrometry_net(ref_path, sources, final_pairs, polarimetry_results,
                                 fwhm=3.0, api_key=None, simbad_radius=0.01*u.deg,
                                 synthetic_name="synthetic.fits"):
    """
    Generate synthetic FITS, solve WCS via Astrometry.net, then SIMBAD cross-match.

    Returns
    -------
    wcs    : WCS object
    enriched: list of dict (adds 'ra','dec','simbad_id')
    """
    hdr = fits.getheader(ref_path)
    ny,nx = hdr['NAXIS2'], hdr['NAXIS1']
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    syn = np.zeros((ny,nx))
    positions=[]
    for entry in polarimetry_results:
        idx = entry['pair_index']
        i,j,_,_ = final_pairs[idx]
        x,y = sources[i]['xcentroid'], sources[i]['ycentroid']
        positions.append((x,y))
        amp = entry['fluxes'][0.0]['ord_flux']+entry['fluxes'][0.0]['ext_flux']
        g = Gaussian2D(amplitude=amp, x_mean=x, y_mean=y, x_stddev=sigma, y_stddev=sigma)
        yy,xx = np.mgrid[0:ny,0:nx]
        syn += g(xx,yy)
    syn_hdr = fits.Header({k:hdr[k] for k in ('RA','DEC','OBJECT') if k in hdr})
    fits.PrimaryHDU(syn,header=syn_hdr).writeto(synthetic_name,overwrite=True)
    ast = AstrometryNet(api_key=api_key)
    sol = ast.solve_from_image(synthetic_name)
    wcs_hdr = fits.Header(sol)
    wcs = WCS(wcs_hdr)
    pix = np.array(positions)
    world = wcs.all_pix2world(pix,1)
    Simbad.reset_votable_fields()
    Simbad.add_votable_fields('otype')
    enriched=[]
    for entry,(ra,dec) in zip(polarimetry_results,world):
        c = coord.SkyCoord(ra=ra*u.deg,dec=dec*u.deg)
        res=Simbad.query_region(c,radius=simbad_radius)
        obj = res['MAIN_ID'][0] if res and len(res)>0 else 'No_ID'
        e=entry.copy(); e.update({'ra':float(ra),'dec':float(dec),'simbad_id':str(obj)})
        enriched.append(e)
    return wcs, enriched