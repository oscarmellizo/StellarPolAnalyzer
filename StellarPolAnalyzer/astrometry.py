"""
astrometry.py

High-level astrometric utilities for the StellarPolAnalyzer library.

This module integrates the workflow for:
  1. Generating a synthetic star field from detected polarimetric pairs.
  2. Solving the World Coordinate System (WCS) of that synthetic image via Astrometry.net.
  3. Converting pixel coordinates of paired stars into sky coordinates (RA, Dec).
  4. Performing a SIMBAD cross-match to annotate each star pair with astronomical identifiers.

Primary Function
----------------
- annotate_with_astrometry_net:
    - Builds a synthetic FITS image by injecting 2D Gaussians at the “ordinary” beam
      positions of each detected star pair.
    - Copies essential header keywords (RA, DEC, OBJECT) from the reference image.
    - Solves the synthetic image’s WCS using Astrometry.net (astroquery).
    - Maps pixel positions to ICRS sky coordinates.
    - Queries SIMBAD for each sky position and enriches the polarimetry results
      with 'ra', 'dec', 'simbad_id', 'object_type', 'classification', and 'bibliography'.
    - Adds an extra entry for the WCS reference coordinate (CRPIX) to ensure the
      reference star is included in the results.

Usage Example
-------------
>>> from StellarPolAnalyzer.astrometry import annotate_with_astrometry_net
>>> wcs, enriched = annotate_with_astrometry_net(
...     ref_path="field_0.fits",
...     sources=sources_table,
...     final_pairs=star_pairs,
...     polarimetry_results=polar_results,
...     fwhm=4.0,
...     api_key="YOUR_API_KEY",
...     simbad_radius=0.02*u.deg,
...     synthetic_name="synthetic_field_0.fits"
... )
>>> for entry in enriched:
...     print(entry["simbad_id"], entry["ra"], entry["dec"], entry["classification"])
"""

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
    Generate a synthetic star field FITS, solve its WCS via Astrometry.net, and
    cross-match each detected pair with SIMBAD, annotating with extended metadata.

    Parameters
    ----------
    ref_path : str
        Path to the reference FITS (0° image).
    sources : astropy.table.Table or list
        Detected sources; must include 'xcentroid', 'ycentroid'.
    final_pairs : list of tuple
        Star pairs (i, j, distance_px, angle_deg).
    polarimetry_results : list of dict
        Results from compute_polarimetry_for_pairs, each with:
          - 'pair_index'
          - 'fluxes'[0.0]['ord_flux'], ['ext_flux']
    fwhm : float
        Full width at half maximum for Gaussian injection (pixels).
    api_key : str
        Astrometry.net API key. If None, WCS solving is skipped.
    simbad_radius : astropy.units.Quantity
        Radius for SIMBAD query around each sky position.
    synthetic_name : str
        Filename for the synthetic FITS output.

    Returns
    -------
    wcs : astropy.wcs.WCS or None
        WCS solution for the synthetic image, or None if not solved.
    enriched : list of dict
        Copy of polarimetry_results augmented with:
          - 'ra'           : Right Ascension (deg)
          - 'dec'          : Declination (deg)
          - 'simbad_id'    : Main SIMBAD identifier or 'No_ID'
          - 'object_type'  : SIMBAD object type
          - 'classification': Human-readable class
          - 'bibliography' : List of Bibcodes from SIMBAD
          - 'note'         : Optional note (e.g., 'reference coordinate')
    """
    # Read header from reference
    hdr = fits.getheader(ref_path)
    ny, nx = hdr['NAXIS2'], hdr['NAXIS1']

    # Convert FWHM to Gaussian sigma
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Build synthetic image
    syn = np.zeros((ny, nx))
    positions = []
    for entry in polarimetry_results:
        idx = entry['pair_index']
        i, j, _, _ = final_pairs[idx]
        x = sources[i]['xcentroid']
        y = sources[i]['ycentroid']
        positions.append((x, y))
        amp = entry['fluxes'][0.0]['ord_flux'] + entry['fluxes'][0.0]['ext_flux']
        g = Gaussian2D(amplitude=amp, x_mean=x, y_mean=y,
                       x_stddev=sigma, y_stddev=sigma)
        yy, xx = np.mgrid[0:ny, 0:nx]
        syn += g(xx, yy)

    # Escribimos el FITS sintético **con todo** el header original
    with fits.open(ref_path) as ref_hdul:
        orig_header = ref_hdul[0].header.copy()

    # Creamos un PrimaryHDU usando la imagen sintética y el header completo
    synthetic_hdu = fits.PrimaryHDU(data=syn, header=orig_header)
    synthetic_hdu.writeto(synthetic_name, overwrite=True)

    # Solve WCS if API key provided
    ast = AstrometryNet()
    ast.api_key = api_key
    sol = ast.solve_from_image(synthetic_name)
    wcs_hdr = fits.Header(sol)
    wcs = WCS(wcs_hdr)
    
    #wcs = None
    #if api_key:
    #    ast = AstrometryNet()
    #    ast.api_key = api_key
#
    #    # Determine reference RA/Dec: prefer 'RA'/'DEC' header keywords, else use WCS(CRPIX)
    #    ra0 = hdr.get('RA')
    #    dec0 = hdr.get('DEC')
    #    if ra0 is None or dec0 is None:
    #        wcs_ref = WCS(hdr)
    #        x0, y0 = hdr.get('CRPIX1', nx/2.0), hdr.get('CRPIX2', ny/2.0)
    #        ra0, dec0 = wcs_ref.all_pix2world([[x0, y0]], 1)[0]
#
    #    # Pixel scale in arcsec/pix from CDELT1
    #    pixscale_deg    = abs(hdr.get('CDELT1', 0.0))
    #    pixscale_arcsec = pixscale_deg * 3600.0
#
    #    settings = {'solve_timeout': 300,
    #                'center_ra': ra0,
    #                'center_dec': dec0,
    #                'scale_units': 'arcsecperpix',
    #                'scale_lower': pixscale_arcsec * 0.9,
    #                'scale_upper': pixscale_arcsec * 1.1}
#
    #    sol = ast.solve_from_image(synthetic_name, **settings)
    #    wcs = WCS(fits.Header(sol))

    # Convert pixel positions to sky coordinates
    pix = np.array(positions)
    world = wcs.all_pix2world(pix, 1)  # returns array [[ra, dec], ...]
    
    if wcs:
        # Extract x,y arrays
        pos_arr = np.array(positions)
        xs_pix = pos_arr[:, 0]
        ys_pix = pos_arr[:, 1]
        # Properly map pixels to sky
        ras_sky, decs_sky = wcs.all_pix2world(xs_pix, ys_pix, 1)
        # Combine into list of tuples
        world = list(zip(ras_sky, decs_sky))
    else:
        world = [(None, None)] * len(positions)
    
    # Configure SIMBAD fields
    Simbad.reset_votable_fields()
    Simbad.add_votable_fields('otype', 'bibcodelist(20)')

    # Mapping of SIMBAD types to polarizing classes
   #olarizing_types = {
   #   'Em*':   'Emission-line star',
   #   'Be*':   'Classical Be star',
   #   'B[e]':  'B[e] star',
   #   'HAeBe': 'Herbig Ae/Be star',
   #   'TT*':   'T Tauri star',
   #   'WR*':   'Wolf–Rayet star',
   #   'C*':    'Carbon star',
   #   'AGB':   'Asymptotic Giant Branch star',
   #   'Mi*':   'Mira variable',
   #   'PN':    'Planetary Nebula',
   #   'RNe':   'Reflection Nebula',
   #   'Sy*':   'Symbiotic star',
   #   'CV*':   'Cataclysmic Variable',
   #}

    enriched = []
    for entry, (ra, dec) in zip(polarimetry_results, world):
        # Validate RA/Dec within physical ranges
        valid_coord = (ra is not None) and (-90.0 <= dec <= 90.0)
        if valid_coord:
            sc = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            res = Simbad.query_region(sc, radius=simbad_radius)
        else:
            res = None

        main_id = 'No_ID'; otype = ''; classification = 'No identification'; bib_list = []
        if res is not None and len(res) > 0:
            main_id = res['MAIN_ID'][0]
            if isinstance(main_id, bytes): main_id = main_id.decode('utf-8')
            otype = res['OTYPE'][0]
            if isinstance(otype, bytes): otype = otype.decode('utf-8')
           #if otype in polarizing_types:
           #    classification = polarizing_types[otype]
           #elif ('be+ x-ray' in otype.lower() and 'binary' in otype.lower()) or 'hmxb' in otype.lower() or ('xrb' in otype.lower() and 'be' in otype.lower()):
           #    classification = 'BeX'
           #else:
           #    classification = 'Non-polarizing candidate'
            for col in res.colnames:
                if 'BIBCODE' in col.upper():
                    bib_data = res[col][0]
                    if isinstance(bib_data, (bytes, str)):
                        bib_list = bib_data.decode('utf-8').split('|') if isinstance(bib_data, bytes) else bib_data.split('|')
                    else:
                        bib_list = [b.decode('utf-8') if isinstance(b, bytes) else str(b) for b in bib_data]
                    break
        
        # Prepare entry with clamped coordinates
        e = entry.copy()
        e.update({
            'ra':           float(ra) if valid_coord else None,
            'dec':          float(dec) if valid_coord else None,
            'simbad_id':    main_id,
            'object_type':  otype,
            #'classification': classification,
            'bibliography': bib_list
        })
        enriched.append(e)
       #e.update({'ra': float(ra) if ra is not None else None,
       #          'dec': float(dec) if dec is not None else None,
       #          'simbad_id': main_id,
       #          'object_type': otype,
       #          'classification': classification,
       #          'bibliography': bib_list})
       #enriched.append(e)

        # Add a fallback entry for the reference coordinate
   #if api_key:
   #    # Initialize fallback with required polarimetry keys to avoid KeyError in visualization
   #    ref_entry = {
   #        'ra': ra0,
   #        'dec': dec0,
   #        'P': None,
   #        'theta': None
   #    }
   #    sc_ref = coord.SkyCoord(ra=ra0*u.deg, dec=dec0*u.deg, frame='icrs')
   #    res_ref = Simbad.query_region(sc_ref, radius=simbad_radius)
   #    if res_ref is not None and len(res_ref) > 0:
   #        main_id_ref = res_ref['MAIN_ID'][0]
   #        if isinstance(main_id_ref, bytes): main_id_ref = main_id_ref.decode('utf-8')
   #        otype_ref = res_ref['OTYPE'][0]
   #        if isinstance(otype_ref, bytes): otype_ref = otype_ref.decode('utf-8')
   #        class_ref = polarizing_types.get(otype_ref, 'Non-polarizing candidate')
   #        bib_list_ref = []
   #        for col in res_ref.colnames:
   #            if 'BIBCODE' in col.upper():
   #                bib_data_ref = res_ref[col][0]
   #                if isinstance(bib_data_ref, (bytes, str)):
   #                    bib_list_ref = bib_data_ref.decode('utf-8').split('|') if isinstance(bib_data_ref, bytes) else bib_data_ref.split('|')
   #                else:
   #                    bib_list_ref = [b.decode('utf-8') if isinstance(b, bytes) else str(b) for b in bib_data_ref]
   #                break
   #        # Update fallback entry with SIMBAD results
   #        ref_entry.update({
   #            'simbad_id':     main_id_ref,
   #            'object_type':   otype_ref,
   #            'classification': class_ref,
   #            'bibliography':  bib_list_ref,
   #            'note':          'reference coordinate'
   #        })
   #    enriched.insert(0, ref_entry)

    return wcs, enriched
