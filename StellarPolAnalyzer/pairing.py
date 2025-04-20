"""
pairing.py

Functions for identifying and filtering star pairs in polarimetric images based on
spatial proximity and consistent separation angle.

This module provides:
  - compute_distance_angle: compute Euclidean distance and minimal symmetric angle.
  - find_candidate_pairs: find all star pairs within a given radius.
  - filter_pairs_by_mode: select pairs whose distance and angle match the field's modal values.
"""

import numpy as np
from collections import Counter


def compute_distance_angle(p1, p2):
    """
    Compute the Euclidean distance and the minimal symmetric angle between two points.

    Given two points p1 and p2, calculates:
      - distance: sqrt((x2-x1)^2 + (y2-y1)^2)
      - angle: absolute angle in degrees, normalized to [0°, 90°] so that
               angles > 90° are replaced by 180° - raw_angle.

    Parameters
    ----------
    p1 : sequence of float
        (x, y) coordinates of the first point.
    p2 : sequence of float
        (x, y) coordinates of the second point.

    Returns
    -------
    distance : float
        The straight‐line separation in the same units as the input coordinates.
    angle : float
        The minimal symmetric angle between the line p1→p2 and the horizontal axis,
        in degrees, constrained to the interval [0°, 90°].

    Examples
    --------
    >>> compute_distance_angle((0, 0), (3, 4))
    (5.0, 53.13010235415599)
    >>> compute_distance_angle((0, 0), (-3, 4))
    (5.0, 53.13010235415599)
    """
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    distance = np.hypot(dx, dy)
    raw = abs(np.degrees(np.arctan2(dy, dx)))
    # Use symmetric angle so that 0°–90° covers both quadrants
    angle = 180 - raw if raw > 90 else raw
    return distance, angle


def find_candidate_pairs(sources, max_distance=75):
    """
    Identify all candidate star pairs within a maximum separation.

    Uses a radius‐based nearest‐neighbors search to find, for each source,
    all other sources whose centroid lies within `max_distance` pixels.
    Each pair (i, j) is returned only once (for j > i).

    Parameters
    ----------
    sources : astropy.table.Table or list of dict-like
        Detected source objects. Each must provide fields 'xcentroid' and 'ycentroid'.
    max_distance : float, optional
        Maximum pixel separation to consider two sources as a candidate pair.
        Default is 75.

    Returns
    -------
    pairs : list of tuple
        List of candidate pairs. Each tuple is:
          (i, j, distance, angle)
        where i and j are indices into `sources`, distance is the pixel separation
        and angle is the minimal symmetric angle between the pair (see compute_distance_angle).

    Examples
    --------
    >>> # Suppose `sources` has centroids at [(0,0), (10,0), (0,10)]
    >>> pairs = find_candidate_pairs(sources, max_distance=15)
    >>> len(pairs)
    3
    """
    from sklearn.neighbors import NearestNeighbors

    coords = np.array([(s['xcentroid'], s['ycentroid']) for s in sources])
    nn = NearestNeighbors(radius=max_distance).fit(coords)
    dists, idxs = nn.radius_neighbors(coords)

    pairs = []
    for i, (nbrs, ds) in enumerate(zip(idxs, dists)):
        for j, d in zip(nbrs, ds):
            if j > i:
                dist, ang = compute_distance_angle(coords[i], coords[j])
                pairs.append((i, j, dist, ang))
    return pairs


def filter_pairs_by_mode(candidate_pairs, tol_distance=0.52, tol_angle=0.30):
    """
    Filter candidate pairs to retain only those matching the field's modal separation.

    From a list of (i, j, distance, angle) tuples, determines:
      - The modal (most frequent) distance rounded to two decimals.
      - The modal angle rounded to two decimals.
    Then keeps only those pairs whose rounded distance and angle lie within
    the absolute tolerances tol_distance and tol_angle of the respective modes.

    Parameters
    ----------
    candidate_pairs : list of tuple
        Output from `find_candidate_pairs`, each tuple:
        (i, j, distance, angle).
    tol_distance : float, optional
        Maximum allowed deviation (in pixels) from the modal distance.
    tol_angle : float, optional
        Maximum allowed deviation (in degrees) from the modal angle.

    Returns
    -------
    final_pairs : list of tuple
        Subset of `candidate_pairs` that satisfy:
          |round(distance,2) - distance_mode| <= tol_distance and
          |round(angle,2)    - angle_mode|    <= tol_angle
    distance_mode : float
        The rounded modal distance (px).
    angle_mode : float
        The rounded modal angle (deg).

    Examples
    --------
    >>> cand = [(0,1,36.37,  5.0),
               (2,3,36.38,  4.9),
               (4,5,40.00, 10.0)]
    >>> final, d_mode, a_mode = filter_pairs_by_mode(cand, tol_distance=0.1, tol_angle=0.2)
    >>> d_mode, a_mode
    (36.37, 5.0)
    >>> final
    [(0,1,36.37,5.0)]
    """
    if not candidate_pairs:
        return [], None, None

    distances = [round(p[2], 2) for p in candidate_pairs]
    angles = [round(p[3], 2) for p in candidate_pairs]

    distance_mode = Counter(distances).most_common(1)[0][0]
    angle_mode = Counter(angles).most_common(1)[0][0]

    final_pairs = [
        p for p in candidate_pairs
        if abs(round(p[2], 2) - distance_mode) <= tol_distance
        and abs(round(p[3], 2) - angle_mode)    <= tol_angle
    ]

    return final_pairs, distance_mode, angle_mode
