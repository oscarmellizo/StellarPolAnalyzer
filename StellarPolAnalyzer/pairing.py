import numpy as np
from collections import Counter


def compute_distance_angle(p1, p2):
    """
    Compute distance and minimal symmetric angle between points.

    Returns
    -------
    distance: float
    angle   : float in [0,90]
    """
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    distance = np.hypot(dx, dy)
    raw = abs(np.degrees(np.arctan2(dy, dx)))
    return distance, (180-raw if raw>90 else raw)


def find_candidate_pairs(sources, max_distance=75):
    """
    Find all star pairs within max_distance using radius neighbors.

    Returns
    -------
    pairs: list of (i, j, distance, angle)
    """
    from sklearn.neighbors import NearestNeighbors
    coords = np.array([(s['xcentroid'], s['ycentroid']) for s in sources])
    nn = NearestNeighbors(radius=max_distance).fit(coords)
    dists, idxs = nn.radius_neighbors(coords)
    pairs = []
    for i, (nbrs, ds) in enumerate(zip(idxs, dists)):
        for j, d in zip(nbrs, ds):
            if j>i:
                dist, ang = compute_distance_angle(coords[i], coords[j])
                pairs.append((i, j, dist, ang))
    return pairs


def filter_pairs_by_mode(candidate_pairs, tol_distance=0.52, tol_angle=0.30):
    """
    Filter pairs whose distance and angle match the modal values within tolerances.

    Returns
    -------
    final_pairs : list of tuples
    d_mode      : float
    a_mode      : float
    """
    if not candidate_pairs:
        return [], None, None
    distances = [round(p[2],2) for p in candidate_pairs]
    angles    = [round(p[3],2) for p in candidate_pairs]
    d_mode = Counter(distances).most_common(1)[0][0]
    a_mode = Counter(angles).most_common(1)[0][0]
    final = [p for p in candidate_pairs
             if abs(round(p[2],2)-d_mode)<=tol_distance
             and abs(round(p[3],2)-a_mode)<=tol_angle]
    return final, d_mode, a_mode