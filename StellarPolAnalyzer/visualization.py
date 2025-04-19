import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.visualization import ZScaleInterval
import numpy as np


def draw_pairs(image_data, sources, pairs, num_stars, mode_distance, mode_angle,
               tol_distance, tol_angle):
    """Plot image with detected pairs and summary legend."""
    interval = ZScaleInterval()
    z1,z2 = interval.get_limits(image_data)
    fig,ax = plt.subplots(figsize=(12,12))
    ax.imshow(image_data, cmap='gray',origin='lower',vmin=z1,vmax=z2)
    ax.set_title('StellarPol Analyzer')
    coords = np.array([(s['xcentroid'],s['ycentroid']) for s in sources])
    for idx,(x,y) in enumerate(coords): ax.plot(x,y,'ro',markersize=2)
    for i,j,_,_ in pairs:
        x1,y1=coords[i]; x2,y2=coords[j]
        ax.plot([x1,x2],[y1,y2],'lime',lw=0.5)
        left,right = (i,j) if x1<x2 else (j,i)
        for idx,col in [(left,'blue'),(right,'red')]:
            c=coords[idx]; ax.add_patch(Circle(c,5,edgecolor=col,facecolor='none',lw=0.5))
    plt.subplots_adjust(right=0.7)
    text=(f"Stars: {num_stars}\nPairs: {len(pairs)}\nDist: {mode_distance}±{tol_distance}\nAng: {mode_angle}±{tol_angle}")
    plt.figtext(0.75,0.5,text,bbox=dict(facecolor='white',alpha=0.7))
    plt.show()