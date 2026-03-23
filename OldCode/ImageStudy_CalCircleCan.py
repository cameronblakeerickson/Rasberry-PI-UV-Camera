#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import time
import os
import rawpy
import imageio.v3 as iio


#8mm spatial scan used 100 microWatts and 2 sec exposure

import numpy as np

def circle_mask(gray, x_c, y_c, radius):
    """
    Returns the pixel intensities within a given radius (in pixels)
    around the point (x_c, y_c).
    """
    h, w = gray.shape
    # Create coordinate grids
    y_indices, x_indices = np.ogrid[:h, :w]
    
    # Compute distance from the center for each pixel
    dist_sq = (x_indices - x_c)**2 + (y_indices - y_c)**2

    # Create mask for pixels inside the circle
    mask = dist_sq <= radius**2

    return mask


# Path to the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

pics=np.arange(1,1+1,1)

xpos=[]
ypos=[]
cals=[] #calibrations
sds=[] #standard deviations


fname="Background3.dng"
file_path = os.path.join(script_dir, "Pictures","SpatialScan_8mm", fname)

raw = rawpy.imread(file_path)
bayer = raw.raw_image.copy()
bayer_pattern=raw.raw_pattern #2 is blue, 3,1 is green, 0 is red
#finds the position of the blue pixel in the 2x2 bayer pattern
y_idx, x_idx = np.where(bayer_pattern == 2) #indicies of blue in the 2x2 pattern
blue_ind=[y_idx[0], x_idx[0]]
black_levels=raw.black_level_per_channel
blue_black_level = black_levels[bayer_pattern[np.where(bayer_pattern == 2)][0]]
white_level=raw.white_level
span=white_level-blue_black_level
raw.close()

blues = bayer[y_idx[0]::2, x_idx[0]::2]
normalize= lambda x,b,w: np.clip(100*(x.astype(np.float32)-b)/(w-b),0,100)
offset= lambda x,b: np.clip(x.astype(np.float32)-b,0,None)

blues=normalize(blues,blue_black_level,white_level)

xpos=np.linspace(150,1850,100)
ypos=756#np.linspace(300,1400-300,50)#756

cals=[]
for x in xpos:
    mask=circle_mask(blues,x,ypos,80)
    wts = blues[mask]
    cals.append(np.mean(wts))

cals=np.asarray(cals)

fig, ax = plt.subplots()
ax.scatter(xpos, cals/np.max(cals))
ax.set_title("Normalized Mean withing 80 pixel radius")
ax.set_xlabel("X pixel center of circle")

fig2,ax2 = plt.subplots()
ax2.set_title("Background Light Test")
ax2.set_ylabel("y pixel")
ax2.set_xlabel("x pixel")
ax2.imshow(blues, cmap='gray', origin='upper')
ax2.axhline(y=756, color='red', linestyle='--', linewidth=1) 
# Circle with diameter 65 (radius 32.5)
circle_initial = patches.Circle(
    (150, 756),80,
    edgecolor='lime', facecolor='none', linewidth=2
)
circle_final = patches.Circle(
    (1850, 756),80,
    edgecolor='orange', facecolor='none', linewidth=2
)
ax2.add_patch(circle_initial)
ax2.add_patch(circle_final)


plt.show()
quit()





