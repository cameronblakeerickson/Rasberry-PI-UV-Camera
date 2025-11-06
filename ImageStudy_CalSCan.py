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

# Path to the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

pics= np.arange(1,8+1,1)# For angle scan: np.arange(3,28+1,1)
#pics=np.delete(pics, np.where(pics == 17)[0][0])

xpos=[]
ypos=[]
cals=[] # calibrations
sds=[]  # standard deviations

for p in pics:
    fname=str(p)+".dng"
    print(fname)
    file_path = os.path.join(script_dir, "Pictures","Angle_Test", fname)

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

    mask = blues > 1.0


    wts = blues[mask]
    # coordinate grids (y rows, x cols)
    ys, xs = np.nonzero(mask)
    mask_vals = blues[ys, xs]  # intensities at those positions
    sum_w = wts.sum()
    x_c = (xs * wts).sum() / sum_w
    y_c = (ys * wts).sum() / sum_w
    print(f"Centroid at x={x_c}, y={y_c}")
    xpos.append(x_c)
    ypos.append(y_c)

    cal=np.sum(wts)/(2.*100.)
    sd=np.std(wts)*np.sqrt(len(wts))/(2*100)
    cals.append(cal)
    sds.append(sd)



print("min pos", np.min(xpos))
print("max pos", np.max(xpos))
# plt.imshow(blues,cmap='gray')
# plt.plot(x_c, y_c, "bo")
# plt.title(r"Center")
# plt.xlabel("x-pixel")
# plt.ylabel("y-pixel")

fig, ax = plt.subplots()
fig2,ax2 = plt.subplots()
ax.imshow(blues, cmap='gray', origin='upper')
ax.set_xlabel("x pixel")
ax.set_ylabel("y pixel")
ax.set_title("Detected Centers of Beam Angle Scan")
for i in range(len(xpos)):

    # Optional: mark the centroid itself
     ax.plot(xpos[i], ypos[i], 'r+', markersize=10)


ax2.errorbar(np.linspace(0,45,len(cals)),cals/np.max(cals),yerr=sds/np.max(cals))
#ax2.set_title(r"Local Mean Calibration from 100$\mu$ W, 8mm Beam, 2 Sec Exposure")
ax2.set_title(r"Incidient Angle Test")
#ax2.set_ylabel(r"Normalized Pixel Response/[sec$\cdot\mu$W]")
ax2.set_ylabel(r"Normalized Pixel Response")
#ax2.set_xlabel(r"beam center x pixel location")
#ax2.set_xlabel(r"beam center x pixel location")
ax2.set_xlabel(r"Approximate Angle [Degrees]")
plt.show()


quit()

