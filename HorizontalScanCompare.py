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

#folder1="SpatialScan_8mm"
folder1="Rail_Test_8mm_2_Sec"
folder2="Rail_Test_8mm_2_Sec_2"
folder3="Rail_Test_8mm_2_Sec_3"

# folder1="Vertical_Rail_Test_8mm_2_Sec"
# folder2="Vertical_Rail_Test_8mm_2_Sec_2"

colors=['r','g','b']

Horizontal=True

#pics1=np.arange(3,28+1,1)# For angle scan: np.arange(3,28+1,1)
# For translation Scan
pics1=np.arange(1,20+1,1)
pics1=np.delete(pics1, np.where(pics1 == 17)[0][0]) # Remove 17 which was forgotten
pics2=np.arange(1,19+1,1)
pics3=np.arange(1,19+1,1)

folder_list=[folder1,folder2,folder3]
pic_list=[pics1,pics2,pics3]

#pics1=np.arange(1,15+1,1)
#pics2=np.arange(1,15+1,1)

#folder_list=[folder1,folder2]
#pic_list=[pics1,pics2]

xpos_list=[]
ypos_list=[]
cals_list=[]
sds_list=[]


for i in range(len(pic_list)):
    pics=pic_list[i]
    folder=folder_list[i]

    xpos=[]
    ypos=[]
    cals=[] # calibrations
    sds=[]  # standard deviations

    for p in pics:
        fname=str(p)+".dng"
        print(fname)
        file_path = os.path.join(script_dir, "Pictures",folder, fname)

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

        #blues=cv2.GaussianBlur(blues, (21, 21), 0)#cv2.blur(blues, (100, 100)) 

        mask = blues > 1.0

        wts = blues[mask]

        #plt.imshow(blues);plt.show();
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
        sd=np.std(wts)*np.sqrt(len(wts))/(2*100.)
        sd=np.sqrt(sd**2+(0.02/cal)**2) #Add 2% error due to laser power drift
        cals.append(cal)
        sds.append(sd)

    xpos_list.append(np.asarray(xpos))
    ypos_list.append(np.asarray(ypos))
    cals_list.append(np.asarray(cals))
    sds_list.append(np.asarray(sds))
    

fig, ax = plt.subplots()
fig2,ax2 = plt.subplots()
ax.imshow(blues, cmap='gray', origin='upper')
ax.set_xlabel("x pixel")
ax.set_ylabel("y pixel")
ax.set_title("Detected Centers of Beam Position Scan")



for i in range(len(xpos_list)):
    xpos=xpos_list[i]
    ypos=ypos_list[i]
    for j in range(len(xpos)):
        ax.plot(xpos[j], ypos[j], colors[i]+'+', markersize=10)


for i in range(len(xpos_list)):
    if Horizontal==True: pos=xpos_list[i]
    else: pos=ypos_list[i]
    ax2.errorbar(pos,cals_list[i],yerr=sds_list[i],c=colors[i],label="set "+str(i+1))

ax2.legend()
ax2.set_title(r"Local Mean Calibration from 100$\mu$ W, 8mm Beam, 2 Sec Exposure")
ax2.set_ylabel(r"Normalized Pixel Resposne/[sec$\cdot\mu$W]")
ax2.set_xlabel(r"beam center x pixel location")

fig3,ax3 = plt.subplots()



errors=np.abs((cals_list[0]-cals_list[1])/sds_list[0])
distances=np.sqrt((xpos_list[0]-xpos_list[1])**2+(ypos_list[0]-ypos_list[1])**2)

ax3.scatter(errors,distances)

errors2=np.abs((cals_list[0]-cals_list[2])/sds_list[0])
distances2=np.sqrt((xpos_list[0]-xpos_list[2])**2+(ypos_list[0]-ypos_list[2])**2)
ax3.scatter(errors2,distances2)


plt.show()


#quit()

