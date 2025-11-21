#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import time
import os
import rawpy
import imageio.v3 as iio
import pandas as pd;


def list_files_in_folder(folder_path,files=True):
    """
    Lists all non-hidden files and directories in a given folder (non-recursive).

    :param folder_path: Path to the folder
    :return: Tuple (files, directories)
    """
    all_files = []
    all_dirs = []

    for item in os.listdir(folder_path):
        if not item.startswith('.'):  # Skip hidden files and directories
            full_path = os.path.join(folder_path, item)
            if os.path.isfile(full_path):
                all_files.append(full_path)
            elif os.path.isdir(full_path):
                all_dirs.append(full_path)

    if files==False: return all_dirs
    else: return all_files
    #return all_files, all_dirs
#8mm spatial scan used 100 microWatts and 2 sec exposure

# Path to the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))


V_all=True
H_all=True

#H_list=[1]#np.arange(1,19+1,1)# For angle scan: np.arange(3,28+1,1)
#V_list=[7,8,17]#np.arange(1,1+8,1)

if V_all==True:
    V_folder_list =list_files_in_folder(os.path.join(script_dir,"..","Pictures","Calibration_8mm_100microWatt"),files=False)
else:
    V_folder_list = ["V"+str(f) for f in V_list]



xpos=[]
ypos=[]
cals=[] # calibrations
sds=[]  # standard deviations

for folder_name in V_folder_list:
    if H_all==True:
        H_fname_list=list_files_in_folder(os.path.join(script_dir,"..","Pictures","Calibration_8mm_100microWatt",folder_name))
    else:
        H_fname_list = [str(p)+".dng" for p in H_list]

    for fname in H_fname_list:
        print(fname)
        file_path = os.path.join(script_dir,"..","Pictures","Calibration_8mm_100microWatt", folder_name,fname)

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

        #Data was taken with 
        cal=100/(np.sum(wts)/2) #microWatts/(ptr/sec)
        cal*=1000 #convert to nanoWatts
        sum_sigma=np.std(wts)*np.sqrt(len(wts))
        sd=1e5*2*sum_sigma/(np.sum(wts))**2 #nanoWatts/(ptr/sec)
        cals.append(cal)
        sds.append(sd)



fig, ax = plt.subplots()
#fig2,ax2 = plt.subplots()
ax.imshow(blues, cmap='gray', origin='upper')
ax.set_xlabel("x pixel")
ax.set_ylabel("y pixel")
ax.set_title("Detected Centers of Beam Angle Scan")
for i in range(len(xpos)):

    # Optional: mark the centroid itself
     ax.plot(xpos[i], ypos[i], 'r+', markersize=10)


output={
"X Position [ppos]": xpos,
"Y Position [ppos]":ypos,
"Calibration [nW/(pstr/sec)]":cals,
"Calibration Sigma [nW/(pstr/sec)]":sds
}

df=pd.DataFrame(output)

#df.to_csv("Calibration.csv", index=False)
#print(df)



#ax2.errorbar(np.linspace(0,45,len(cals)),cals/np.max(cals),yerr=sds/np.max(cals))
#ax2.set_title(r"Local Mean Calibration from 100$\mu$ W, 8mm Beam, 2 Sec Exposure")
#ax2.set_title(r"Incidient Angle Test")
#ax2.set_ylabel(r"Normalized Pixel Response/[sec$\cdot\mu$W]")
#ax2.set_ylabel(r"Normalized Pixel Response")
#ax2.set_xlabel(r"beam center x pixel location")
#ax2.set_xlabel(r"beam center x pixel location")
#ax2.set_xlabel(r"Approximate Angle [Degrees]")
plt.show()


quit()

