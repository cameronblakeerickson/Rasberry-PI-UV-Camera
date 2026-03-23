import sys
import os

# Import CalibrationCode modules relative to this file so the script works from any cwd.
script_dir = os.path.dirname(os.path.abspath(__file__))

repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
calibration_dir = os.path.join(repo_root, "CalibrationCode")
if calibration_dir not in sys.path:
    sys.path.insert(0, calibration_dir)

import Image_Tools
import numpy as np
import matplotlib.pyplot as plt
import cv2
from lmfit import Model

"""
This code demonstrates the mask obtained from a contour.

is for demonstrating how to find the contour whose interoir contains a given fraction of the total power.
"""

# Join with your file name
#distances
fname="No_Ring_43muW_1sec.dng"
file_path = os.path.join(script_dir,"..","..", "Pictures","Uncoated_Bot_Ring","Transmission_Effeciency", fname)
image=Image_Tools.load_DNG(file_path)

calibration = np.load(os.path.join(repo_root, "calibration_map.npy"))


interoir_power=[]
contour_list=[]

image=Image_Tools.circular_blur(image,64)
image=image/1. #to pxstr/sec
image*=calibration

contours=Image_Tools.get_contours(image,0.5)
# Select one contour (example: the first)
cnt = contours[0]


#testing of new function to get mask of interoir of contour:
interoir_mask=Image_Tools.get_contour_mask(image, cnt)
pts_closed = Image_Tools.contour_to_plt_pts(cnt)


#plotting the results:
fig, ax = plt.subplots()

# Base image
im = ax.imshow(image, cmap='gray')
fig.colorbar(im, ax=ax, label='Intensity')

# Mask overlay (semi-transparent)
ax.imshow(
    interoir_mask,
    cmap='Reds',      # choose a contrasting colormap
    alpha=0.3,        # transparency (0 = invisible, 1 = solid)
    interpolation='none'
)

# Contour on top
ax.plot(pts_closed[:, 0], pts_closed[:, 1], color='cyan', linewidth=2)

ax.set_title("Mask + Contour Overlay")
plt.show()
