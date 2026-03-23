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
This code is for demonstrating how to find the contour whose interoir contains a given fraction of the total power.
"""

# Join with your file name
#distances
fname="No_Ring_43muW_1sec.dng"
file_path = os.path.join(script_dir,"..","..", "Pictures","Uncoated_Bot_Ring","Transmission_Effeciency", fname)
image=Image_Tools.load_DNG(file_path)

calibration = np.load(os.path.join(repo_root, "calibration_map.npy"))
image=Image_Tools.circular_blur(image,64)
image=image/1. #to pxstr/sec
image*=calibration #to nW

plt.imshow(image[1000:1100, 1000:1100], cmap='gray')
#plt.colorbar(label='Intensity (nW)')
plt.title("Calibrated Image")
plt.show();quit()

interoir_powers=[]
equivalent_diameters=[]
thresholds=np.logspace(np.log10(0.01),np.log10(0.9), 15)#[0.9,0.5,0.2,0.1,0.05,0.02,0.01]

for t in thresholds:

    contours=Image_Tools.get_contours(image,t)
    # Select one contour (example: the first)
    cnt = contours[0]

    area=cv2.contourArea(cnt)
    equivalent_diameter=(64e-4)*np.sqrt(area/np.pi)*2 #in cm, since pixel size is 64 microns
    equivalent_diameters.append(equivalent_diameter)

    #mask of the interoir of that contour:
    interoir_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(interoir_mask, [cnt], contourIdx=-1, color=1, thickness=-1)

    image_masked = image * interoir_mask
    interoir_power = image_masked.sum()*1e-3 #to microWatts

    interoir_powers.append(interoir_power)

    #pts_closed = Image_Tools.contour_to_plt_pts(cnt)

    # fig, ax = plt.subplots()

    # # Base image
    # im = ax.imshow(image, cmap='gray')
    # fig.colorbar(im, ax=ax, label='Intensity')

    # # Contour on top
    # ax.plot(pts_closed[:, 0], pts_closed[:, 1], color='cyan', linewidth=2)

    # ax.set_title("Mask + Contour Overlay")
    # plt.show()


plt.plot(thresholds,interoir_powers)
plt.xlabel("Threshold for contour")
plt.ylabel("Power in interoir of contour (uW)")
plt.xscale("log")
#plt.gca().invert_xaxis()
plt.figure()
plt.plot(thresholds,equivalent_diameters)
plt.xlabel("Threshold for contour")
plt.ylabel("Equivalent Diameter of Contour (cm)")
plt.xscale("log")
plt.show()
