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
import matplotlib.patches as patches

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


# Define slice bounds
y1, y2 = 1000, 1200
x1, x2 = 400, 600

#Image_Tools.view_slice(image, y1, y2, x1, x2)

image=Image_Tools.background_filter_from_slice(image, 3, y1, y2, x1, x2)
pixel_count=image[image > 0].size
print("Pixel Count: ", pixel_count)
#quit()
plt.imshow(image, cmap='gray')
plt.colorbar(label='Intensity (nW)')
plt.title("Calibrated Image with Background Subtracted")
#plt.show()

power=image.sum()*1e-3 #to microWatts
print("Power: ", power)

quit()
