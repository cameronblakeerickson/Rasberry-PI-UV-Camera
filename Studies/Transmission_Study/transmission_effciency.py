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

calibration = np.load(os.path.join(repo_root, "calibration_map.npy"))


#fnames=["No_Ring_0.1sec.dng","After_Ring_0.1sec.dng"]
#fnames=[ "No_Ring_43muW_1sec.dng","After_Ring_43muW_1sec.dng"]
fnames=["No_Ring_NoLens_43muW_1sec.dng", "After_Ring_NoLens_43muW_1sec.dng"]
exposures=[1.,1.]
powers=[]

# Define slice bounds
y1, y2 = 1000, 1200
x1, x2 = 400, 600

for i in range(len(fnames)):
    fname=fnames[i]
    print(fname)

    file_path = os.path.join(script_dir,"..","..", "Pictures","Uncoated_Bot_Ring","Transmission_Effeciency", fname)
    image=Image_Tools.load_DNG(file_path)


    image=Image_Tools.circular_blur(image,64)
    image=image/exposures[i] #to pxstr/sec
    image*=calibration #to nW

    image=Image_Tools.background_filter_from_slice(image, 3, y1, y2, x1, x2)

    plt.imshow(image, cmap='gray')
    plt.colorbar(label='Intensity (nW)')
    plt.title("Calibrated Image with Background Subtracted")

    print("power: ", image.sum()*1e-3) #to microWatts
    plt.show()


    powers.append(image.sum()*1e-3) #to microWatts


print("Power Ratio (After/Before): ", powers[1]/powers[0])


