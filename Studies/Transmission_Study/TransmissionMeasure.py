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
This code is for 
"""

# Join with your file name
#distances
fname_dist=["No_Ring_43muW_1sec.dng","After_Ring_43muW_1sec.dng"]
fname_exposures=[1,1]


calibration = np.load(os.path.join(repo_root, "calibration_map.npy"))

Total_Powers=[]


for i in range(len(fname_dist)):
    fname=fname_dist[i]
    print(fname)

    file_path = os.path.join(script_dir,"..","..", "Pictures","Uncoated_Bot_Ring","Transmission_Effeciency", fname)
    


    image=Image_Tools.load_DNG(file_path)
    image=Image_Tools.circular_blur(image,64)


    image=image/float(fname_exposures[i]) #to pxstr/sec

    image*=calibration

    image=image[image > np.max(image)*0.1] #to ignore negative pixels

    Total_Powers.append(image.sum()*1e-3) #to microWatts

    #contours=Image_Tools.get_contours(image,0.5)#, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contour_list.append(contours[0])

#plt.imshow(calibration,alpha=0.3)
#plt.show()
Total_Powers=np.asarray(Total_Powers)
print("Powers:")
for p in Total_Powers:
    print(p)
#print(np.max(Total_Powers))
normalized_Powers=Total_Powers/np.max(Total_Powers)

print("Transmission Efficiency:")
print(np.min(normalized_Powers))
