import CircularBlur
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from lmfit import Model


# Path to the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Join with your file name
#distances
fname_dist=["Pos1","Pos2","Pos3"]
fname_exposures=["1","3","6"]
distances=[0,150,300] #in cm

# fname_dist=["Near","Mid","Far"]
# fname_exposures=["16","128","512"]
# distances=[0,167.2,384.7] #in cm


calibration_file_path=os.path.join(script_dir,"calibration.csv")
c=CircularBlur.Calibration(calibration_file_path)

fname=fname_dist[0]+"_"+fname_exposures[0]+"sec.dng"
#fname=fname_dist[i]+fname_exposures[i]+"sec.dng"
print(fname)
file_path = os.path.join(script_dir,"..", "Pictures","FHG_Fix", fname)

image=np.ones((1520, 2028), dtype=np.float64)  # choose dtype explicitly#CircularBlur.load_DNG(file_path)
image=c.calibrate_image(image)

plt.imshow(image,cmap='gray')
np.save("calibration_map.npy", image)
print("Exported calibrated image")
plt.show()

quit()

