import cv2
import numpy as np
import matplotlib.pyplot as plt
import rawpy
import os
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

"""
This code contains functions for processing the raw DNG images from the Paper UV camera.
"""

"""
replaces each pixel with the average of its neighbors inside a circular aperture.
The radius of the circular aperture can be adjusted to control the amount of smoothing.
"""
def circular_blur(img, radius):
    
    # Create circular kernel
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel = mask.astype(np.float32)
    
    # Normalize so sum = 1
    kernel /= kernel.sum()
    
    #code to inspect the kernel (for debuggin/testing)
    # plt.figure(figsize=(5,5))
    # plt.imshow(kernel, cmap='gray')
    # plt.title(f"Circular Averaging Kernel (radius={radius})")
    # plt.colorbar(label='weight')
    # plt.show()

    # Apply filter
    return cv2.filter2D(img, -1, kernel)


"""
This code loads the DNG image, extracts the blue channel, and normalizes it to a 0–100 range. 
The resulting image is a 2D array of normalized pixel values that can be used for further processing or visualization.
"""
def load_DNG(file_path):
    try:
        raw = rawpy.imread(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to read RAW file '{file_path}': {e}")
        quit()

    #raw = rawpy.imread(file_path)
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

    #blues=offset(blues,blue_black_level)
    return normalize(blues,blue_black_level,white_level)


"""
This class takes in the calibration data from "calibration.csv" and builds an interpolating function that maps pixel coordinates to calibration values. 
The calibrate_image method applies this mapping to an input image, effectively converting raw pixel values (in pxstr/sec) to calibrated values (in nW).
Ideally this is done only once and via the "View Calibration Map" option in CalibrationPlot.py
"""
class Calibration:
    def __init__(self,fpath):
        self.df=pd.read_csv(fpath)
        x=self.df["X Position [ppos]"].to_numpy()
        y=self.df["Y Position [ppos]"].to_numpy()
        z_values=self.df["Calibration [nW/(pstr/sec)]"].to_numpy()
        points = np.column_stack((x, y))
        self.calibrator = LinearNDInterpolator(points, z_values,fill_value=0)

    #takes image of units pxstr/sec and converts to nW
    def calibrate_image(self,image):
        ny, nx = image.shape

        # Build a grid of (i, j) coordinates
        i_idx = np.arange(ny)
        j_idx = np.arange(nx)
        J, I = np.meshgrid(j_idx, i_idx)      # I = row index, J = col index

        # Evaluate interpolator on all coordinates at once
        coords = np.stack([J.ravel(), I.ravel()], axis=-1)  # shape (ny*nx, 2)
        scale = self.calibrator(coords).reshape(image.shape)         # same shape as image

        return image*scale


"""
Returns contour(s) from the input image at a specified threshold level. 
The contours are returned as a list of arrays, where each array contains the coordinates of the contour points.
Ex: [[[x1,y1]],[[x2,y2]],...]
"""
def get_contours(image,threshold):

    _, thresh = cv2.threshold(image, np.max(image)*threshold, 255, cv2.THRESH_BINARY)
    thresh=thresh.astype(np.uint8)

    #cv2.RETR_LIST
    #contours is a list with each element = [[[x1,y1]],[[x2,y2]],...]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

#takes a contour and returns the coordinates of the contour points as a closed loop (first point repeated at the end) for plotting
#to plot take x=pts_closed[:, 0], y=pts_closed[:, 1]
def contour_to_plt_pts(contour):
    pts = contour[:, 0, :]
    pts_closed = np.vstack([pts, pts[0]]) 
    return pts_closed


#This function takes an image and a contour, creates a binary mask from the contour. The pixels inside the contour are set to 1, and the pixels outside the contour are set to 0.
def get_contour_mask(image, contour):
    interoir_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(interoir_mask, [contour], contourIdx=-1, color=1, thickness=-1)
    return interoir_mask

"""
This code views a specificed slice of the image, highlighted with a mask. The slice is defined by the index bounds y1, y2, x1, x2.
"""
def view_slice(image, y1, y2, x1, x2):

    mask = np.zeros_like(image, dtype=bool)
    mask[y1:y2, x1:x2] = 1#

    highlighted = mask#np.ma.masked_where(~mask, image)

    fig, ax = plt.subplots()

    # full image, faint
    ax.imshow(image, cmap='gray', alpha=1)

    # slice region, full strength
    ax.imshow(highlighted, cmap='gray', alpha=0.3)

    ax.set_title("Slice highlighted with mask")
    plt.show()


"""
sets values below a cutoff to 0, where cutoff is defined as mean + sigma_level*std of the pixel values 
in the slice defined by y1, y2, x1, x2 meant to be the background region of the image
"""
def background_filter_from_slice(image, sigma_level, y1, y2, x1, x2):
    background_slice=image[y1:y2, x1:x2].ravel()
    mean=np.mean(background_slice)
    std=np.std(background_slice)

    filtered_image=image - mean

    cutoff=sigma_level*std

    filtered_image[filtered_image < cutoff] = 0

    return filtered_image


"""
#Example code using these functions

# Path to the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

calibration_file_path=os.path.join(script_dir,"calibration.csv")

c=Calibration(calibration_file_path)


# Join with your file name
fname="Far512Sec.dng"
file_path = os.path.join(script_dir,"..", "Pictures","Divergence_Test","No_Ring_2mm", fname)

blues=load_DNG(file_path)
blues= circular_blur(blues,64)

#_, thresh = cv2.threshold(blues, np.max(blues)*0.5, 255, cv2.THRESH_BINARY)
#thresh=thresh.astype(np.uint8)

# Find contours
#cv2.RETR_LIST
#contours is a list with each element = [[[x1,y1]],[[x2,y2]],...]
contours=get_contours(blues,0.5)#, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Select one contour (example: the first)
cnt = contours[0]
# Convert contour from shape (N, 1, 2) → (N, 2)
pts = cnt[:, 0, :]
pts_closed = np.vstack([pts, pts[0]]) 
area=cv2.contourArea(cnt)
print("Equivalent radius: ",np.sqrt(area/np.pi))

plt.imshow(blues,cmap='gray')
plt.plot(pts_closed[:, 0], pts_closed[:, 1], 'r-', linewidth=1.5)
plt.title(r"Power Resolution After Calibration (8 mm Diameter)")
plt.xlabel("x-pixel")
plt.ylabel("y-pixel")

plt.show();quit()

plt.figure()

plt.imshow(c.calibrate_image(blues),cmap='gray')
plt.title(r"Callibrated Values [nW]")
plt.xlabel("x-pixel")
plt.ylabel("y-pixel")

plt.show()

"""