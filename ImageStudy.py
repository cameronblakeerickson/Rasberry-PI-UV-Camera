#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Path to the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Join with your file name
fname="20microW_Exp_100000_8mm2.jpg"
file_path = os.path.join(script_dir, "Pictures","DistributionShift", fname)


image = cv2.imread(file_path)

image=image[400:800, 800:1200]

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



# Extract the Blue channel (OpenCV stores channels as B, G, R)
blue_channel = image[:, :, 0]

# Define bin edges so each bin covers exactly one integer value
# e.g. [0, 1), [1, 2), ..., [254, 255)
bins = np.arange(257)-0.5  # 257 edges → 256 bins

# Plot the histogram
plt.hist(blue_channel.ravel(), bins=bins, color='blue', alpha=0.7)
plt.title("Histogram of Blue Channel")
plt.xlabel("Pixel Intensity (0–255)")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.3)


plt.figure()


#inspect image 
plt.imshow(image_rgb)
#plt.title("Gray Image")
#plt.axis("off")
plt.show()
quit()

# Also keep a color version for drawing
img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

# Smooth to reduce noise
#img_blur = cv2.medianBlur(img_gray, 5) #cv2.bilateralFilter(img_gray, d=15, sigmaColor=150, sigmaSpace=150)
#img_blur0 = cv2.GaussianBlur(img_gray, (13, 13), 20)
#img_blur = cv2.medianBlur(img_blur0, 9)
# ~ plt.imshow(img_blur, cmap="gray")
# ~ plt.title("Gray Image")
# ~ plt.axis("off")
# ~ plt.show()
# ~ quit()

img_blur = cv2.GaussianBlur(img_gray, (13, 13), 0)

# Plot image
fig, ax = plt.subplots()
ax.imshow(img_blur, cmap="gray")
ax.set_title("Detected Circles")
ax.invert_yaxis()   # so (0,0) is top-left like in OpenCV


# Detect circles
circles = cv2.HoughCircles(
    img_blur,
    cv2.HOUGH_GRADIENT,
    dp=1, #1 or 2, 1 
    minDist=30,
    param1=200,  # upper threshold for Canny edge detector
    param2=20, # accumulator threshold for center detection
    minRadius=10,
    maxRadius=60
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0, :]:
        # Outline circle in red
        circ = plt.Circle((x, y), r, color="red", fill=False, linewidth=2)
        ax.add_patch(circ)
        # Mark center in blue
        ax.plot(x, y, "bo")
        print(f"Circle center=({x},{y}), radius={r}")

# Show the result
end = time.time()     # record end time
print(f"Execution time: {end - start:.4f} seconds")
plt.show()

