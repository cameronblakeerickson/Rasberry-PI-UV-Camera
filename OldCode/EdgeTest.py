#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

start= time.time()
#quit()
# Load image in grayscale
img_gray = cv2.imread("Pictures/RecentCalibrationPicture.jpg", cv2.IMREAD_GRAYSCALE)

#arguments are greyscale image, 
#"kernal (the x pixel by x pixel square considered for the local average (must be odd)) and 
#SigmaX=standard deviation of the Gaussian for each pixels (in pixels). 0 auto selects based off of kernal dimensions)
img_blur = cv2.GaussianBlur(img_gray, (13, 13), 0)

#Edge detection test
T1=100
edges = cv2.Canny(img_blur, threshold1=T1, threshold2=T1*2)
plt.imshow(edges, cmap="gray")
plt.title("Canny Edges")
plt.axis("off")
plt.show();quit()


#crop the image
#img_gray = img_gray[1320:1570, 1625:1895] #[y1:y2, x1:x2]





#inspect image 
# ~ plt.imshow(img_gray, cmap="gray")
# ~ plt.title("Gray Image")
# ~ plt.axis("off")
# ~ plt.show()
# ~ quit()

# Also keep a color version for drawing
img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

# Smooth to reduce noise
img_blur = cv2.medianBlur(img_gray, 5) #cv2.bilateralFilter(img_gray, d=15, sigmaColor=150, sigmaSpace=150)
#img_blur0 = cv2.GaussianBlur(img_gray, (13, 13), 20)
#img_blur = cv2.medianBlur(img_blur0, 9)
# ~ plt.imshow(img_blur, cmap="gray")
# ~ plt.title("Gray Image")
# ~ plt.axis("off")
# ~ plt.show()
# ~ quit()



# Plot image
fig, ax = plt.subplots()
ax.imshow(img_blur, cmap="gray")
ax.set_title("Detected Circles")
ax.invert_yaxis()   # so (0,0) is top-left like in OpenCV




# Detect circles
circles = cv2.HoughCircles(
    img_blur,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=200,
    param1=400,
    param2=5,
    minRadius=60,
    maxRadius=90
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

