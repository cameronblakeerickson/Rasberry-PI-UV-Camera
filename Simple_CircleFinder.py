#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
#quit()

# Load image in grayscale
img_gray = cv2.imread("out.jpg", cv2.IMREAD_GRAYSCALE)


#crop the image
img_gray = img_gray[1320:1570, 1625:1895] #[y1:y2, x1:x2]


#inspect image 
# ~ plt.imshow(img_gray, cmap="gray")
# ~ plt.title("Gray Image")
# ~ plt.axis("off")
# ~ plt.show()
# ~ quit()

# Also keep a color version for drawing
img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

# Smooth to reduce noise
img_blur = cv2.medianBlur(img_gray, 5)

# ~ plt.imshow(img_blur, cmap="gray")
# ~ plt.title("Gray Image")
# ~ plt.axis("off")
# ~ plt.show()
# ~ quit()



# Plot image
fig, ax = plt.subplots()
ax.imshow(img_gray, cmap="gray")
ax.set_title("Detected Circles")
ax.invert_yaxis()   # so (0,0) is top-left like in OpenCV




# Detect circles
circles = cv2.HoughCircles(
    img_blur,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=500,
    param1=200,
    param2=10,
    minRadius=20,
    maxRadius=300
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
plt.show()

