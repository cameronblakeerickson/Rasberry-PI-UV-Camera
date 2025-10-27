#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import rawpy
import imageio.v3 as iio


# Path to the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Join with your file name
fname="90microWatt.dng"
file_path = os.path.join(script_dir, "Pictures","DistributionShift","3SecExposureCalibration", fname)


raw = rawpy.imread(file_path)
bayer = raw.raw_image.copy()
bayer_pattern=raw.raw_pattern #2 is blue, 3,1 is green, 0 is red
#finds the position of the blue pixel in the 2x2 bayer pattern
y_idx, x_idx = np.where(bayer_pattern == 2) #indicies of blue in the 2x2 pattern
blue_ind=[y_idx[0], x_idx[0]]
black_levels=raw.black_level_per_channel
blue_black_level = black_levels[bayer_pattern[np.where(bayer_pattern == 2)][0]]
white_level=raw.white_level
span=white_level-blue_black_level

print("Black level(s):", black_levels)
print("blue black level:", blue_black_level)
print("white level:",raw.white_level)
print("range:",raw.white_level-raw.black_level_per_channel[0])
raw.close()



blues = bayer[y_idx[0]::2, x_idx[0]::2]
print("max before norm",np.max(blues))
normalize= lambda x,b,w: np.clip(100*(x.astype(np.float32)-b)/(w-b),0,100)
offset= lambda x,b: np.clip(x.astype(np.float32)-b,0,None)

#blues=offset(blues,blue_black_level)
blues=normalize(blues,blue_black_level,white_level)

#goes y range then x range
#blues=blues[1200:1375,950:1150]

#For on Axis measurements:
blues=blues[700:1000,800:1200]

mask = blues > 0.3

#if not np.any(mask):
#    h, w = g.shape
#    (w/2.0, h/2.0)

wts = blues[mask]
# coordinate grids (y rows, x cols)
ys, xs = np.nonzero(mask)
sum_w = wts.sum()
x_c = (xs * wts).sum() / sum_w
y_c = (ys * wts).sum() / sum_w
print(f"Centroid at x={x_c}, y={y_c}")


plt.imshow(blues,cmap='gray')
plt.plot(x_c, y_c, "bo")
plt.title(r"Center")
plt.xlabel("x-pixel")
plt.ylabel("y-pixel")
plt.show()
quit()


# Compute 2D Fourier Transform
f = np.fft.fft2(blues)
fshift = np.log(np.fft.fftshift(f))*20  # Shift zero frequency to center

# Compute magnitude spectrum
magnitude_spectrum = np.abs(fshift)

# Display results
plt.figure()
plt.subplot(121), plt.imshow(blues, cmap='gray')
plt.title('Original Image'), plt.axis('off')
plt.subplot(122), plt.imshow(np.abs(fshift), cmap='gray')
plt.title('Magnitude Spectrum'), plt.axis('off')
plt.show()

quit()





mean=np.mean(blues.ravel())
std=np.std(blues.ravel())
print(f"mean: {mean}\n std: {std}")

plt.figure()



# Define bin edges so each bin covers exactly one integer value
# e.g. [0, 1), [1, 2), ...
#bins = np.arange(span)-0.5  # 257 edges → 256 bins
#bins = np.linspace(0, 100, 500)
bins = np.linspace(0, 100, 100)

# Plot the histogram
plt.hist(blues.ravel(), bins=bins, color='blue', alpha=0.7)
plt.title("Histogram of Pixel Intensities")
plt.xlabel(f"normalized value (0–{100})")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.3)
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

