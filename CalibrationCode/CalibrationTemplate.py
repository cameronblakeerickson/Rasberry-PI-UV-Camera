import CircularBlur
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from lmfit import Model

"""
This code is for basic loading, inspection, and comparison of calibrated DNG files
"""

# Path to the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Join with your file name
#distances
fname_dist=["No_Ring_0mm_OffWidnow_4sec.dng","Ring_0mm_OffWidnow_60sec.dng","Ring_4mm_OffWidnow_60sec.dng","Ring_8mm_OffWidnow_60sec.dng","Ring_12mm_OffWidnow_60sec.dng","Ring_16mm_OffWidnow_60sec.dng","Ring_20mm_OffWidnow_60sec.dng","Ring_24mm_OffWidnow_60sec.dng","Ring_-4mm_OffWidnow_60sec.dng","Ring_-8mm_OffWidnow_60sec.dng","Ring_-12mm_OffWidnow_60sec.dng","Ring_-6mm_OffWidnow_60sec.dng","Ring_-5mm_OffWidnow_60sec.dng"]#["Ring_bad_OffWidnow_60sec.dng","Ring_-90deg_OffWidnow_60sec.dng","Ring_-47mm_OffWidnow_60sec.dng","Ring_-36mm_OffWidnow_60sec.dng","Ring_-90deg_OffWidnow_60sec.dng"]#["Ring_0mm_OffWidnow_60sec.dng","Ring_4mm_OffWidnow_60sec.dng","Ring_8mm_OffWidnow_60sec.dng","Ring_12mm_OffWidnow_60sec.dng","Ring_16mm_OffWidnow_60sec.dng","Ring_20mm_OffWidnow_60sec.dng","Ring_24mm_OffWidnow_60sec.dng","Ring_-4mm_OffWidnow_60sec.dng","Ring_-8mm_OffWidnow_60sec.dng","Ring_-12mm_OffWidnow_60sec.dng","Ring_-16mm_OffWidnow_60sec.dng","Ring_-6mm_OffWidnow_60sec.dng","Ring_-5mm_OffWidnow_60sec.dng"]
fname_exposures=np.zeros(len(fname_dist))+60.#[60.,60.,60.,60.,60.,60.,60.]#,4.,4]
fname_exposures[0]=4. #first one is 4 seconds, rest are 60 seconds
xvals=[0,0,4,8,12,16,20,24,-4.,-8.,-12.,-6.,-5.] #in cm#[-90,-47.,-36.,-90,0]#[0,4,8,12,16,20,24,-4.,-8.,-12.,-16.,-6.,-5.] #in cm
xvals=np.asarray(xvals)
xvals=xvals*(0.4/42.)*180./np.pi #to meters, OD of ring is 84 cm
calibration = np.load("calibration_map.npy")


Total_Powers=[]
contour_list=[]
countour_plot_pts=[]
sds=[]

for i in range(len(fname_dist)):
    fname=fname_dist[i]
    print(fname)

    file_path = os.path.join(script_dir,"..", "Pictures","Coated_Top_Ring","Window_Scan_Monday", fname)
    #file_path = os.path.join(script_dir,"..", "Pictures","Divergence_Test","No_Ring_2mm", fname)


    image=CircularBlur.load_DNG(file_path)
    image=CircularBlur.circular_blur(image,64)


    image=image/float(fname_exposures[i]) #to pxstr/sec

    image*=calibration

    Total_Powers.append(image.sum()*1e-3) #to microWatts

    contours=CircularBlur.get_contours(image,0.5)#, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_list.append(contours[0])

    # Select one contour (example: the first)
    cnt = contours[0]
    # Convert contour from shape (N, 1, 2) → (N, 2)
    pts = cnt[:, 0, :]
    pts_closed = np.vstack([pts, pts[0]]) 
    area=cv2.contourArea(cnt)
    print("Equivalent diameter: ",64e-4*np.sqrt(area/np.pi)*2,"cm")

    #mask = image > np.max(image)*0.05

    # ny, nx = image.shape
    # pixel_size_cm = 64e-4 
    # x_half = nx * pixel_size_cm / 2
    # y_half = ny * pixel_size_cm / 2
    # plt.imshow(image,cmap="gray",extent=[-x_half,x_half,-y_half,y_half],origin="lower")
    # plt.title(f"Beam Profile After Ring with 50% Contour, Degree Offset : ~{xvals[i]:.1f} deg")
    # plt.xlabel("x [cm]")
    # plt.ylabel("y [cm]")
    # plt.plot(pts_closed[:,0]*pixel_size_cm-x_half,pts_closed[:,1]*pixel_size_cm-y_half)
    # plt.show()
    # quit()
    #Choose contour levels (e.g. 10 equally spaced levels)
    levels = np.linspace(image.max()*0.1, image.max(), 10)

    #Draw contour lines
    # plt.contour(image, levels=levels, colors='red', linewidths=0.5)

    # plt.title(f"Degree Offset : {xvals[i]:.1f} deg")
    # plt.xlabel("x-pixel")
    # plt.ylabel("y-pixel")
    # plt.show()
#plt.imshow(calibration,alpha=0.3)
#plt.show()
Total_Powers=np.asarray(Total_Powers)
print("Powers:")
for p in Total_Powers:
    print(p)
#print(np.max(Total_Powers))
normalized_Powers=Total_Powers/np.max(Total_Powers)
#np.save("LowPowerDistances.npy", distances)
#np.save("LowPowerValues.npy", normalized_Powers)
plt.scatter(xvals,Total_Powers/217.)
#plt.plot(xvals,Total_Powers)
plt.grid()
# plt.scatter(np.load("LowPowerDistances.npy"),np.load("LowPowerValues.npy"),label=r"Before, 4$\mu$W")
# plt.plot(np.load("LowPowerDistances.npy"),np.load("LowPowerValues.npy"))
plt.xlabel("Angle Offset From Window [degrees]")
plt.ylabel("Relative Power Reduction")
plt.title("Total Power Loss vs Ring Angle")
plt.legend()
plt.show()
#quit()
fig, ax = plt.subplots(figsize=(5,5))
labels=["0 cm","150 cm","300 cm"]
radi=[]

center=contour_list[i]
center=center[:, 0, :]
center=np.mean(center)
centers=[]
for i in range(len(xvals)):
    c=contour_list[i]
    radi.append(64e-4*np.sqrt(cv2.contourArea(c)/np.pi)) #in cm

        # Moments
    M = cv2.moments(c)

    if M["m00"] != 0:  # avoid division by zero
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = np.nan, np.nan

    centers.append((cx, cy))


    pts = c[:, 0, :]
    pts = np.vstack([pts, pts[0]]) 
    pts=pts-center
    pts*=64e-4 #convert to cm
    ax.plot(pts[:,0], pts[:,1], label=f"Angle Offset : {xvals[i]:.1f} deg")   # red points connected with line

radi=np.asarray(radi)

ax.set_title("50% Peak Power Contours")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.legend()

fig2, ax2 = plt.subplots(figsize=(5,5))
ax2.scatter(xvals,radi*2)
ax2.set_title("Effective 50% Beam Diameters Vs Ring Angle")
ax2.set_xlabel("Angle Offset From Window [degrees]")
ax2.set_ylabel("Effective 50% Beam Diameter [cm]")
ax2.axhline(y=1.5, c='r',linestyle='--',label="Without Ring")
ax2.legend()
ax2.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# --- inputs assumed to already exist ---
# xvals: array-like of angle offsets (degrees), length N
# centers: list of (cx, cy) in PIXELS, built like: centers.append((cx, cy))

# --- calibration ---
pix_to_cm = 64e-4 
ny, nx = image.shape
cx_pix = np.array([c[0] for c in centers])
cy_pix = np.array([c[1] for c in centers])

# Flip Y to match origin='lower'
cy_pix = ny/2. - cy_pix
cx_pix = cx_pix - nx/2.

# Convert to cm
cx_vals = cx_pix * pix_to_cm
cy_vals = cy_pix * pix_to_cm

# Optional: recenter so mean position is (0,0)
#cx_vals -= np.mean(cx_vals)
#cy_vals -= np.mean(cy_vals)

# --- XY plane plot with angle labels ---
fig3, ax3 = plt.subplots(figsize=(5, 5))

# First point (reference position)
ax3.scatter(
    cx_vals[0],
    cy_vals[0],
    color='red',
    s=80,
    label='No Ring'
)

# Remaining points
ax3.scatter(
    cx_vals[1:],
    cy_vals[1:]
)


#ax3.scatter(cx_vals, cy_vals)
#ax3.plot(cx_vals, cy_vals, alpha=0.5)  # optional: connect points to show progression

for i, angle in enumerate(xvals):
    ax3.text(
        cx_vals[i], cy_vals[i],
        f"{angle:.1f}°",
        fontsize=9,
        ha="left", va="bottom"
    )

# --- order trajectory by xvals ---
# --- exclude first point from trajectory ---
xvals_arr = np.array(xvals)

cx_traj = cx_vals[1:]
cy_traj = cy_vals[1:]
xvals_traj = xvals_arr[1:]

# Order trajectory by angle
order = np.argsort(xvals_traj)

cx_sorted = cx_traj[order]
cy_sorted = cy_traj[order]

ax3.plot(
    cx_sorted,
    cy_sorted,
    linewidth=2,
    alpha=0.7
)

ax3.set_title("Beam Profile Centers")
ax3.set_xlabel("X Center [cm]")
ax3.set_ylabel("Y Center [cm]")

ax3.axhline(0, linestyle="--", linewidth=1)
ax3.axvline(0, linestyle="--", linewidth=1)

ax3.set_aspect("equal", adjustable="box")
ax3.legend()
ax3.grid(True)

plt.show()

# Define linear model
def line(x, m, b):
    return m * x + b

# Build model
model = Model(line)

# Convert lists to numpy arrays (lmfit prefers arrays)
x = np.array(distances)
y = np.array(radi)

# Initial parameter guesses
params = model.make_params(m=1.0, b=0.0)

# Fit
result = model.fit(y, params, x=x)

# Print fit report
print(result.fit_report())

# Extract best-fit values
m_fit = result.best_values['m']
b_fit = result.best_values['b']

print("Slope m =", m_fit)
print("Offset b =", b_fit)

# Create a smooth x-axis for plotting the fitted line
x_fit = np.linspace(x.min(), x.max(), 200)
y_fit = m_fit * x_fit + b_fit

# --- Plot on Axes ax2 ---
ax2.scatter(x, y, color='blue', s=20, label='Data')
ax2.plot(x_fit, y_fit, color='red', linewidth=2, label='Fit')

ax2.set_xlabel("Beam Distance [cm]")
ax2.set_ylabel("Beam Radius [cm]")
ax2.legend()

slope=m_fit#*64e-4 #slope in cm/cm
print("halAngle = ",np.atan(slope)*180/np.pi)


plt.show()

quit()