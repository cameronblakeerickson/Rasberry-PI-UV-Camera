import CircularBlur
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from lmfit import Model

"""
This code computes beam divergence given 3 points
"""

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


calibration = np.load("calibration_map.npy")


Total_Powers=[]
contour_list=[]
countour_plot_pts=[]
sds=[]

for i in range(3):
    fname=fname_dist[i]+"_"+fname_exposures[i]+"sec.dng"
    #fname=fname_dist[i]+fname_exposures[i]+"sec.dng"
    print(fname)

    file_path = os.path.join(script_dir,"..", "Pictures","FHG_Fix", fname)
    #file_path = os.path.join(script_dir,"..", "Pictures","Divergence_Test","No_Ring_2mm", fname)


    image=CircularBlur.load_DNG(file_path)
    image=CircularBlur.circular_blur(image,64)


    image=image/float(fname_exposures[i]) #to pxstr/sec

    image*=calibration

    Total_Powers.append(image.sum()*1e-3) #to microWatts

    contours=CircularBlur.get_contours(image,0.1)#, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_list.append(contours[0])

    # Select one contour (example: the first)
    cnt = contours[0]
    # Convert contour from shape (N, 1, 2) → (N, 2)
    pts = cnt[:, 0, :]
    pts_closed = np.vstack([pts, pts[0]]) 
    area=cv2.contourArea(cnt)
    #print("Equivalent radius: ",np.sqrt(area/np.pi))

    #mask = image > np.max(image)*0.05

    #plt.imshow(image,cmap="gray")
    #plt.plot(pts_closed[:,0],pts_closed[:,1])
    #plt.show()

    # Choose contour levels (e.g. 10 equally spaced levels)
    #levels = np.linspace(image.max()*0.1, image.max(), 10)

    # Draw contour lines
    #plt.contour(image, levels=levels, colors='red', linewidths=0.5)

    # plt.title(f"Beam Distance : {distances[i]:.1f} cm")
    # plt.xlabel("x-pixel")
    # plt.ylabel("y-pixel")
    # plt.show()
#plt.imshow(calibration,alpha=0.3)
#plt.show()
Total_Powers=np.asarray(Total_Powers)
print("Power:")
print(np.max(Total_Powers))
normalized_Powers=Total_Powers/np.max(Total_Powers)
#np.save("LowPowerDistances.npy", distances)
#np.save("LowPowerValues.npy", normalized_Powers)
plt.scatter(distances,Total_Powers/np.max(Total_Powers),label=r"After FHG + Lens improvements, 600$\mu$W")
plt.plot(distances,Total_Powers/np.max(Total_Powers))

plt.scatter(np.load("LowPowerDistances.npy"),np.load("LowPowerValues.npy"),label=r"Before, 4$\mu$W")
plt.plot(np.load("LowPowerDistances.npy"),np.load("LowPowerValues.npy"))
plt.xlabel("Camera Distance [cm]")
plt.ylabel("Normalized Total Power")
plt.title("Total Power vs Camera Distance")
plt.legend()
plt.show()
quit()
fig, ax = plt.subplots(figsize=(5,5))
labels=["0 cm","150 cm","300 cm"]
radi=[]

center=contour_list[i]
center=center[:, 0, :]
center=np.mean(center)

for i in range(3):
    c=contour_list[i]
    radi.append(64e-4*np.sqrt(cv2.contourArea(c)/np.pi)) #in cm
    pts = c[:, 0, :]
    pts = np.vstack([pts, pts[0]]) 
    pts=pts-center
    pts*=64e-4 #convert to cm
    ax.plot(pts[:,0], pts[:,1], label=f"Beam Distance : {distances[i]:.1f} cm")   # red points connected with line

radi=np.asarray(radi)
distances=np.asarray(distances)

ax.set_title("10% Peak Power Contours")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.legend()

fig2, ax2 = plt.subplots(figsize=(5,5))
ax2.scatter(distances,radi)
ax2.set_title("Effective Beam Radii")

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