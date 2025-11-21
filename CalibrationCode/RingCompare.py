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
fname_dist=["Before_Ring_Far","After_Ring_Far"]
fname_exposures=["512","512"]
distances=[384.7,384.7] #in cm

calibration_file_path=os.path.join(script_dir,"calibration.csv")
c=CircularBlur.Calibration(calibration_file_path)



Total_Powers=[]
contour_list=[]
countour_plot_pts=[]
sds=[]
titles=["Beam Without Ring Transmission", "Beam With Ring Transmission"]

for i in range(len(fname_dist)):
    fname=fname_dist[i]+fname_exposures[i]+"Sec.dng"
    print(fname)
    file_path = os.path.join(script_dir,"..", "Pictures","Divergence_Test","Ring_2mm", fname)

    image=CircularBlur.load_DNG(file_path)
    image=CircularBlur.circular_blur(image,64)


    image=image/float(fname_exposures[i]) #to pxstr/sec

    image=c.calibrate_image(image) #

    Total_Powers.append(image.sum()*1e-3) #to microWatts

    contours=CircularBlur.get_contours(image,0.5)#, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_list.append(contours[0])

    # Select one contour (example: the first)
    #cnt = contours[0]
    # Convert contour from shape (N, 1, 2) → (N, 2)
    #pts = cnt[:, 0, :]
    #pts_closed = np.vstack([pts, pts[0]]) 
    #area=cv2.contourArea(cnt)
    #print("Equivalent radius: ",np.sqrt(area/np.pi))

    #mask = image > np.max(image)*0.05

    #plt.imshow(image[450:1150,750:1350],cmap='gray')
    #plt.plot(pts_closed[:,0],pts_closed[:,1])

    # Choose contour levels (e.g. 10 equally spaced levels)
    #levels = np.linspace(image.max()*0.1, image.max(), 10)

    # Draw contour lines
    #plt.contour(image, levels=levels, colors='red', linewidths=0.5)

    # plt.title(titles[i])
    # plt.xlabel("x-pixel")
    # plt.ylabel("y-pixel")
    # plt.show()

#plt.show()
#plt.scatter(distances,Total_Powers)

print("power loss",Total_Powers[-1]/Total_Powers[0])
#plt.show()
#quit()
fig, ax = plt.subplots(figsize=(5,5))
labels=["Without Ring","With Ring"]
areas=[]
dx=[]
dy=[]


for i in range(2):
    c=contour_list[i]
    areas.append(cv2.contourArea(c))
    pts = c[:, 0, :]
    pts = np.vstack([pts, pts[0]])*64e-4 #convert to cm
    
    dx.append(np.max(pts[:,0])-np.min(pts[:,0]))
    dy.append(np.max(pts[:,1])-np.min(pts[:,1]))
    ax.plot(pts[:,0], pts[:,1], label=labels[i])   # red points connected with line

ax.set_aspect('equal', adjustable='box')
ax.set_xlabel("X position [cm]")
ax.set_ylabel("Y position [cm]")
dx=np.asarray(dx)
dy=np.asarray(dy)

print("Area without/with ",areas[0]/areas[1])
print("Without ring dy/dx:",dy[0]/dx[0])
print("With ring dy/dx:",dy[1]/dx[1])
print("Ring/No Ring dx",dx[1]/dx[0] )
print("Ring/No Ring dy",dy[1]/dy[0] )

ax.set_title("50% Max Power Contours")
#ax.set_xlabel("x")
#ax.set_ylabel("y")
ax.legend()


plt.show()
quit()

radi=np.asarray(radi)
distances=np.asarray(distances)



fig2, ax2 = plt.subplots(figsize=(5,5))
ax2.scatter(distances,radi)
ax2.set_title("Radii")

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

ax2.set_xlabel("Distance")
ax2.set_ylabel("Radius")
ax2.legend()

slope=m_fit*64e-4 #slope in cm/cm
print("Angle = ",np.atan(slope)*180/np.pi)


plt.show()

quit()