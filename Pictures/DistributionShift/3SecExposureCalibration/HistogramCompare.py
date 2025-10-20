#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import rawpy
from lmfit import Model
from scipy.stats import chi2
from uncertainties import ufloat


# Path to the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Join with your file name
powers=[30,50,70,90]#,70,90]
bins=np.linspace(4, 100, 100)
histograms=[]
means=[]
sums=[]
stds=[]
mean_stds=[]

for p in powers:
    fname=str(int(p))+"microWatt.dng"
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
    raw.close()

    blues = bayer[y_idx[0]::2, x_idx[0]::2]
    normalize= lambda x,b,w: np.clip(100*(x.astype(np.float32)-b)/(w-b),0,100)
    offset= lambda x,b: np.clip(x.astype(np.float32)-b,0,None)

    #blues=offset(blues,blue_black_level)
    blues=normalize(blues,blue_black_level,white_level)

    blues=blues[700:1000,800:1200]
    blues=blues[75:250,100:350] #good for 90

    single_histo, _ =np.histogram(blues.ravel(), bins=bins)
    histograms.append(single_histo)
    means.append(np.mean(blues.ravel()))
    stds.append(np.std(blues.ravel()))
    sums.append(np.sum(blues.ravel()))
    N=np.size(blues)
    mean_stds.append(stds[-1]/np.sqrt(N))

# counts=histograms
# # for i in range(len(histograms)):
# #     counts.append(histograms[i]-histograms[0])

# # Plot
# plt.figure(figsize=(8,5))
# #counts=histograms
# width = (bins[1] - bins[0]) / (len(counts))  # narrower per group
# offsets = np.linspace(-width*(len(counts)-1)/2, width*(len(counts)-1)/2, len(counts))

# for i, c in enumerate(counts):
#     plt.bar(bins[:-1] + offsets[i], c, width=width, alpha=0.8, label=f'Hist {i+1}')

# plt.title("Multiple Histograms (Bar Style)")
# plt.xlabel("Value")
# plt.ylabel("Count")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.3)
# plt.show()


# Convert to arrays and sanity-check
x = np.asarray(powers, dtype=float)
y = np.asarray(means, dtype=float)*N
yerr = np.asarray(mean_stds, dtype=float)*N


# plt.scatter(x,np.asarray(stds)/y)
# plt.show();quit()

# Define linear model: y = m*x + b
def line(x, m, b):
    return m * x + b

model = Model(line)

# Initial guesses (optional but helpful)
params = model.make_params(m=1.0, b=0)
params['b'].set(vary=False)  


# Weights are 1/σ so that the objective is chi-square: Σ((y - f)/σ)^2
result = model.fit(y, params, x=x, weights=1.0 / yerr)

# Print a nice summary (parameters, 1σ errors, correlations, χ², etc.)
print(result.fit_report())

# Extract best-fit values and 1σ uncertainties
m = result.params['m'].value
b = result.params['b'].value
m_err = result.params['m'].stderr
b_err = result.params['b'].stderr

mfit = ufloat(m,m_err)
print("Fit:",mfit)
print(f"\nSlope m = {m:.6g} ± {m_err:.6g}")
print(f"Intercept b = {b:.6g} ± {b_err:.6g}")
print(f"Reduced chi-square = {result.redchi:.3g}")



nu = result.nfree                         # degrees of freedom
chi2_obs = result.redchi * nu             # or simply: result.chisqr
p_right = chi2.sf(chi2_obs, df=nu)        # P(Chi^2 >= observed)

print(f"nu (dof)        = {nu}")
print(f"chi^2 observed  = {chi2_obs:.3g}")
print(f"reduced chi^2   = {result.redchi:.3g}")
print(f"p-value (right) = {p_right:.3g}")


# Residuals
print("residuals: ",y - result.best_fit)  # unweighted residuals
print("weighted residuals:", (y - result.best_fit)/yerr)

# Generate smooth fit line
xfit = np.linspace(min(x), max(x), 500)
yfit = line(xfit, m, b)

plt.figure(figsize=(7,5))
plt.errorbar(x, y, yerr=yerr, fmt='none', elinewidth=1.5, capsize=3, label='Data (±1σ)', color='black')
plt.plot(xfit, yfit, label=f'Fit: y = ({m:.3f} ± {m_err:.3f})x', color='tab:blue')
plt.xlabel(r'Laser Power [$\mu$Watts]')
plt.ylabel('ADC Mean (Normalized 0 to 100)')
plt.title('Linear Fit')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()


quit()
# plt.scatter(powers,means)
# plt.errorbar(powers,means,yerr=mean_stds)

# plt.show();quit()


#Plot them all on the same figure
plt.figure(figsize=(8,5))
for i, h in enumerate(histograms):
    plt.plot(bins[:-1], h, label=rf'{powers[i]}$\mu$ Watts', alpha=0.8)

plt.title("Laser Power Scan")
plt.xlabel("Normalized Value (0 to 100)")
plt.ylabel("Count")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

quit()
