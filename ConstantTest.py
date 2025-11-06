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

#X errors are ~ +/- 0.05 microW
#100 to 100.8
#85 to 84.6
#70 to 69.6
#55 to 54.3
#45 to 44.8 (with filter)
#35 to 35.1 (with filter)
#25 to 24.88 (with filter)
#15 to 14.95 (with filter)
#1 micro watt to 0.99 microwatt
#0.8 micro watt to 0.81 microwatt
#0.6 to 0.61 microWatt
#70 to 70.1 microW
#50 to 49.7 mciroW
#30

y=[0.02842,
0.02468,
0.02521,
0.0243
]

yerr=[
0.00013,
0.00015,
0.00012,
0.0004
]

x=range(len(y))

# Convert to arrays and sanity-check
x = np.asarray(x)
y = np.asarray(y)
yerr = np.asarray(yerr)

x=x[1::];y=y[1::];yerr=yerr[1::]

# plt.scatter(x,np.asarray(stds)/y)
# plt.show();quit()

# Define linear model: y = m*x + b
def const(x, b):
    return b

model = Model(const)

# Initial guesses (optional but helpful)
params = model.make_params(b=0.02)
#params['b'].set(vary=False)  


# Weights are 1/σ so that the objective is chi-square: Σ((y - f)/σ)^2
result = model.fit(y, params, x=x, weights=1.0/yerr,scale_covar=False)

# Print a nice summary (parameters, 1σ errors, correlations, χ², etc.)
print(result.fit_report())

# Extract best-fit values and 1σ uncertainties

b = result.params['b'].value
b_err = result.params['b'].stderr
bfit = ufloat(b,b_err)


print("Calibration:",bfit)

#print(f"Intercept b = {b:.6g} ± {b_err:.6g}")

nu = result.nfree                         # degrees of freedom
chi2_obs = result.redchi * nu             # or simply: result.chisqr
p_right = chi2.sf(chi2_obs, df=nu)        # P(Chi^2 >= observed)

print(f"nu (dof)        = {nu}")
print(f"chi^2 observed  = {chi2_obs:.3g}")
print(f"reduced chi^2 (with x+y errors)   = {result.redchi:.3g}")
print(f"p-value (right) = {p_right:.3g}")


# Residuals
print("residuals: ",y - result.best_fit)  # unweighted residuals
print("weighted residuals:", (y - result.best_fit)/yerr)

# Generate smooth fit line
xfit = np.linspace(min(x), max(x), 500)
yfit = np.zeros(len(xfit))+b

plt.figure(figsize=(7,5))
plt.errorbar(x, y, yerr=yerr, fmt='none', elinewidth=1.5, capsize=3, label='Data', color='black')
plt.plot(xfit, yfit, label=f'Weighted Mean', color='tab:red')
plt.xlabel(r'Chronological Ordering')
plt.ylabel('Camera Calibration Normalized By Exposure')
plt.title('Consistency Test')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()


quit()

