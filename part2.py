import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#1. Assign the Miller indexes to all the measured peaks:
    # 38°: (111)
    # 44°: (200)
    # 65°: (220)
    # 78°: (311)
    # 82°: (222)
#2. Calculate for each peak the corresponding value of the lattice constant a.
#3. Using the Scherrer equation, calculate the nanoparticle size.

#%%
# FUNCTION DEFINITIONS

def to_rad(x):
    return x*np.pi/180

def multiwrite(outfile, string):
    outfile.write(string + "\n")
    print(string)
    
def fit_function(theta, A, m, s, r, a, b):
    # A: normalization, m: average, s: FWHM of Lorentzian, r: relative ratio
    sg = s/np.sqrt(2*np.log(2)) # sigma of gaussian
    lor = A/np.pi*s/((theta-m)**2+s**2)
    gaus = A/(sg*np.sqrt(2*np.pi))*np.exp(-(theta-m)**2/(2*sg**2))
    line = a*l+b
    return r*lor+(1-r)*gaus+line

def d(theta):
    return l/(2*np.sin(theta))

def lattice_constant(d, h, k, l):
    return  d*np.sqrt(h**2+k**2+l**2)

def Scherrer(theta, beta):
    return K*l/(beta*np.cos(theta))


#%%
# DATA

fs = 20 # standardize the fontsize
l = 0.15406 # [nm], from CuKa1
beta_instr = to_rad(0.27)
K = 0.89 # using FWHM
r_guess = 0.5
a_guess = 0

data = pd.read_csv("Group01_GIXRD1.txt", sep=" ", header=None)
data.columns = ["2theta", "intensity"]
two_theta_deg = np.array(data["2theta"]) # [deg], detector's angle
theta = np.array(data["2theta"])*np.pi/360 # [rad], Bragg's angle
intensity = np.array(data["intensity"]) #counts

#%%
# INTENSITY SPECTRUM PLOT

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(two_theta_deg, intensity, color="blue")
plt.title("Intensity spectrum", fontdict={"fontname": "Calibri", "fontsize": fs})
plt.xlabel(r"$2\theta$ (deg)", fontdict={"fontsize": fs})
plt.ylabel("I (counts)", fontdict={"fontsize": fs})
plt.tight_layout()

#%%
# FITTING

# TODO:
    # fit each Au peak with fit_function
    # as initial guesses use:
        # r_guess and a_guess for r and a respectively
        # guesses for A, m, s, b depend on the peak and can be obtained visually; for (111) for instance:
            # b: 700 (background), A: 2600 (normalization-background), m: 38/2=19 (average), s=0.5 (FWHM)
            # notice that the Bragg's angle is half of the detector's angle