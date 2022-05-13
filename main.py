import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# TODO(EASY):
    # fix units of measure, names, ... in graphs
    # find more refined values for bulk constants (Ashcroft Mermin)
# TODO(MEDIUM):
    # fit Absorbance_JC and plot with the vest fit values
# TODO(HARD):
    # fit Absorbance and do all the chi-squared shenaningans

wp = 1.37*10**16  #[Hz]
vf = 1.4*10**15  #[nm/s]
Gamma_bulk = 1.08*10**14  #[Hz]
c = 299792458*10**9 # [nm/s]
z = 10**7 # [nm]

bulk_data = pd.read_csv("bulk gold dieletric functions.txt", sep="\t", header=None)
bulk_data.columns = ["lambda", "epsilon1", "epsilon2"]
epsilon1_bulk = np.array(bulk_data["epsilon1"]) # from 200nm to 900nm
epsilon2_bulk = np.array(bulk_data["epsilon2"]) # from 200nm to 900nm
epsilon1_bulk_restricted = epsilon1_bulk[200:601] # from 400nm to 800nm
epsilon2_bulk_restricted = epsilon2_bulk[200:601] # from 400nm to 800nm

np_data = pd.read_csv("G01-NPs.dat", sep="\t", header=None)
np_data.columns = ["lambda", "absorbance"]
absorbance = np.array(np_data["absorbance"])

l = np.array(np_data["lambda"])  # from 400nm to 800nm
l_bulk = np.array(bulk_data["lambda"]) # from 200nm to 900nm
#%%
# FUNCTION DEFINITIONS


def Gamma(R): # [Hz]
    return Gamma_bulk * (1+(np.pi*vf/(4*Gamma_bulk*R)))


def omega(l):  #[Hz]
    return 2*np.pi*c/l


def epsilon1(l, R):
    try:
        e1 = epsilon1_bulk[l-200]+wp**2 * (1/(omega(l)**2+Gamma_bulk**2)-1/(omega(l)**2+Gamma(R)**2))
        return e1
    except:
        print("Invalid wavelength")
        
        
def epsilon2(l, R):
    try:
        e2 = epsilon2_bulk[l-200]-wp**2/omega(l) * (Gamma_bulk/(omega(l)**2+Gamma_bulk**2)-Gamma(R)/(omega(l)**2+Gamma(R)**2))
        return e2
    except:
        print("Invalid wavelength")
    
    
def Absorbance(l, R, epsilonm, rho): # rho = [nm**(-3)]
    return np.log10(np.e)*9*z*omega(l)/c*epsilonm**(3/2)*4/3*np.pi*R**3*rho*epsilon2(l, R)/((epsilon1(l, R)+2*epsilonm)**2+(epsilon2(l, R))**2)


def Absorbance_JC(l, epsilonm, f):
    return np.log10(np.e)*9*z*omega(l)/c*epsilonm**(3/2)*f*epsilon2_bulk[l-200]/((epsilon1_bulk[l-200]+2*epsilonm)**2+(epsilon2_bulk[l-200])**2)

#%%
# FITTING

# syntax: (p0 is an appropriate initial guess)
# par_fit, par_cov = curve_fit(function, xdata, ydata, p0=(1, 1, 1))

#%%
# PLOTTING

# plt.figure(figsize=(10, 6), dpi=100)
# plt.plot(l_bulk, epsilon1_bulk, color="green", label=r"$\epsilon_1$")
# plt.plot(l_bulk, epsilon2_bulk, color="red", label=r"$\epsilon_2$")
# plt.title("Bulk gold dielectric functions", fontdict={"fontname": "Calibri", "fontsize": 20})
# plt.xlabel(r"$\lambda$", fontdict={"fontsize": 14})
# plt.ylabel(r"$\epsilon$", fontdict={"fontsize": 14})
# plt.legend(fontsize=10, ncol=2)
# plt.tight_layout()

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(l, absorbance, color="green", label=r"$\epsilon_1$")
plt.plot(l, Absorbance(l, R=10, epsilonm=2, rho=0.4*10**(-9)), color="red", label=r"$\epsilon_1$")  # just a qualitative check
plt.plot(l, Absorbance_JC(l, epsilonm=2, f=2*10**(-6)), color="blue", label=r"$\epsilon_1$") # just a qualitative check
plt.title("Absorbance", fontdict={"fontname": "Calibri", "fontsize": 20})
plt.xlabel(r"$\lambda$", fontdict={"fontsize": 14})
plt.ylabel(r"$\epsilon$", fontdict={"fontsize": 14})
plt.legend(fontsize=10, ncol=2)
plt.tight_layout()

