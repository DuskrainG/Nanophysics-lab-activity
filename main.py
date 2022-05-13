import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# TODO(EASY):
    # fix units of measure, names, labels ... in graphs
    # find more refined values for bulk constants (Ashcroft Mermin)
# TODO(HARD):
    # Do all the chi-squared shenaningans for Absorbance

wp = 1.37*10**16  #[Hz]
vf = 1.4*10**15  #[nm/s]
Gamma_bulk = 1.08*10**14  #[Hz]  #0.476
c = 299792458*10**9 # [nm/s]
z = 10**7 # [nm]

bulk_data = pd.read_csv("bulk gold dieletric functions.txt", sep="\t", header=None)
bulk_data.columns = ["lambda", "epsilon1", "epsilon2"]
epsilon1_bulk = np.array(bulk_data["epsilon1"]) # from 200nm to 900nm
epsilon2_bulk = np.array(bulk_data["epsilon2"]) # from 200nm to 900nm

np_data = pd.read_csv("G01-NPs.dat", sep="\t", header=None)
np_data.columns = ["lambda", "absorbance"]
absorbance = np.array(np_data["absorbance"])
absorbance_r = absorbance[50:201]  # _r variables are restricted to the selected fit region

l = np.array(np_data["lambda"])  # from 400nm to 800nm
l_bulk = np.array(bulk_data["lambda"]) # from 200nm to 900nm
l_r = l[50:201]
#%%
# FUNCTION DEFINITIONS

def Gamma(R): # [Hz]
    return Gamma_bulk * (1+(np.pi*vf/(4*Gamma_bulk*R)))

def omega(l):  #[Hz]
    return 2*np.pi*c/l

def epsilon1(l, R):
    l=l.astype(int)-200
    try:
        e1 = epsilon1_bulk[l]+wp**2 * (1/(omega(l)**2+Gamma_bulk**2)-1/(omega(l)**2+Gamma(R)**2))
        return e1
    except:
        print("Invalid wavelength")
          
def epsilon2(l, R):
    l=l.astype(int)-200
    try:
        e2 = epsilon2_bulk[l]-wp**2/omega(l) * (Gamma_bulk/(omega(l)**2+Gamma_bulk**2)-Gamma(R)/(omega(l)**2+Gamma(R)**2))
        return e2
    except:
        print("Invalid wavelength")

def Absorbance(l, R, epsilonm, rho): # rho = [nm**(-3)]
    return np.log10(np.e)*9*z*omega(l)/c*epsilonm**(3/2)*4/3*np.pi*R**3*rho*epsilon2(l, R)/((epsilon1(l, R)+2*epsilonm)**2+(epsilon2(l, R))**2)

def Absorbance_JC(l, epsilonm, f):
    l=l.astype(int)-200
    return np.log10(np.e)*9*z*omega(l)/c*epsilonm**(3/2)*f*epsilon2_bulk[l]/((epsilon1_bulk[l]+2*epsilonm)**2+(epsilon2_bulk[l])**2)

def Chi(observed, expected):
    return ((observed-expected)**2/expected).sum()

def multiwrite(outfile, string):
    outfile.write(string + "\n")
    print(string)

#%%
# FITTING: Johnson and Christy

par_fit_JC, par_cov_JC = curve_fit(Absorbance_JC, l, absorbance, p0=(2, 2*10**-6))
Absorbance_JC_fitted = Absorbance_JC(l, epsilonm=par_fit_JC[0], f=par_fit_JC[1])

par_fit_JC_r, par_cov_JC_r = curve_fit(Absorbance_JC, l_r, absorbance_r, p0=(2, 2*10**-6))
Absorbance_JC_fitted_r = Absorbance_JC(l, epsilonm=par_fit_JC_r[0], f=par_fit_JC_r[1])

with open("outputfile.txt", "w") as outfile:
    multiwrite(outfile, "Full curve JC fit:")
    multiwrite(outfile, "espilonm_JC = " + str(par_fit_JC[0]) + " with error " + str(np.sqrt(par_cov_JC[0,0])))
    multiwrite(outfile, "f_JC = " + str(par_fit_JC[1]) + " with error " + str(np.sqrt(par_cov_JC[1,1])))
    multiwrite(outfile, "Chi-squared = " + str(Chi(absorbance, Absorbance_JC_fitted)))
    multiwrite(outfile, "")
    
    multiwrite(outfile, "Restricting the JC fit to 470nm-570nm:")
    multiwrite(outfile, "espilonm_JC = " + str(par_fit_JC_r[0]) + " with error " + str(np.sqrt(par_cov_JC_r[0,0])))
    multiwrite(outfile, "f_JC = " + str(par_fit_JC_r[1]) + " with error " + str(np.sqrt(par_cov_JC_r[1,1])))
    multiwrite(outfile, "Chi-squared = " + str(Chi(absorbance, Absorbance_JC_fitted_r)))
    multiwrite(outfile, "")
    
#%%
# FITTING: Size dependent

# Initial 3 parameter fit: this can give the initial guess for the more refined fit later, but results are not totally realiable
# Indeed rho*R**3 is constant, so the dependency on R and rho is very weak and there probably are a lot of local minima
par_fit, par_cov = curve_fit(Absorbance, l_r, absorbance_r, p0=(10, 2, 3*10**-9))
Absorbance_fitted = Absorbance(l, R=par_fit[0], epsilonm=par_fit[1], rho=par_fit[2])

with open("outputfile.txt", "a") as outfile:
    multiwrite(outfile, "Fit restricted to 470nm-570nm:")
    multiwrite(outfile, "R = " + str(par_fit[0]) + " nm, with error " + str(np.sqrt(par_cov[0,0])) + "nm")
    multiwrite(outfile, "espilonm = " + str(par_fit[1]) + " with error " + str(np.sqrt(par_cov[1,1])))
    multiwrite(outfile, "rho = " + str(par_fit[2]) + " nm**-3, with error " + str(np.sqrt(par_cov[2,2])) + "nm**-3")
    multiwrite(outfile, "Chi-squared = " + str(Chi(absorbance, Absorbance_fitted)))
    multiwrite(outfile, "")

#%%
# PLOTTING

# Bulk gold dielectric functions
# plt.figure(figsize=(10, 6), dpi=100)
# plt.plot(l_bulk, epsilon1_bulk, color="green", label=r"$\epsilon_1$")
# plt.plot(l_bulk, epsilon2_bulk, color="red", label=r"$\epsilon_2$")
# plt.title("Bulk gold dielectric functions", fontdict={"fontname": "Calibri", "fontsize": 20})
# plt.xlabel(r"$\lambda$", fontdict={"fontsize": 14})
# plt.ylabel(r"$\epsilon$", fontdict={"fontsize": 14})
# plt.legend(fontsize=10, ncol=2)
# plt.tight_layout()

# JC plots
# plt.figure(figsize=(10, 6), dpi=100)
# plt.plot(l, absorbance, color="green", label=r"$\epsilon_1$")
# plt.plot(l, Absorbance_JC_fitted, color="blue", label=r"$\epsilon_1$")
# plt.plot(l, Absorbance_JC_fitted_r, color="red", label=r"$\epsilon_1$")
# plt.title("Absorbance", fontdict={"fontname": "Calibri", "fontsize": 20})
# plt.xlabel(r"$\lambda$", fontdict={"fontsize": 14})
# plt.ylabel(r"$\epsilon$", fontdict={"fontsize": 14})
# plt.legend(fontsize=10, ncol=2)
# plt.tight_layout()

# Size dependent plots
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(l, absorbance, color="green", label=r"$\epsilon_1$")
plt.plot(l, Absorbance_fitted, color="blue", label=r"$\epsilon_1$")
plt.title("Absorbance", fontdict={"fontname": "Calibri", "fontsize": 20})
plt.xlabel(r"$\lambda$", fontdict={"fontsize": 14})
plt.ylabel(r"$\epsilon$", fontdict={"fontsize": 14})
plt.legend(fontsize=10, ncol=2)
plt.tight_layout()


