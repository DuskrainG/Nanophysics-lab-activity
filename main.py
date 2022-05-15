import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import ticker

# skgo
# TODO(EASY):
    # fix units of measure, names, labels ... in graphs
# TODO(MEDIUM):
    # double fits
# TODO(HARD):
    # NOTHING, I already took care of the clusterfuck for Chi-squared maps

wp = 1.37*10**16  #[Hz]
vf = 1.40*10**15  #[nm/s]
Gamma_bulk = 0.476*10**14  #[Hz] # from Ashcroft-Mermin (at 100°C)
c = 299792458*10**9 # [nm/s]
z = 10**7 # [nm]
# guesses of beest fit values from chi-squared maps: use those as initial guesses for the size dependent fits
epsilonm_guess = 2.3
rho_guess = 3.5*10**-9
R_guess = 5
epsilonm_fit = epsilonm_guess
rho_fit = rho_guess
R_fit = R_guess

bulk_data = pd.read_csv("bulk gold dieletric functions.txt", sep="\t", header=None)
bulk_data.columns = ["lambda", "epsilon1", "epsilon2"]
epsilon1_bulk = np.array(bulk_data["epsilon1"]) # from 200nm to 900nm
epsilon2_bulk = np.array(bulk_data["epsilon2"]) # from 200nm to 900nm

np_data = pd.read_csv("G01-NPs.dat", sep="\t", header=None)
np_data.columns = ["lambda", "absorbance"]
absorbance = np.array(np_data["absorbance"])
absorbance_r = absorbance[50:201]  # _r variables are restricted to the selected fit region # I've also tried [70:171]

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

def Chi_R_rho(R, rho):  # only accounts for the chi squared in the selected fit range
    return Chi(absorbance_r, Absorbance(l_r, R, epsilonm_fit, rho))

def Chi_R_epsilonm(R, epsilonm): # only accounts for the chi squared in the selected fit range
    return Chi(absorbance_r, Absorbance(l_r, R, epsilonm, rho_fit))

def multiwrite(outfile, string):
    outfile.write(string + "\n")
    print(string)

#%%
# FITTING: Johnson and Christy

par_fit_JC, par_cov_JC = curve_fit(Absorbance_JC, l, absorbance, p0=(epsilonm_guess, 2*10**-6))
Absorbance_JC_fitted = Absorbance_JC(l, epsilonm=par_fit_JC[0], f=par_fit_JC[1])

par_fit_JC_r, par_cov_JC_r = curve_fit(Absorbance_JC, l_r, absorbance_r, p0=(epsilonm_guess, 2*10**-6))
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
# FITTING: size dependent, initial triple fit

# Initial 3 parameter fit: this can give the initial guess for the more refined fit later, but results are not totally realiable
# Indeed rho*R**3 is constant, so the dependency on R and rho is very weak and there probably are a lot of local minima
par_fit, par_cov = curve_fit(Absorbance, l_r, absorbance_r, p0=(R_guess, epsilonm_guess, rho_guess))
Absorbance_fitted = Absorbance(l, R=par_fit[0], epsilonm=par_fit[1], rho=par_fit[2])

with open("outputfile.txt", "a") as outfile:
    multiwrite(outfile, "Fit restricted to 470nm-570nm:")
    multiwrite(outfile, "R = " + str(par_fit[0]) + " nm, with error " + str(np.sqrt(par_cov[0,0])) + "nm")
    multiwrite(outfile, "espilonm = " + str(par_fit[1]) + " with error " + str(np.sqrt(par_cov[1,1])))
    multiwrite(outfile, "rho = " + str(par_fit[2]) + " nm**-3, with error " + str(np.sqrt(par_cov[2,2])) + "nm**-3")
    multiwrite(outfile, "Chi-squared = " + str(Chi(absorbance, Absorbance_fitted)))
    multiwrite(outfile, "")


#%% FITTING: size dependent, double fits

#TODO:
    # do a fit fixing epsilonm to epsilonm_guess and find (R, rho) pair
        # note: this fit is probably going to give shit because R and rho are obviously correlated
        # so there's a curve of values that basically all give the same fit
        # really we should use the filling fraction, because that is what really matters, together with R
        # but we're asked to do it, so just do it and hope
        # possibly put some soundaries so that we don't get R<1nm (as in the triple fit)
        # if you have time and want to improve the results it might be a good idea to really use (R, f) pair, even if we weren't asked to do so
    # also do the (R, epsilonm) fit fixing rho either to rho_guess or to rho_fit (after updating it)
        # this will give no problems
    # update epsilonm_fit, ... values to the found values
    # do some plots of the results

#%%
# FITTING: Chi squared maps

R_domain_1 = np.arange(2, 15, 0.1)
rho_domain = np.arange(1*10**-10, 15*10**-9, 10**-10)
epsilonm_domain = np.arange(1.4, 3, 0.01)
R_domain_2 = np.arange(3, 7, 0.1)


Chi_R_rho_values = np.zeros((len(R_domain_1), len(rho_domain)))    
for i in range(len(R_domain_1)):
    for j in range(len(rho_domain)):
        Chi_R_rho_values[i, j] = Chi_R_rho(R_domain_1[i], rho_domain[j])

plt.figure(figsize=(10, 6), dpi=100)
contour_plot = plt.contourf(R_domain_1, rho_domain, Chi_R_rho_values.transpose(), 
                            np.logspace(np.log10(Chi_R_rho_values.min()), np.log10(Chi_R_rho_values.max()), 25),
                            locator=ticker.LogLocator(), cmap="plasma")
plt.colorbar(label=r"$\chi^2$")
plt.title("Title", fontdict={"fontname": "Calibri", "fontsize": 20})
plt.xlabel(r"$\lambda$", fontdict={"fontsize": 14})
plt.ylabel(r"$\epsilon$", fontdict={"fontsize": 14})
plt.tight_layout()


Chi_R_epsilonm_values = np.zeros((len(R_domain_2), len(epsilonm_domain)))
for i in range(len(R_domain_2)):
    for j in range(len(epsilonm_domain)):
        Chi_R_epsilonm_values[i, j] = Chi_R_epsilonm(R_domain_2[i], epsilonm_domain[j])
        
plt.figure(figsize=(10, 6), dpi=100)
contour_plot = plt.contourf(R_domain_2, epsilonm_domain, Chi_R_epsilonm_values.transpose(),
                            np.logspace(np.log10(Chi_R_epsilonm_values.min()), np.log10(Chi_R_epsilonm_values.max()), 30),
                            locator=ticker.LogLocator(), cmap="plasma")
plt.colorbar(label=r"$\chi^2$")
plt.title("Title", fontdict={"fontname": "Calibri", "fontsize": 20})
plt.xlabel(r"$\lambda$", fontdict={"fontsize": 14})
plt.ylabel(r"$\epsilon$", fontdict={"fontsize": 14})
plt.tight_layout()

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
# plt.figure(figsize=(10, 6), dpi=100)
# plt.plot(l, absorbance, color="green", label=r"$\epsilon_1$")
# plt.plot(l, Absorbance_fitted, color="blue", label=r"$\epsilon_1$")
# plt.title("Absorbance", fontdict={"fontname": "Calibri", "fontsize": 20})
# plt.xlabel(r"$\lambda$", fontdict={"fontsize": 14})
# plt.ylabel(r"$\epsilon$", fontdict={"fontsize": 14})
# plt.legend(fontsize=10, ncol=2)
# plt.tight_layout()


